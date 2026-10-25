"""
Interactive Q&A over the HVAC asset graph.

Lets you ask any free-form question. It looks for a named asset ID or
matching keywords, pulls the matching assets' full rule-engine output, and
asks the LLM to answer using only that retrieved context.

NOTE ON RETRIEVAL: the original design of this file used embedding-based
vector search over :Document nodes (sentence-transformers + a Neo4j vector
index). That infrastructure doesn't exist in this Neo4j instance -- the
graph only has Asset, EquipmentType, Repair, and Cost nodes, no Document
nodes and no vector index. Retrieval here is therefore keyword-based
instead: it matches question terms against issue_description, findings,
equipment type name, and physical_location. If you build out the
Document/embedding layer later, swap retrieve_relevant_assets() below for
a vector-search version -- build_qa_prompt and ask_llm don't need to change.

Run locally where your Neo4j Desktop instance is reachable:
    pip install neo4j groq
    export GROQ_API_KEY="your-key-here"
    python ask.py
"""

import os
import re
from neo4j import GraphDatabase
from groq import Groq

# ---- connection settings: match your "AssetGraphV2" instance ----
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your neo4j pass"   # <-- set this

MODEL = "llama-3.3-70b-versatile"
TOP_K = 4  # how many assets to retrieve per question when falling back / keyword-matching

# Matches asset IDs like AHU-V02-BLDG01, WCC-V09-BLDG04, CU-V10-BLDG03, etc.
ASSET_ID_PATTERN = re.compile(r"\b[A-Za-z]{2,5}-[A-Za-z0-9]{2,6}-[A-Za-z0-9]{2,10}\b")

# Common words to ignore when pulling keywords out of a free-form question
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "how", "much", "many", "will", "it", "take", "to", "for", "of", "on",
    "in", "with", "and", "or", "should", "do", "does", "did", "can", "could",
    "would", "has", "have", "had", "this", "that", "be", "about",
}


def find_named_asset_ids(question):
    """
    If the question explicitly names an asset ID (e.g. 'what's wrong with
    WCC-V09-BLDG04'), pull those out so we can fetch them directly instead
    of relying on keyword overlap to happen to surface them.
    """
    return [m.upper() for m in ASSET_ID_PATTERN.findall(question)]


def extract_keywords(question):
    words = re.findall(r"[a-zA-Z]+", question.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


ASSET_CONTEXT_QUERY = """
MATCH (a:Asset {assetId: $asset_id})
OPTIONAL MATCH (a)-[:HAS_TYPE]->(et:EquipmentType)
OPTIONAL MATCH (a)-[:UNDERGOES]->(r:Repair)
OPTIONAL MATCH (a)-[:HAS_COST]->(c:Cost)
RETURN properties(a) AS asset,
       properties(et) AS equipment_type,
       properties(r) AS repair,
       properties(c) AS cost
"""

KEYWORD_SEARCH_QUERY = """
MATCH (a:Asset)
OPTIONAL MATCH (a)-[:HAS_TYPE]->(et:EquipmentType)
OPTIONAL MATCH (a)-[:UNDERGOES]->(r:Repair)
OPTIONAL MATCH (a)-[:HAS_COST]->(c:Cost)
WITH a, et, r, c,
     toLower(coalesce(r.issue_description, '') + ' ' + coalesce(r.findings, '') + ' ' +
             coalesce(et.type_name, '') + ' ' + coalesce(a.physical_location, '') + ' ' +
             coalesce(a.criticality, '') + ' ' + coalesce(a.served_area, '')) AS haystack
WHERE ANY(kw IN $keywords WHERE haystack CONTAINS kw)
RETURN properties(a) AS asset,
       properties(et) AS equipment_type,
       properties(r) AS repair,
       properties(c) AS cost
LIMIT $k
"""

# Fallback for open-ended questions with no named asset and no keyword hit
# (e.g. "which assets should be replaced first?") -- surface the assets
# already flagged REPLACE, worst condition first.
FALLBACK_QUERY = """
MATCH (a:Asset)
OPTIONAL MATCH (a)-[:HAS_TYPE]->(et:EquipmentType)
OPTIONAL MATCH (a)-[:UNDERGOES]->(r:Repair)
OPTIONAL MATCH (a)-[:HAS_COST]->(c:Cost)
RETURN properties(a) AS asset,
       properties(et) AS equipment_type,
       properties(r) AS repair,
       properties(c) AS cost
ORDER BY CASE asset.engine_recommendation WHEN 'REPLACE' THEN 0 ELSE 1 END,
         coalesce(asset.condition_rating, 5) ASC
LIMIT $k
"""


def flatten_record(record):
    """Turn one ASSET_CONTEXT_QUERY-shaped row into a single flat dict."""
    asset = dict(record["asset"] or {})
    equipment_type = record["equipment_type"] or {}
    repair = record["repair"] or {}
    cost = record["cost"] or {}

    asset["equipment_type"] = equipment_type.get("type_name")
    asset["manufacturer"] = equipment_type.get("manufacturer")
    asset["model"] = equipment_type.get("model")
    asset["design_life_years"] = equipment_type.get("design_life_years")
    asset["mtbf_days"] = equipment_type.get("mtbf_days")
    asset["baseline_energy_kwh_per_ton"] = equipment_type.get("baseline_energy_kwh_per_ton")
    asset["issue_description"] = repair.get("issue_description")
    asset["findings"] = repair.get("findings")
    asset["repair_status"] = repair.get("status")
    asset["repair_cost"] = cost.get("repair_cost")
    asset["replacement_cost"] = cost.get("replacement_cost")
    asset["purchase_cost"] = cost.get("purchase_cost")
    return asset


def fetch_asset_by_id(session, asset_id):
    record = session.run(ASSET_CONTEXT_QUERY, asset_id=asset_id).single()
    return flatten_record(record) if record and record["asset"] else None


def retrieve_relevant_assets(session, question, carry_over_ids=None):
    """
    Three-part retrieval, in priority order:
      1. Exact match -- if the question names a specific asset ID, fetch it
         directly (guaranteed to be included, score 1.0). If it names no ID
         but the conversation was already focused on one (carry_over_ids,
         e.g. from the previous turn), stay on that asset instead --
         otherwise a follow-up like "how much will it take" has nothing to
         anchor to.
      2. Keyword search -- match question terms against issue/findings/
         equipment-type/location text, to surface relevant assets even when
         no ID is named (e.g. "which assets have compressor problems?").
      3. Fallback -- if nothing matched, surface the assets most likely to
         be relevant to a general fleet question (REPLACE-flagged, worst
         condition first).
    Results are merged and deduplicated by assetId.
    """
    results = []
    seen_ids = set()

    ids_to_fetch = find_named_asset_ids(question)
    if not ids_to_fetch and carry_over_ids:
        ids_to_fetch = list(carry_over_ids)

    for asset_id in ids_to_fetch:
        asset = fetch_asset_by_id(session, asset_id)
        if asset and asset.get("assetId") not in seen_ids:
            results.append({"props": asset, "score": 1.0})
            seen_ids.add(asset.get("assetId"))

    keywords = extract_keywords(question)
    if keywords:
        keyword_records = session.run(KEYWORD_SEARCH_QUERY, keywords=keywords, k=TOP_K).data()
        for r in keyword_records:
            asset = flatten_record(r)
            if asset.get("assetId") not in seen_ids:
                results.append({"props": asset, "score": 0.75})
                seen_ids.add(asset.get("assetId"))

    if not results:
        fallback_records = session.run(FALLBACK_QUERY, k=TOP_K).data()
        for r in fallback_records:
            asset = flatten_record(r)
            if asset.get("assetId") not in seen_ids:
                results.append({"props": asset, "score": 0.5})
                seen_ids.add(asset.get("assetId"))

    return results


def build_qa_prompt(question, retrieved, history=None):
    context_blocks = []
    for r in retrieved:
        a = r["props"]
        score = r["score"]
        recommendation = a.get("engine_recommendation") or "Not yet evaluated"
        triggered = a.get("engine_triggered_rules") or "None"
        context_blocks.append(
            f"""Asset {a.get('assetId')} (relevance: {score:.2f}):
  Equipment: {a.get('equipment_type')} ({a.get('manufacturer')} {a.get('model')})
  Location: {a.get('physical_location')}
  Installed: {a.get('install_date', 'not recorded')}
  Design service life: {a.get('design_life_years', 'not recorded')} years
  Mean time between failures: {a.get('mtbf_days', 'not recorded')} days
  Current issue: {a.get('issue_description')}
  Findings: {a.get('findings')}
  Condition rating: {a.get('condition_rating')}/5
  Work order count: {a.get('work_order_count', 'not recorded')}
  Major failure count (last 24 months): {a.get('major_failure_count', 'not recorded')}
  Criticality: {a.get('criticality', 'not recorded')}
  Served area: {a.get('served_area', 'not recorded')}
  Repair cost: {a.get('repair_cost')} | Replacement cost: {a.get('replacement_cost')}
  Recommendation: {recommendation}
  Triggered rules: {triggered}"""
        )
    context = "\n\n".join(context_blocks)

    history_text = ""
    if history:
        turns = "\n".join(
            f"{'Technician' if h.get('role') == 'user' else 'Advisor'}: {h.get('content', '')}"
            for h in history
        )
        history_text = f"""
CONVERSATION SO FAR (use this to resolve references like "it" or "that asset"):
{turns}
"""

    prompt = f"""You are a facilities maintenance advisor for an airport HVAC fleet.
Answer the technician's question using ONLY the asset data below and the conversation
so far. If the data doesn't fully answer the question, say what's missing rather than
guessing. Cite specific asset IDs when relevant. Keep the answer concise and
practical.

IMPORTANT: When you reference which rules triggered a recommendation, list
every rule number present in that asset's "Triggered rules" field exactly
as given -- do not summarize, paraphrase, or select a subset of them. If
you don't have space to explain each one, still name all the rule numbers
that fired, even briefly.
{history_text}
RETRIEVED ASSET DATA:
{context}

QUESTION: {question}

ANSWER:"""
    return prompt


def ask_llm(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def format_rule_footer(retrieved):
    """
    Build a footer listing each retrieved asset's real triggered_rules
    verbatim from the decision engine. LLMs (especially smaller/free
    models) don't reliably enumerate every rule in freeform prose even
    when instructed to -- this guarantees the technician always sees the
    ground truth regardless of what the model chose to summarize.
    Only includes assets that actually have rule data worth showing.
    """
    lines = []
    for r in retrieved:
        a = r["props"]
        rules = a.get("engine_triggered_rules")
        if not rules or rules == "None":
            continue
        lines.append(f"  {a.get('assetId')}: {rules}")
    if not lines:
        return ""
    return "\n\nTriggered rules (from decision engine, verbatim):\n" + "\n".join(lines)


def answer_with_rule_footer(client, question, retrieved, history=None):
    """Full answer = LLM prose + the deterministic rule footer appended."""
    prompt = build_qa_prompt(question, retrieved, history=history)
    answer = ask_llm(client, prompt)
    footer = format_rule_footer(retrieved)
    return answer + footer if footer else answer


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set the GROQ_API_KEY environment variable first.")
        print('  export GROQ_API_KEY="your-key-here"')
        return

    client = Groq(api_key=api_key)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("\nHVAC Fleet Q&A -- ask anything about your assets.")
    print("Examples: 'Which assets should be replaced first?' or")
    print("          'What's wrong with the chiller?' or 'quit' to exit.\n")

    focus_ids = []  # assets the conversation is currently "about" -- carries
                     # forward across turns so a follow-up like "how much will
                     # it take" without naming an asset stays on the last one(s)
    history = []

    with driver.session() as session:
        while True:
            question = input("Ask> ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            retrieved = retrieve_relevant_assets(session, question, carry_over_ids=focus_ids)

            print(f"\n  (retrieved {len(retrieved)} relevant assets: "
                  f"{', '.join(r['props'].get('assetId') for r in retrieved)})\n")

            prompt = build_qa_prompt(question, retrieved, history=history)
            answer = ask_llm(client, prompt)
            footer = format_rule_footer(retrieved)
            full_answer = answer + footer if footer else answer
            print(f"{full_answer}\n")
            print("-" * 80 + "\n")

            # Only refocus onto exactly-named assets -- a keyword/fallback
            # match (score < 1.0) shouldn't hijack focus away from what the
            # technician actually asked about by name.
            named_ids = [r["props"].get("assetId") for r in retrieved if r["score"] == 1.0]
            if named_ids:
                focus_ids = named_ids

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

    driver.close()
    print("Session ended.")


if __name__ == "__main__":
    main()
