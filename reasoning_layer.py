"""
LLM Reasoning Layer for the Asset Repair vs. Replace Decision Engine.
(Groq version -- free tier, no credit card required)

For each asset, this pulls:
  1. The rule-engine's recommendation + which rules fired (from decision_engine.py)
  2. Other assets in the same equipment CATEGORY (e.g. "Rooftop Unit"),
     as "similar asset" context

...and asks an LLM (via Groq) to write a plain-English maintenance explanation
grounded in both, storing the result back on the Asset node as `llm_explanation`.

NOTE ON SIMILARITY (fixed from the original version of this file): matching
on identical (manufacturer, model) via the shared EquipmentType node
returns ZERO results for this dataset -- every one of the 10 assets has a
unique manufacturer/model combination, so no two assets ever point at the
same EquipmentType node. The "similar asset" context was therefore always
empty. This version instead matches on equipment CATEGORY, parsed as the
text before " - " in EquipmentType.type_name (e.g. "Rooftop Unit - Supply
Air" -> "Rooftop Unit"), which actually finds matches: RTU-V01/RTU-V08,
FCU-V03/FCU-V07, CU-V05/CU-V10 in the current 10-asset dataset.

NOTE ON COST FIELDS: now includes maintenance_labor_spend,
parts_materials_spend, and total_cost_of_ownership from the Cost node
(added after the original Neo4j backfill), alongside repair/replacement/
purchase cost, so the LLM has the full financial picture available when
reasoning -- not just the three fields the original version exposed.

NOTE ON RULES: Rules 4 and 8 in decision_engine.py still use substitute
fields (baseline_energy_kwh_per_ton, Repair.status) because new-unit
energy specs and warranty status were never loaded into this graph.
Rules 3 and 6 use the real major_failure_count and criticality fields.
The prompt below reflects that -- it references major failures and
criticality directly.

Run locally where your Neo4j Desktop instance is reachable:
    pip install neo4j groq
    export GROQ_API_KEY="your-key-here"
    python reasoning_layer_groq.py

Get a free Groq API key (no card required) at: https://console.groq.com/keys
"""

import os
from neo4j import GraphDatabase
from groq import Groq

# ---- connection settings: match your "AssetGraphV2" instance ----
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your neo4j pass"   # <-- set this

# Groq API key is read from the GROQ_API_KEY environment variable.
# Set it in your terminal before running:  export GROQ_API_KEY="gsk_..."
MODEL = "llama-3.3-70b-versatile"

TOP_K_SIMILAR = 3


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

# Matches on equipment CATEGORY (text before " - " in type_name), not on
# sharing the literal same EquipmentType node -- see module docstring.
SIMILAR_ASSETS_QUERY = """
MATCH (a:Asset {assetId: $asset_id})-[:HAS_TYPE]->(et:EquipmentType)
WITH a, split(et.type_name, ' - ')[0] AS category
MATCH (other:Asset)-[:HAS_TYPE]->(other_et:EquipmentType)
WHERE other.assetId <> $asset_id
  AND split(other_et.type_name, ' - ')[0] = category
OPTIONAL MATCH (other)-[:UNDERGOES]->(r:Repair)
RETURN other.assetId AS assetId,
       r.issue_description AS issue_description,
       other.engine_recommendation AS engine_recommendation
LIMIT $k
"""


def fetch_asset_with_context(session, asset_id):
    """
    Pull one asset's full properties plus its EquipmentType, Repair, and
    Cost neighbors, and its top-K similar assets (same equipment category)
    -- everything the LLM needs to reason about this asset without
    hallucinating details.
    """
    record = session.run(ASSET_CONTEXT_QUERY, asset_id=asset_id).single()
    if record is None:
        return None, None

    asset = dict(record["asset"] or {})
    equipment_type = record["equipment_type"] or {}
    repair = record["repair"] or {}
    cost = record["cost"] or {}

    # Flatten what build_prompt needs onto one dict, using the real
    # property names confirmed from the live graph.
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
    asset["maintenance_labor_spend"] = cost.get("maintenance_labor_spend")
    asset["parts_materials_spend"] = cost.get("parts_materials_spend")
    asset["total_cost_of_ownership"] = cost.get("total_cost_of_ownership")

    similar = session.run(SIMILAR_ASSETS_QUERY, asset_id=asset_id, k=TOP_K_SIMILAR).data()

    return asset, similar


def build_prompt(asset, similar_assets):
    triggered_rules = asset.get("engine_triggered_rules") or "None"
    recommendation = asset.get("engine_recommendation") or "Unknown"
    category = (asset.get("equipment_type") or "").split(" - ")[0] or "this equipment type"

    similar_text = "\n".join(
        f"  - {s['assetId']}: issue was \"{s['issue_description'] or 'not recorded'}\", "
        f"recommendation was {s['engine_recommendation'] or 'not yet evaluated'}"
        for s in similar_assets
    ) or f"  (no other {category} assets in the fleet yet)"

    def fmt_cost(value):
        return f"${value:,.0f}" if value is not None else "not recorded"

    prompt = f"""You are a facilities maintenance advisor for an airport HVAC fleet.
Write a short, plain-English explanation for a maintenance technician about
why this asset was recommended for {recommendation}.

ASSET DETAILS:
- Asset ID: {asset.get('assetId')}
- Equipment: {asset.get('equipment_type')} ({asset.get('manufacturer')} {asset.get('model')})
- Location: {asset.get('physical_location')}
- Condition rating: {asset.get('condition_rating', 'not recorded')}/5
- Work order count: {asset.get('work_order_count', 'not recorded')}
- Major failure count (last 24 months): {asset.get('major_failure_count', 'not recorded')}
- Criticality: {asset.get('criticality', 'not recorded')}
- Served area: {asset.get('served_area', 'not recorded')}
- Current energy use: {asset.get('current_energy_kwh_per_ton', 'not recorded')} kWh/ton
  (equipment baseline: {asset.get('baseline_energy_kwh_per_ton', 'not recorded')} kWh/ton)
- Design life: {asset.get('design_life_years', 'not recorded')} years
- MTBF: {asset.get('mtbf_days', 'not recorded')} days
- Current repair issue: {asset.get('issue_description', 'none open')}
- Repair status: {asset.get('repair_status', 'not recorded')}
- Warranty status: {'Expired' if asset.get('warranty_expired') else ('Active' if asset.get('warranty_expired') is False else 'not recorded')}
- Flagged for review (Rule 8): {'Yes' if asset.get('engine_flagged_for_review') else 'No'}

COST BREAKDOWN:
- Repair cost estimate: {fmt_cost(asset.get('repair_cost'))}
- Replacement cost estimate: {fmt_cost(asset.get('replacement_cost'))}
- Original purchase cost: {fmt_cost(asset.get('purchase_cost'))}
- Cumulative maintenance labor spend: {fmt_cost(asset.get('maintenance_labor_spend'))}
- Cumulative parts & materials spend: {fmt_cost(asset.get('parts_materials_spend'))}
- Total cost of ownership to date: {fmt_cost(asset.get('total_cost_of_ownership'))}

DECISION ENGINE OUTPUT:
- Recommendation: {recommendation}
- Triggered rules: {triggered_rules}

OTHER {category.upper()} ASSETS IN THE FLEET:
{similar_text}

Write 3-4 sentences a technician could read in the field. Name every rule
number listed in "Triggered rules" above at least briefly -- do not
silently drop any of them, even if you can't elaborate on each one in
depth. If a similar asset's outcome is relevant, mention it briefly as
supporting context. You may reference the cost breakdown (e.g. total cost
of ownership vs. purchase price) if it strengthens the explanation. Do not
invent facts not present above. Do not use bullet points -- write it as
flowing prose."""

    return prompt


def generate_explanation(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def write_explanation(tx, asset_id, explanation):
    tx.run(
        """
        MATCH (a:Asset {assetId: $asset_id})
        SET a.llm_explanation = $explanation,
            a.llm_model = $model,
            a.llm_generated_at = datetime()
        """,
        asset_id=asset_id,
        explanation=explanation,
        model=MODEL,
    )


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set the GROQ_API_KEY environment variable first.")
        print('  export GROQ_API_KEY="your-key-here"')
        print("  Get a free key (no card required) at https://console.groq.com/keys")
        return

    client = Groq(api_key=api_key)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        asset_ids = [
            r["assetId"] for r in session.run(
                "MATCH (a:Asset) RETURN a.assetId AS assetId ORDER BY a.assetId"
            )
        ]
        total = len(asset_ids)
        print(f"Generating explanations for {total} assets using Groq ({MODEL})...\n")

        for i, asset_id in enumerate(asset_ids, start=1):
            print(f"[{i}/{total}] {asset_id}")
            asset, similar = fetch_asset_with_context(session, asset_id)
            if asset is None:
                print("    (asset not found, skipping)\n")
                continue

            prompt = build_prompt(asset, similar)
            explanation = generate_explanation(client, prompt)

            print(f"    {explanation}\n")

            session.execute_write(write_explanation, asset_id, explanation)

    driver.close()
    print("Done. Each asset now has a.llm_explanation stored in Neo4j.")


if __name__ == "__main__":
    main()
