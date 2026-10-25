"""
Flask backend for the HVAC Fleet dashboard chat box (index.html).

Wraps ask.py's retrieval + reasoning logic in two HTTP endpoints that
index.html already expects:
    GET  /health  -> liveness check (drives the chat status dot)
    POST /ask     -> { "question": "..." }
                      -> { "answer": "...", "retrieved_tags": [...] }

Also exposes /asset-graph/<assetId> for the dashboard's Graph Explorer
panel, traversing the actual graph shape: Asset-[:HAS_TYPE]->EquipmentType,
Asset-[:UNDERGOES]->Repair, Asset-[:HAS_COST]->Cost,
Repair-[:HAS_COST]->Cost.

Run alongside your Neo4j Desktop instance:
    pip install flask flask-cors neo4j groq
    export GROQ_API_KEY="your-key-here"
    python server.py

index.html's CHAT_BACKEND constant points at http://127.0.0.1:5001,
so just leave this running in a terminal while you use the dashboard.
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase
from groq import Groq

from ask import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    retrieve_relevant_assets,
    build_qa_prompt,
    ask_llm,
    format_rule_footer,
)

app = Flask(__name__)
CORS(app)  # index.html is opened as a local file / different origin, so allow it

_state = {}


def sanitize_props(props):
    """
    Neo4j returns some property values (like Date objects on install_date,
    or the DateTime objects written by decision_engine.py's `datetime()`
    and reasoning_layer_groq.py's `datetime()`) as native neo4j.time types,
    which Flask's jsonify can't serialize. Convert anything that isn't a
    plain JSON-safe type to a string.
    """
    if props is None:
        return None
    sanitized = {}
    for key, value in props.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [
                item if isinstance(item, (str, int, float, bool)) or item is None else str(item)
                for item in value
            ]
        else:
            sanitized[key] = str(value)
    return sanitized


def init_backend():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            'Set the GROQ_API_KEY environment variable first: export GROQ_API_KEY="your-key-here"'
        )

    _state["client"] = Groq(api_key=api_key)
    _state["driver"] = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("Backend ready -- listening on http://127.0.0.1:5001")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/asset-graph/<asset_id>", methods=["GET"])
def asset_graph(asset_id):
    """
    Traverses the actual graph relationships for one asset:
        Asset -[:HAS_TYPE]-> EquipmentType
        Asset -[:UNDERGOES]-> Repair -[:HAS_COST]-> Cost
        Asset -[:HAS_COST]-> Cost
    Used by the "Graph Explorer" view in index.html to render the actual
    node/edge structure, not just the flattened asset properties.
    """
    asset_id = asset_id.strip().upper()
    query = """
    MATCH (a:Asset {assetId: $asset_id})
    OPTIONAL MATCH (a)-[:HAS_TYPE]->(et:EquipmentType)
    OPTIONAL MATCH (a)-[:UNDERGOES]->(r:Repair)
    OPTIONAL MATCH (a)-[:HAS_COST]->(c:Cost)
    RETURN properties(a) AS asset,
           properties(et) AS equipment_type,
           properties(r) AS repair,
           properties(c) AS cost
    """
    try:
        with _state["driver"].session() as session:
            record = session.run(query, asset_id=asset_id).single()

        if not record or not record["asset"]:
            return jsonify({"error": f"No asset found with assetId {asset_id}"}), 404

        return jsonify(
            {
                "asset": sanitize_props(record["asset"]),
                "equipment_type": sanitize_props(record["equipment_type"]),
                "repair": sanitize_props(record["repair"]),
                "cost": sanitize_props(record["cost"]),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    history = data.get("history") or []
    focus_ids = data.get("focus_tags") or data.get("focus_ids") or []
    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        with _state["driver"].session() as session:
            retrieved = retrieve_relevant_assets(
                session, question, carry_over_ids=focus_ids
            )

            if not retrieved:
                return jsonify(
                    {
                        "answer": "I couldn't find any assets matching that question.",
                        "retrieved_tags": [],
                    }
                )

            prompt = build_qa_prompt(question, retrieved, history=history)
            answer = ask_llm(_state["client"], prompt)
            footer = format_rule_footer(retrieved)
            full_answer = answer + footer if footer else answer
            asset_ids = [r["props"].get("assetId") for r in retrieved]

        return jsonify({"answer": full_answer, "retrieved_tags": asset_ids})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    init_backend()
    app.run(host="127.0.0.1", port=5001, debug=False)