"""
Export all Asset data from Neo4j into a static assets.json file for the
dashboard -- no backend needed, the dashboard just fetches this file.

Also computes a floor-plan (x, y) position for each asset by parsing its
`physical_location` property (e.g. "Bldg 2, Roof West, Zone 4"), so the
dashboard can plot assets on a blueprint-style building map without
needing real GPS coordinates.

NOTE ON OUTPUT SHAPE: index2.html reads assets.json expecting keys like
`tag`, `equipment_type`, `location`, `defect`, `findings`, `criticality`,
`served_area`, `lifecycle_phase`, etc. This graph only has 4 node types
(Asset, EquipmentType, Repair, Cost) and doesn't carry criticality,
served_area, or lifecycle_phase anywhere -- those are exported as null
below rather than invented. index2.html already renders missing fields as
"Not recorded", so the dashboard degrades gracefully instead of breaking.

Run locally where your Neo4j Desktop instance is reachable:
    pip install neo4j
    python export_to_json.py
"""

import json
import re
from neo4j import GraphDatabase

# ---- connection settings: match your "AssetGraphV2" instance ----
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your neo4j pass"   # <-- set this

OUTPUT_PATH = "assets.json"

# Each building occupies a quadrant of a 1000x1000 floor-plan grid.
# Adjust these if you add buildings 5+ later.
BUILDING_QUADRANTS = {
    1: (0, 0),      # top-left
    2: (500, 0),    # top-right
    3: (0, 500),    # bottom-left
    4: (500, 500),  # bottom-right
}

# Zones are spread within a building's quadrant in a simple 5-slot ring,
# so assets in the same building don't all stack on one point.
ZONE_OFFSETS = {
    1: (100, 100),
    2: (350, 130),
    3: (400, 350),
    4: (150, 400),
    5: (250, 250),
}


def parse_location(location_text):
    """
    'Bldg 2, Roof West, Zone 4' -> (building=2, sub_location='Roof West', zone=4)
    'Bldg 3, Floor 1, Baggage Handling Area' -> (building=3, sub_location='Floor 1, Baggage Handling Area', zone=1)
    """
    if not location_text:
        return 1, "Unknown", 1

    building_match = re.search(r"Bldg\s*(\d+)", location_text)
    building = int(building_match.group(1)) if building_match else 1

    zone_match = re.search(r"Zone\s*(\d+)", location_text)
    zone = int(zone_match.group(1)) if zone_match else 1

    # sub_location = everything between "Bldg N, " and ", Zone N" (or end of string)
    sub = location_text
    sub = re.sub(r"Bldg\s*\d+,?\s*", "", sub)
    sub = re.sub(r",?\s*Zone\s*\d+", "", sub)
    sub_location = sub.strip()

    return building, sub_location, zone


def compute_position(building, zone):
    qx, qy = BUILDING_QUADRANTS.get(building, (0, 0))
    ox, oy = ZONE_OFFSETS.get(zone, (250, 250))
    return qx + ox, qy + oy


FETCH_QUERY = """
MATCH (a:Asset)
OPTIONAL MATCH (a)-[:HAS_TYPE]->(et:EquipmentType)
OPTIONAL MATCH (a)-[:UNDERGOES]->(r:Repair)
OPTIONAL MATCH (a)-[:HAS_COST]->(c:Cost)
RETURN properties(a) AS asset,
       properties(et) AS equipment_type,
       properties(r) AS repair,
       properties(c) AS cost
"""


def fetch_assets(driver):
    with driver.session() as session:
        result = session.run(FETCH_QUERY)
        rows = [record.data() for record in result]

    # Sort by the asset's creation number (the "V01".."V10" in its assetId)
    # rather than alphabetically -- sorting alphabetically would sort by
    # equipment-type prefix first (e.g. "CU" before "FCU"), which changes
    # which asset lands first in a shared building/zone bucket and
    # therefore which one gets jittered. Sorting by creation order keeps
    # jitter placement stable and matches the original floor-plan layout.
    def asset_number(row):
        asset_id = (row["asset"] or {}).get("assetId", "")
        match = re.search(r"-V(\d+)-", asset_id)
        return int(match.group(1)) if match else 999

    rows.sort(key=asset_number)
    return rows


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    rows = fetch_assets(driver)
    driver.close()

    output = []
    bucket_counts = {}  # (building, zone) -> how many assets already placed there

    for row in rows:
        asset = row["asset"] or {}
        equipment_type = row["equipment_type"] or {}
        repair = row["repair"] or {}
        cost = row["cost"] or {}

        location = asset.get("physical_location", "")
        building, sub_location, zone = parse_location(location)
        x, y = compute_position(building, zone)

        # Jitter assets that land in the same building/zone bucket so they
        # don't stack exactly on top of each other on the map.
        bucket = (building, zone)
        count = bucket_counts.get(bucket, 0)
        bucket_counts[bucket] = count + 1
        if count > 0:
            radius = 18 + count * 10
            x += round(radius * (0.8 if count % 2 == 0 else -0.8))
            y += round(radius * (0.6 if count % 3 == 0 else -0.6))

        recommendation = asset.get("engine_recommendation") or "UNKNOWN"
        condition_rating = asset.get("condition_rating")

        output.append({
            "tag": asset.get("assetId"),
            "equipment_type": equipment_type.get("type_name"),
            "manufacturer": equipment_type.get("manufacturer"),
            "model": equipment_type.get("model"),
            "location": location,
            "building": building,
            "sub_location": sub_location,
            "zone": zone,
            "x": x,
            "y": y,
            "lifecycle_phase": None,  # still not present anywhere in this graph
            "criticality": asset.get("criticality"),
            "served_area": asset.get("served_area"),
            "major_failure_count": asset.get("major_failure_count"),
            "warranty_expired": asset.get("warranty_expired"),
            "warranty_expiry_date": asset.get("warranty_expiry_date"),
            "defect": repair.get("issue_description"),
            "findings": repair.get("findings"),
            "repair_status": repair.get("status"),
            "inspection_result": f"{condition_rating}/5" if condition_rating is not None else None,
            "condition_rating": condition_rating,
            "work_order_count": asset.get("work_order_count"),
            "design_life_years": equipment_type.get("design_life_years"),
            "mtbf_days": equipment_type.get("mtbf_days"),
            "repair_cost": cost.get("repair_cost"),
            "replacement_cost": cost.get("replacement_cost"),
            "purchase_cost": cost.get("purchase_cost"),
            # Added alongside the Neo4j Cost-node backfill (maintenance
            # labor, parts/materials, and running total cost of ownership)
            # so the map's detail panel can show the same full cost
            # picture as the Graph Explorer instead of a reduced subset.
            "maintenance_labor_spend": cost.get("maintenance_labor_spend"),
            "parts_materials_spend": cost.get("parts_materials_spend"),
            "total_cost_of_ownership": cost.get("total_cost_of_ownership"),
            "recommendation": recommendation,
            "triggered_rules": asset.get("engine_triggered_rules"),
            "flagged_for_review": asset.get("engine_flagged_for_review", False),
            "llm_explanation": asset.get("llm_explanation"),
            "llm_model": asset.get("llm_model"),
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Exported {len(output)} assets to {OUTPUT_PATH}")
    replace_count = sum(1 for a in output if a["recommendation"] == "REPLACE")
    repair_count = sum(1 for a in output if a["recommendation"] == "REPAIR")
    print(f"  REPLACE: {replace_count}, REPAIR: {repair_count}")


if __name__ == "__main__":
    main()
