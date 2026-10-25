"""
Asset Repair vs. Replace Decision Engine
Reads Asset nodes (+ their EquipmentType, Repair, Cost neighbors) from
Neo4j, evaluates Rules 1-8, and outputs a recommendation (REPAIR / REPLACE)
with the reasoning behind it.

NOTE ON RULE 4:
Rules 3, 6, and 8 originally needed major_failure_count, criticality, and
warranty_expired, none of which were loaded into the graph at first --
all three have since been migrated in from the source xlsx, so Rules 3,
6, and 8 now use the real fields again. Rule 4 is the one holdout: it
still compares current energy use to the EquipmentType's
baseline_energy_kwh_per_ton instead of a hypothetical new-unit spec,
since that spec was never loaded. Rules 1, 2, 5, 7 were always
straightforward.

Run locally where your Neo4j Desktop instance is reachable:
    pip install neo4j
    python decision_engine.py
"""

import datetime
from neo4j import GraphDatabase

# ---- connection settings: match your "AssetGraphV2" instance ----
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tanishka123"   # <-- set this

CURRENT_DATE = datetime.date.today()

# ---------------------------------------------------------------------
# Rule thresholds -- centralized so they're easy to tune
# ---------------------------------------------------------------------

REPAIR_COST_RATIO_THRESHOLD = 0.5       # Rule 1
AGE_PCT_OF_DESIGN_LIFE_THRESHOLD = 0.8  # Rule 2
MAJOR_FAILURE_COUNT_THRESHOLD = 3       # Rule 3
ENERGY_OVERAGE_PCT_THRESHOLD = 0.2      # Rule 4
CONDITION_RATING_THRESHOLD = 2          # Rule 5
RULE6_CRITICALITY = "High"              # Rule 6
RULE6_MAJOR_FAILURE_THRESHOLD = 2       # Rule 6
MTBF_DAYS_THRESHOLD = 100               # Rule 7
RULE8_AGE_PCT_THRESHOLD = 0.6           # Rule 8


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_install_year(install_date):
    """Neo4j Date/DateTime -> int year. Also handles plain strings/dates."""
    if not install_date:
        return None
    if hasattr(install_date, "year"):
        return install_date.year
    match = str(install_date)[:4]
    return int(match) if match.isdigit() else None


def get_age_years(install_date):
    install_year = parse_install_year(install_date)
    if install_year is None:
        return None
    return CURRENT_DATE.year - install_year


# ---------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------

def evaluate_rules(asset, equipment_type, repair, cost):
    """
    asset, equipment_type, repair, cost: dicts of node properties
    (equipment_type, repair, cost may be None if the relationship is missing)
    Returns (triggered_rules: list[str], insufficient: list[str], flagged_for_review: bool)
    """
    triggered = []
    insufficient = []

    condition_rating = asset.get("condition_rating")
    current_energy = asset.get("current_energy_kwh_per_ton")
    install_date = asset.get("install_date")
    major_failure_count = asset.get("major_failure_count")
    criticality = asset.get("criticality")

    design_life_years = (equipment_type or {}).get("design_life_years")
    mtbf_days = (equipment_type or {}).get("mtbf_days")
    baseline_energy = (equipment_type or {}).get("baseline_energy_kwh_per_ton")

    warranty_expired = asset.get("warranty_expired")

    repair_cost = (cost or {}).get("repair_cost")
    replacement_cost = (cost or {}).get("replacement_cost")

    age = get_age_years(install_date)

    # Rule 1: Repair cost > 50% of replacement cost
    if repair_cost is not None and replacement_cost:
        pct = repair_cost / replacement_cost
        if pct > REPAIR_COST_RATIO_THRESHOLD:
            triggered.append(f"Rule 1: Repair cost is {pct:.0%} of replacement cost (>50%)")
    else:
        insufficient.append("Rule 1 (missing repair_cost/replacement_cost on Cost node)")

    # Rule 2: Age > 80% of design service life
    if age is not None and design_life_years:
        pct = age / design_life_years
        if pct > AGE_PCT_OF_DESIGN_LIFE_THRESHOLD:
            triggered.append(f"Rule 2: Asset age is {pct:.0%} of design service life (>80%)")
    else:
        insufficient.append("Rule 2 (missing install_date/design_life_years)")

    # Rule 3: Major failure count >= 3 in last 24 months
    # This was previously approximated with work_order_count because
    # major_failure_count wasn't loaded into the graph. It's now been
    # migrated in from the source xlsx, so this checks the real field.
    if major_failure_count is not None:
        if major_failure_count >= MAJOR_FAILURE_COUNT_THRESHOLD:
            triggered.append(f"Rule 3: {major_failure_count} major failures in last 24 months (>=3)")
    else:
        insufficient.append("Rule 3 (missing major_failure_count)")

    # Rule 4 (REDEFINED): current energy use exceeds the equipment type's
    # baseline by more than 20%.
    # Original compared current usage to a hypothetical new-unit spec that
    # was never loaded; baseline_energy_kwh_per_ton on EquipmentType is the
    # real field available, so we measure drift from that instead.
    if current_energy and baseline_energy:
        overage = (current_energy - baseline_energy) / baseline_energy
        if overage > ENERGY_OVERAGE_PCT_THRESHOLD:
            triggered.append(
                f"Rule 4: Current energy use is {overage:.1%} above baseline for this equipment type (>20%)"
            )
    else:
        insufficient.append("Rule 4 (missing current_energy_kwh_per_ton/baseline_energy_kwh_per_ton)")

    # Rule 5: Condition rating <= 2 (poor/critical inspection result)
    if condition_rating is not None:
        if condition_rating <= CONDITION_RATING_THRESHOLD:
            triggered.append(f"Rule 5: Condition rating is {condition_rating}/5 (poor or critical)")
    else:
        insufficient.append("Rule 5 (missing condition_rating)")

    # Rule 6: High criticality + major failures >= 2 (safety-weighted)
    # Previously approximated with work_order_count + condition_rating
    # because criticality wasn't loaded. Now uses the real field.
    if criticality is not None and major_failure_count is not None:
        if criticality == RULE6_CRITICALITY and major_failure_count >= RULE6_MAJOR_FAILURE_THRESHOLD:
            triggered.append(
                f"Rule 6: High-criticality asset with {major_failure_count} major failures (>=2)"
            )
    else:
        insufficient.append("Rule 6 (missing criticality/major_failure_count)")

    # Rule 7: MTBF < 100 days (frequent unplanned failures)
    if mtbf_days is not None:
        if mtbf_days < MTBF_DAYS_THRESHOLD:
            triggered.append(f"Rule 7: MTBF is {mtbf_days} days (<100), indicating frequent unplanned failures")
    else:
        insufficient.append("Rule 7 (missing mtbf_days)")

    # Rule 8 (soft flag -- does not force REPLACE on its own): warranty
    # expired AND age > 60% of design life.
    # Previously approximated with Repair.status == "Open" because
    # warranty_expired wasn't loaded. Now uses the real field, migrated in
    # from the source xlsx.
    flagged_for_review = False
    if warranty_expired is not None and age is not None and design_life_years:
        pct = age / design_life_years
        if warranty_expired and pct > RULE8_AGE_PCT_THRESHOLD:
            flagged_for_review = True
    else:
        insufficient.append("Rule 8 (missing warranty_expired/install_date/design_life_years)")

    return triggered, insufficient, flagged_for_review


def get_recommendation(triggered_rules):
    return "REPLACE" if triggered_rules else "REPAIR"


# ---------------------------------------------------------------------
# Neo4j fetch / write-back
# ---------------------------------------------------------------------

FETCH_QUERY = """
MATCH (a:Asset)
OPTIONAL MATCH (a)-[:HAS_TYPE]->(et:EquipmentType)
OPTIONAL MATCH (a)-[:UNDERGOES]->(r:Repair)
OPTIONAL MATCH (a)-[:HAS_COST]->(c:Cost)
RETURN properties(a) AS asset,
       properties(et) AS equipment_type,
       properties(r) AS repair,
       properties(c) AS cost
ORDER BY asset.assetId
"""


def fetch_assets(driver):
    with driver.session() as session:
        result = session.run(FETCH_QUERY)
        return [record.data() for record in result]


def write_result_to_neo4j(tx, asset_id, recommendation, triggered_rules, flagged_for_review):
    tx.run(
        """
        MATCH (a:Asset {assetId: $asset_id})
        SET a.engine_recommendation = $recommendation,
            a.engine_triggered_rules = $triggered_rules,
            a.engine_flagged_for_review = $flagged_for_review,
            a.engine_evaluated_at = datetime()
        """,
        asset_id=asset_id,
        recommendation=recommendation,
        triggered_rules=triggered_rules,
        flagged_for_review=flagged_for_review,
    )


def main(write_back=True):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    rows = fetch_assets(driver)

    print(f"Evaluating {len(rows)} assets against Rules 1-8\n")
    print("=" * 100)

    for row in rows:
        asset = row["asset"]
        equipment_type = row["equipment_type"]
        repair = row["repair"]
        cost = row["cost"]

        asset_id = asset.get("assetId", "UNKNOWN")
        triggered, insufficient, flagged_for_review = evaluate_rules(
            asset, equipment_type, repair, cost
        )
        recommendation = get_recommendation(triggered)

        print(f"\nAsset: {asset_id}")
        print(f"  Recommendation: {recommendation}")

        if triggered:
            print("  Triggered Rules:")
            for rule in triggered:
                print(f"    - {rule}")
        else:
            print("  Triggered Rules: None")

        if flagged_for_review:
            print("  \u26a0 Rule 8: Warranty expired on an asset >60% through its design life -> flag for review")

        if insufficient:
            print(f"  Insufficient data to evaluate: {', '.join(insufficient)}")

        if write_back:
            with driver.session() as session:
                session.execute_write(
                    write_result_to_neo4j,
                    asset_id,
                    recommendation,
                    "; ".join(triggered) if triggered else "None",
                    flagged_for_review,
                )
            print("  -> Written to Neo4j as a.engine_recommendation")

    driver.close()

    print("\n" + "=" * 100)
    if write_back:
        print("Engine results written back to each Asset node as: "
              "engine_recommendation, engine_triggered_rules, engine_flagged_for_review, engine_evaluated_at")


if __name__ == "__main__":
    main(write_back=True)