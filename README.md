# GraphRAG-Based Asset Repair and Replace Decision Engine

AI-powered assistance using Knowledge Graphs and Large Language Models for airport asset management.

Developed for **Kempegowda International Airport Bengaluru (BIAL)** — ICT Transformation Projects[cite: 1].

---

## 📌 Executive Summary
Airports run multiple critical assets—such as HVAC chillers, Air Handling Units (AHUs), Passenger Boarding Bridges (PBBs), elevators, escalators, moving walkways, and baggage handling systems—spread across multiple buildings and operational zones[cite: 2].

Each asset generates disconnected streams of data over its operational lifecycle: inspection results, work orders, repair costs, energy readings, and warranty records[cite: 2]. Traditionally, repair-vs-replace decisions are made manually based on subjective expert judgment[cite: 3]. This approach lacks fleet-wide visibility and breaks down at scale[cite: 3].

This project unifies disconnected asset data into a **Neo4j Knowledge Graph**, evaluates maintenance triggers using a **Configurable 8-Rule Decision Engine**, and generates plain-English, traceable explanations using **Groq (LLaMA 3.3)**[cite: 2, 4, 16].

---

## 🚀 Key Features

* **Neo4j Knowledge Graph Schema**: Links assets, equipment specifications, inspection/repair logs, and financial records into a queryable graph[cite: 15].
* **Configurable Decision Engine**: Enforces deterministic threshold logic across financial, lifecycle, reliability, efficiency, and safety criteria[cite: 7].
* **GraphRAG Reasoning Layer**: Combines multi-hop graph traversal with LLaMA 3.3 to produce field-ready, auditable justifications citing exact rule triggers[cite: 11, 28].
* **Interactive Operations Dashboard**: Built with Leaflet.js and HTML5/CSS Grid to deliver an airport blueprint map, color-coded recommendation statuses, live graph neighborhood visualizer, and a fleet Q&A assistant[cite: 17, 23].

---

## 🏗️ GraphRAG vs. Traditional Vector RAG

| Dimension | Vector RAG | GraphRAG (Our Architecture) |
| :--- | :--- | :--- |
| **Data Representation** | Unstructured document chunks[cite: 10, 13] | Connected nodes and relationship graphs[cite: 11, 13] |
| **Relationship Traversal** | Cannot model interconnected entities[cite: 10, 14] | Native multi-hop traversal (`Asset` $\rightarrow$ `Type` $\rightarrow$ `Repair` $\rightarrow$ `Cost`)[cite: 10, 14] |
| **Logic & Thresholds** | Relies on semantic similarity; fails math logic[cite: 10, 14] | Deterministic business rules + numerical criteria[cite: 10, 14] |
| **Explainability** | Black-box output; difficult to trace provenance[cite: 10, 14] | Auditable; traced back to exact graph nodes and fired rules[cite: 4, 14] |
| **Query Complexity** | $\sim \mathcal{O}(n)$ over total chunk corpus[cite: 13] | $\sim \mathcal{O}(k)$ neighborhood traversal[cite: 13] |

---

## 📊 Knowledge Graph Schema
(:Asset)
      /   │    \
     /    │     \
[:HAS_TYPE] │  [:UNDERGOES]
   /      │       \
  ▼       │        ▼
(:EquipmentType)  (:Repair)
          │        /
     [:HAS_COST]  / [:HAS_COST]
          │      /
          ▼     ▼
          (:Cost)

* **`(:Asset)`**: `assetId`, `physical_location`, `condition_rating`, `criticality`, `served_area`, `install_date`, `current_energy_kwh_per_ton`, `major_failure_count`.
* **`(:EquipmentType)`**: `type_name`, `manufacturer`, `model`, `design_life_years`, `mtbf_days`, `baseline_energy_kwh_per_ton`.
* **`(:Repair)`**: `issue_description`, `findings`, `status`.
* **`(:Cost)`**: `repair_cost`, `replacement_cost`, `purchase_cost`, `maintenance_labor_spend`, `parts_materials_spend`, `total_cost_of_ownership`.

---

## ⚙️ Decision Engine Rules

An asset triggers a **REPLACE** recommendation if any single rule (1–7) is satisfied. Rule 8 serves as a soft flag for review. If no rules trigger, the recommendation is **REPAIR**.

| Rule # | Trigger Condition | Category | Recommendation |
| :---: | :--- | :--- | :---: |
| **Rule 1** | Repair cost $> 50\%$ of replacement cost | Financial | **REPLACE** |
| **Rule 2** | Asset age $> 80\%$ of design service life | Lifecycle | **REPLACE** |
| **Rule 3** | $\ge 3$ major failures in the last 24 months | Reliability | **REPLACE** |
| **Rule 4** | Current energy use $> 20\%$ above equipment baseline | Efficiency | **REPLACE** |
| **Rule 5** | Condition rating $\le 2$ out of 5 | Inspection | **REPLACE** |
| **Rule 6** | High criticality $+\ \ge 2$ major failures | Safety-Weighted | **REPLACE** |
| **Rule 7** | Mean Time Between Failures (MTBF) $< 100$ days | Reliability | **REPLACE** |
| **Rule 8** | Warranty expired $+$ age $> 60\%$ of design service life | Soft Flag | **FLAG FOR REVIEW** |

---

## 📈 Validation & Results

The engine was validated against a sample dataset of 10 HVAC assets spanning 6 equipment classes across 4 airport buildings:
* **Accuracy**: **100% (10/10)** match against domain-expert ground truth classifications.
* **Replace Recommendations**: **6 assets** (`RTU-V01-BLDG02`, `CU-V05-BLDG01`, `CHP-V06-BLDG04`, `FCU-V07-BLDG03`, `RTU-V08-BLDG02`, `WCC-V09-BLDG04`).
* **Repair Recommendations**: **4 assets** (`AHU-V02-BLDG01`, `FCU-V03-BLDG03`, `EF-V04-BLDG02`, `CU-V10-BLDG03`).
* **Auditability**: Every generated recommendation is traceable back to the triggering rules and verified graph metrics.

---

## 🛠️ Tech Stack

* **Graph Database**: Neo4j (Cypher Query Language)
* **LLM & Inference**: Groq API (`llama-3.3-70b-versatile`)
* **Backend**: Python 3.10+, Flask, Flask-CORS
* **Data Processing**: pandas, openpyxl
* **Frontend**: HTML5, Vanilla JavaScript, CSS Grid, Leaflet.js

---

## 📂 Project Structure

```text
├── assets.json                         # Exported asset data for the dashboard
├── index.html                          # Leaflet campus map, detail panel, graph explorer
├── server.py                           # Flask backend API (port 5001)
├── ask.py                              # Interactive CLI for Cypher retrieval & Q&A
├── decision_engine.py                  # 8-rule deterministic evaluation engine
├── reasoning_layer.py                  # LLM explanation generator via Groq
├── export_to_json.py                   # Graph-to-JSON export script with layout jittering
├── asset_repair_vs_replace_data_V2.xlsx # Raw source dataset
├── start_server.command                # Shell launcher for backend
├── requirements.txt                    # Python package dependencies
├── .env.example                        # Environment variables template
└── README.md

## ⚡ Quickstart Guide

```markdown
### 1. Prerequisites
* Python 3.10 or higher
* Neo4j Desktop or Neo4j AuraDB instance running on bolt://127.0.0.1:7687
* A free Groq API Key

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/graphrag-asset-decision-engine.git
cd graphrag-asset-decision-engine

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Export necessary credentials
export GROQ_API_KEY="gsk_your_groq_api_key_here"
export NEO4J_PASSWORD="your_neo4j_password_here"

### 3. Pipeline Execution
```bash
# 1. Ingest Excel asset data into Neo4j
python src/load_data.py

# 2. Run the deterministic decision engine (evaluates rules 1-8 and writes back recommendations)
python src/decision_engine.py

# 3. Generate LLM explanations via Groq LLaMA 3.3
python src/reasoning_layer.py

# 4. Export graph data into assets.json for the front-end dashboard
python src/export_to_json.py

### 4. Running the Web Application
```bash
# Start the Flask backend API (runs on http://127.0.0.1:5001)
python src/server.py

python -m http.server 8000 --directory frontend/

## 👥 Authors & Acknowledgments

* **Tanishka Sharma** — Intern, ICT Transformation Projects[cite: 1]
* **Under the Guidance of:**
  * **Mr. Subhash Narayan C** — Assistant General Manager, ICT Transformation Projects[cite: 1]
  * **Mr. Shakti Mitra** — Deputy General Manager, ICT Transformation Projects[cite: 1]
  * **Mr. Jacxine Fernandez** — Senior Vice President, ICT Transformation Projects, Cyber Security, Operations & Service Delivery[cite: 1]
* **Organization**: Bangalore International Airport Limited (BIAL)[cite: 1, 6]

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
