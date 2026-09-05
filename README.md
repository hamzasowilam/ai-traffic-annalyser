# AI-Powered Network Traffic Analyzer & Intrusion Detection System (NIDS)

An intelligent, real-time Network Intrusion Detection System (NIDS) combining supervised learning, unsupervised anomaly detection, and LLM-powered remediation analysis.

---

## Architecture Overview

* **Machine Learning Pipeline:** 
  * **XGBoost Classifier:** High-precision supervised multi-class threat classification.
  * **Isolation Forest:** Unsupervised anomaly detection for zero-day and out-of-distribution traffic patterns.
* **Backend API (`FastAPI`):** High-throughput asynchronous REST API for real-time packet ingest, feature extraction, and prediction endpoints.
* **Interactive Dashboard (`Streamlit`):** Live telemetry visualization, threat alerts, drift monitoring, and packet stream breakdowns.
* **LLM Remediation Agent:** Integrated Groq LLM client for generating automated incident narratives and concrete mitigation playbooks.
* **Traffic Simulation (`Scapy`):** Attack generation tool for testing real-time detection against active exploits and benign baseline flows.

---

## Repository Structure

```text
├── ml/
│   ├── data/            # KDD training and testing datasets
│   ├── artifacts/       # Serialized models (.pkl) & feature schemas
│   └── train.py         # Model training & evaluation pipeline
├── api.py               # FastAPI backend service
├── app.py               # Application entrypoint wrapper
├── dashboard.py         # Streamlit analytical dashboard
├── intelligence.py      # Fingerprinting, drift monitoring & LLM analysis
├── attacker_sim.py      # Network packet & attack simulation engine
└── .gitignore
