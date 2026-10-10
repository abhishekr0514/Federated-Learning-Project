# Privacy Masked Federated Learning for Heart Disease Diagnosis (MedShield-FL)

[![Project Status](https://img.shields.io/badge/Status-Prompt%2001%20Foundation-blue.svg)](docs/PROJECT_STATUS.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**MedShield-FL** is a privacy-preserving, multimodal federated learning system engineered for collaborative cardiovascular disease diagnosis across decentralized healthcare institutions without centralizing sensitive patient data.

---

## Authoritative Repository & Integrity Rules

- **GitHub Repository**: `https://github.com/abhishekr0514/Federated-Learning-Project`
- **Architectural Rules**: Modality fusion MUST use Graph Attention Networks (GAT) across 1D CNN-BiLSTM ECG embeddings (128d), BioClinicalBERT clinical text embeddings (128d), and Neural Tabular Encoder features (64d).
- **Privacy Boundaries**: Raw patient data, ECG signals, clinical notes, and PII NEVER leave local client hospital boundaries. Central databases store zero raw clinical text or direct patient identifiers.

---

## Current Development Status

- **Prompt 00 Audit & Contracts**: Complete. Baseline governance contracts are locked under `docs/`:
  - [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)
  - [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
  - [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md)
  - [`docs/MODEL_CONTRACT.md`](docs/MODEL_CONTRACT.md)
  - [`docs/SECURITY_CONTRACT.md`](docs/SECURITY_CONTRACT.md)
  - [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md)
  - [`docs/DECISIONS.md`](docs/DECISIONS.md)
  - [`docs/RESTORATION_NOTES.md`](docs/RESTORATION_NOTES.md)
- **Prompt 01 Foundation**: Complete. Repository structure, environment configuration (`.env.example`), FastAPI backend (`GET /health`), SQLAlchemy 2.x / Alembic setup, React frontend status shell, and pytest test suite initialized.

---

## Repository Structure

```
MedShield-FL/
├── backend/                  # FastAPI Application & Server Infrastructure
│   ├── app/
│   │   ├── api/             # API Router Endpoints
│   │   ├── core/            # Config, Security, and App Settings
│   │   ├── db/              # SQLAlchemy 2.x Session, Engine & Alembic Migrations
│   │   ├── models/          # ORM Database Models
│   │   ├── schemas/         # Pydantic Schemas
│   │   ├── repositories/    # Data Access Repositories
│   │   ├── services/        # Business Logic Services
│   │   ├── security/        # Auth, JWT, and RBAC
│   │   ├── workers/         # Background Tasks
│   │   └── main.py          # FastAPI Application Entry Point
│   └── tests/
│
├── frontend/                 # React (Vite) User Interface Console
│   ├── src/
│   │   ├── components/      # UI Components
│   │   ├── layouts/         # Layout Shells
│   │   ├── pages/           # Role-based Portal Pages
│   │   ├── App.jsx          # System Status Console App
│   │   └── main.jsx         # React Entry Point
│   ├── index.html
│   └── vite.config.js
│
├── ml/                       # Machine Learning Pipelines
│   ├── models/              # ECG 1D CNN-BiLSTM, BioClinicalBERT, Tabular, GAT Fusion
│   ├── datasets/            # Dataset Wrappers
│   ├── preprocessing/       # Signal Processing & Normalization
│   ├── training/            # Model Training Routines
│   ├── evaluation/          # Metrics Calculation
│   ├── inference/           # Inference Engine
│   ├── explainability/      # DiCE Counterfactual Explanations
│   └── causal/              # DoWhy Causal DAG Inference
│
├── fl/                       # Federated Learning Infrastructure
│   ├── client/              # Flower Local Client
│   ├── server/              # Flower Server & Parameter Aggregator
│   ├── strategies/          # Differential Privacy & Aggregation Strategies
│   └── validation/          # Federated Model Validation
│
├── privacy/                  # Privacy & Differential Privacy Layer
│   ├── pii/                 # PII Sanitization
│   ├── masking/             # Clinical Text Masking
│   └── policies/            # Differential Privacy Budgeting (RDP)
│
├── data/                     # Local Dataset Directories (Git Ignored)
│   ├── raw/                 # Raw Input Datasets (PTB-XL, Cardio, NBME)
│   ├── processed/           # Processed Tensor Files
│   └── manifests/           # Patient-Independent Split Manifests
│
├── tests/                    # Pytest System Test Suite
├── docs/                     # Compliance & Architecture Contracts
├── scripts/                  # Development & Maintenance Helper Scripts
├── deployment/               # Container & Server Deployment Manifests
├── .env.example              # Environment Variables Example Template
├── pyproject.toml            # Python Project Dependencies & Tooling Config
├── Makefile                  # Developer Commands Shortcuts
└── README.md                 # Main Documentation
```

---

## Environment Setup & Local Execution

### 1. Environment Configuration
Copy the example environment template to `.env`:
```bash
cp .env.example .env
```
Update configuration parameters in `.env` as required for your local setup.

### 2. Python Dependencies Installation
Install backend and core dependencies:
```bash
pip install -e .
```
For ML or FL dependencies:
```bash
pip install -e ".[ml,fl,dev]"
```

### 3. Running the Backend Server
Start the FastAPI backend application:
```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
Verify operational readiness by visiting:
- Health Check: `http://127.0.0.1:8000/health`
- OpenAPI Documentation: `http://127.0.0.1:8000/docs`

### 4. Running the Frontend Development Server
Navigate to `frontend/` and start Vite:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Testing & Quality Assurance

Run the test suite using pytest:
```bash
pytest tests
```

---

## Current Scientific & Technical Limitations

1. **Proxy Task Notice**: The PTB-XL ECG binary target currently evaluates normal sinus rhythm (SNOMED 426783006) vs pathological superclasses (`MI`, `STTC`). This is a **proxy task only** and must not be cited as validated clinical heart-disease prediction.
2. **Dataset Isolation**: Raw datasets (`cardio_train.csv`, NBME clinical notes) remain separate and require verified patient linkage prior to joint multimodal training.
3. **No Metric Fabrication**: System APIs report operational component readiness only. GAT model inference endpoints will be enabled upon full training and validation in upcoming phases.