# MedShield-FL System Architecture

## Overview

**MedShield-FL** (Privacy Masked Federated Learning for Heart Disease Diagnosis) is a privacy-preserving, multimodal healthcare AI system designed to perform collaborative diagnostic modeling across multiple healthcare institutions without centralizing sensitive patient data.

---

## Architectural Topology & Privacy Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HOSPITAL / CLIENT BOUNDARY                            │
│                                                                                 │
│  ┌─────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐  │
│  │ 12-lead ECG Data│   │  Clinical Notes/Text │   │ Tabular Patient Metrics  │  │
│  └────────┬────────┘   └──────────┬───────────┘   └────────────┬─────────────┘  │
│           │                       │                            │                │
│           ▼                       ▼                            ▼                │
│  ┌─────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐  │
│  │ 1D CNN + BiLSTM │   │   BioClinicalBERT    │   │  Neural Tabular Encoder  │  │
│  │   (128d Embedding)  │   │  (768d ➔ 128d Proj)  │   │      (64d Embedding)     │  │
│  └────────┬────────┘   └──────────┬───────────┘   └────────────┬─────────────┘  │
│           │                       │                            │                │
│           └───────────────────────┼────────────────────────────┘                │
│                                   ▼                                             │
│                     ┌──────────────────────────┐                                │
│                     │   GAT Multimodal Fusion  │                                │
│                     │(3 Nodes ➔ 384d ➔ Logits) │                                │
│                     └─────────────┬────────────┘                                │
│                                   │                                             │
│                                   ▼                                             │
│                     ┌──────────────────────────┐                                │
│                     │ Privacy Masking / DP     │                                │
│                     └─────────────┬────────────┘                                │
└───────────────────────────────────┼─────────────────────────────────────────────┘
                                    │ (Encrypted Model Updates ONLY)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CENTRAL SERVER BOUNDARY                              │
│                                                                                 │
│                     ┌──────────────────────────┐                                │
│                     │  Flower FL Aggregator    │                                │
│                     └─────────────┬────────────┘                                │
│                                   │                                             │
│            ┌──────────────────────┴──────────────────────┐                      │
│            ▼                                             ▼                      │
│  ┌──────────────────┐                           ┌──────────────────┐            │
│  │  FastAPI Backend │                           │ Web UI Frontend  │            │
│  │  PostgreSQL / DB │                           │ (4 Role Portals) │            │
│  └──────────────────┘                           └──────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Multimodal Model Architecture

### 1. ECG Modality Pipeline
- **Input Tensor**: `12 × 1000` (12-lead ECG signal sampled at 1000 timesteps)
- **Feature Extractor**: Multi-stage 1D Convolutional Neural Network (1D CNN) capturing local waveform features and QRS complex dynamics.
- **Sequential Encoder**: Bidirectional LSTM (BiLSTM) capturing temporal dependencies across cardiac cycles.
- **Output Representation**: `128`-dimensional dense embedding vector.

### 2. Clinical Text Modality Pipeline
- **Input Context**: Unstructured clinical history, diagnostic notes, and patient records.
- **Pretrained Encoder**: `BioClinicalBERT` extracting a `768`-dimensional domain-specific contextual vector.
- **Adapter Projection**: Learned linear projection with Layer Normalization mapping representation to a `128`-dimensional dense embedding vector.

### 3. Structured / Tabular Modality Pipeline
- **Input Vector**: `10`-dimensional clinical parameter vector (age, sex, systolic BP, diastolic BP, cholesterol, glucose, BMI, smoking status, alcohol intake, physical activity).
- **Encoder**: Neural Tabular Encoder with non-linear feature embeddings and residual skip connections.
- **Output Representation**: `64`-dimensional dense embedding vector.

### 4. Graph Attention Network (GAT) Multimodal Fusion
- **Graph Construction**:
  - Modality nodes: Node 1 (ECG, 128d), Node 2 (Clinical Text, 128d), Node 3 (Tabular, 64d).
  - Modality Alignment: Projection layer standardizing modality embeddings into unified node spaces.
  - Fully-connected graph edges capturing cross-modality attention weights.
- **GAT Layering**: Multi-head Graph Attention layers computing dynamic attention coefficients $\alpha_{ij}$ between modalities.
- **Classification Head**:
  - Flattened Graph Node Vector: `384`-dimensional representation.
  - Hidden Layer: `128`-dimensional fully-connected layer with Dropout and ReLU activation.
  - Output Layer: 2-class logits for binary cardiovascular disease diagnosis.

> **Architectural Constraint**: Modality fusion MUST use the GAT architecture. Substituting GAT with simple concatenation, XGBoost, DenseNet, or MLPs as the primary fusion mechanism is strictly prohibited.

---

## Federated Learning & Privacy Infrastructure

1. **Flower Framework**: Client-server communication protocol for federated rounds, model parameter synchronization, and global weight aggregation.
2. **Local Client Boundaries**: Raw patient data, raw ECG arrays, clinical notes, and PII NEVER leave local client hospital instances.
3. **Privacy Masking**: Model gradients and parameter updates undergo noise addition / gradient clipping prior to outbound transmission.

---

## Explainable & Causal AI Layer

1. **DiCE (Diverse Counterfactual Explanations)**: Generates actionable counterfactual feature perturbations to explain diagnostic model predictions to clinicians.
2. **DoWhy (Causal Inference Framework)**: Models causal DAGs linking clinical interventions (e.g., lifestyle, medication) to diagnostic outcomes, verifying non-spurious feature relationships.

---

## Application & Database Architecture

- **Backend Framework**: FastAPI (Python 3.10+) with async request handling.
- **Persistence Layer**: PostgreSQL with SQLAlchemy 2.x ORM and Alembic schema versioning.
- **Security & RBAC**: JWT token authentication enforcing role-based access control across:
  1. **Super Admin**: System governance, node registration, global model oversight.
  2. **Hospital Admin**: Client node configuration, local training policies, audit logs.
  3. **Clinician**: Diagnostic portal, patient evaluation, XAI counterfactual views.
  4. **Patient**: Personal privacy status, consent management, diagnostic reports.
