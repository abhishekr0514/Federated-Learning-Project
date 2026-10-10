# MedShield-FL REST API Contract

## Overview

The MedShield-FL backend provides a standardized FastAPI RESTful interface for managing federated learning orchestration, role authentication, diagnostic inference, counterfactual explainability, and system telemetry.

---

## Authentication Endpoints (`/api/v1/auth`)

### 1. `POST /api/v1/auth/login`
- **Desc**: Authenticates user credentials and returns JWT access/refresh tokens.
- **Request Body**:
  ```json
  {
    "username": "clinician_user",
    "password": "SecurePassword123!"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "access_token": "eyJhbGciOi...",
    "refresh_token": "eyJhbGciOi...",
    "token_type": "bearer",
    "expires_in": 900,
    "role": "Clinician"
  }
  ```

### 2. `POST /api/v1/auth/refresh`
- **Desc**: Refreshes expired access tokens using a valid refresh token.

---

## Federated Learning Management (`/api/v1/fl`)

### 1. `GET /api/v1/fl/rounds`
- **Desc**: Lists active and completed federated training rounds, participation statistics, and global metrics.
- **Access**: Super Admin, Hospital Admin.

### 2. `POST /api/v1/fl/start`
- **Desc**: Initiates a new federated training campaign with specified hyperparameters (rounds, fraction_fit, min_fit_clients, local_epochs).
- **Access**: Super Admin.

### 3. `GET /api/v1/fl/metrics`
- **Desc**: Retrieves historical evaluation metrics (loss, accuracy, AUC) across FL rounds.
- **Access**: Super Admin, Hospital Admin, Clinician.

---

## Client Node Telemetry (`/api/v1/nodes`)

### 1. `GET /api/v1/nodes`
- **Desc**: Retrieves status, health, and privacy configuration of registered client hospital nodes.
- **Access**: Super Admin.

### 2. `POST /api/v1/nodes/register`
- **Desc**: Registers a new hospital client node into the FL network.
- **Access**: Super Admin, Hospital Admin.

---

## Clinical Inference & Explainability (`/api/v1/clinical`)

### 1. `POST /api/v1/clinical/diagnose`
- **Desc**: Evaluates patient multimodal inputs and returns diagnostic risk probabilities.
- **Access**: Clinician.
- **Request Body**:
  ```json
  {
    "patient_id": "P-10492",
    "ecg_record_id": "ECG-89210",
    "tabular_features": {
      "age": 58,
      "sex": 1,
      "systolic_bp": 142,
      "diastolic_bp": 90,
      "cholesterol": 230,
      "glucose": 110,
      "bmi": 28.4,
      "smoking": 1,
      "alcohol": 0,
      "active": 1
    },
    "clinical_notes": "Patient presents with exertional chest pain and mild dyspnea over 3 days."
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "patient_id": "P-10492",
    "diagnosis": "Cardiovascular Disease Positive",
    "probability": 0.874,
    "risk_category": "High Risk",
    "modality_contributions": {
      "ecg_weight": 0.45,
      "text_weight": 0.35,
      "tabular_weight": 0.20
    }
  }
  ```

### 2. `POST /api/v1/clinical/explain/dice`
- **Desc**: Generates DiCE counterfactual explanations demonstrating minimal clinical feature changes required to transition diagnosis.
- **Access**: Clinician.

### 3. `POST /api/v1/clinical/explain/dowhy`
- **Desc**: Evaluates causal DAG relationships and calculates average treatment effects (ATE) for specified clinical interventions.
- **Access**: Clinician.
