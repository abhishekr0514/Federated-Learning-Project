# MedShield-FL Security & Privacy Contract

## Security Architecture Overview

The MedShield-FL platform enforces multi-layered security controls encompassing data isolation, identity and access management (IAM), transport security, and differential privacy guarantees.

---

## Identity & Access Control (IAM)

### Authentication
- Protocol: JSON Web Tokens (JWT) signed with HS256 / RS256 algorithm.
- Password Hashing: Argon2id or bcrypt with minimum work factor of 12.
- Token Lifetime: Access Tokens expire in 15 minutes; Refresh Tokens expire in 7 days with token rotation.

### Role-Based Access Control (RBAC) Matrix

| Operation / Resource | Super Admin | Hospital Admin | Clinician | Patient |
| :--- | :---: | :---: | :---: | :---: |
| **System Node Management** |  |  |  |  |
| **Local FL Training Config** |  |  |  |  |
| **Submit FL Model Updates** |  | (Client Node) |  |  |
| **View Aggregated Metrics** |  |  |  |  |
| **Execute XAI / DiCE** |  |  |  |  |
| **View Patient Diagnosis** |  |  |  |  |
| **View Personal Consent** |  |  |  |  |

---

## Data Privacy & Storage Isolation Rules

1. **Central Database Prohibition**:
   - The central PostgreSQL database is forbidden from storing raw ECG signal vectors, raw clinical report strings, or direct Patient Identifying Information (PII such as full name, SSN, national ID).
   - Only pseudonymous client/node identifiers and aggregated model parameters are retained centrally.

2. **Hospital Node Privacy Boundary**:
   - Local patient records remain within local hospital databases / disk storage.
   - Access to raw data inside local nodes is restricted to authenticated local hospital workers.

3. **Federated Privacy Masking**:
   - **Gradient Clipping**: Local update gradients are clipped using $L_2$ norm threshold $C$.
   - **Differential Privacy (DP)**: Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ is added to local model weight updates prior to Flower FL server transmission.
   - **Privacy Budget Tracking**: $(\epsilon, \delta)$-differential privacy budgets are tracked across training rounds using Rényi Differential Privacy (RDP) accumulators.

---

## Audit & Compliance Logging

- All API endpoints record immutable audit logs containing timestamp, subject ID, role, action, target resource, and IP address.
- Audit logs are protected against tampering and stored separately from operational transaction tables.
