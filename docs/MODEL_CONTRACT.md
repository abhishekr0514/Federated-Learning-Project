# MedShield-FL Model Contract & Architecture Specifications

## Model Architecture Compliance Matrix

| Component | Target Architecture | Permissible Modifications | Prohibited Modifications |
| :--- | :--- | :--- | :--- |
| **ECG Encoder** | 12×1000 → 1D CNN → BiLSTM → 128d | Layer normalization, kernel size tuning, dropout adjustment | Replacing 1D CNN + BiLSTM with 2D image transforms or ResNet |
| **Text Encoder** | BioClinicalBERT → 768d → Linear Proj → 128d | Freeze/unfreeze transformer layers, projection head regularization | Replacing BioClinicalBERT with non-domain generic word embeddings |
| **Tabular Encoder**| 10d Vector → Neural Tabular Encoder → 64d | Activation function choice, layer depth adjustment | Linear regression or raw unencoded feature pass-through |
| **Multimodal Fusion**| Modality Alignment → 3 Nodes → GAT → 384d → 128d → Logits | Number of GAT heads, graph attention dropout | Replacing GAT with XGBoost, DenseNet, Concat, MLP, or Transformer fusion |

---

## Technical Specifications

### 1. ECG Sub-Network Specification
- **Input Dimensions**: $(B, 12, 1000)$ where $B$ is batch size, 12 signal channels (leads), 1000 time steps.
- **Convolutional Stage**:
  - Conv1D Block 1: 32 filters, kernel size 7, stride 2, padding 3, BatchNorm1d, ReLU, MaxPool1d(2).
  - Conv1D Block 2: 64 filters, kernel size 5, stride 1, padding 2, BatchNorm1d, ReLU, MaxPool1d(2).
  - Conv1D Block 3: 128 filters, kernel size 3, stride 1, padding 1, BatchNorm1d, ReLU.
- **Recurrent Stage**:
  - BiLSTM: input_size=128, hidden_size=64, num_layers=2, bidirectional=True, batch_first=True.
- **Embedding Projection**: Linear(128, 128) + LayerNorm. Output dimension: $(B, 128)$.

### 2. Clinical Text Sub-Network Specification
- **Input**: Tokenized clinical text sequences (max sequence length = 512).
- **Transformer Encoder**: `emilyalsentzer/Bio_ClinicalBERT` outputting pooled output vector $(B, 768)$.
- **Projection Head**:
  - Linear(768, 256) → GELU → Dropout(0.2) → Linear(256, 128) → LayerNorm.
  - Output dimension: $(B, 128)$.

### 3. Structured / Tabular Sub-Network Specification
- **Input Vector**: $(B, 10)$ normalized numeric features.
- **Encoder Architecture**:
  - Dense Block 1: Linear(10, 64) → BatchNorm1d → ReLU → Dropout(0.1).
  - Dense Block 2: Linear(64, 64) + Residual connection → ReLU → LayerNorm.
  - Output dimension: $(B, 64)$.

### 4. GAT Fusion Network Specification
- **Modality Node Mapping**:
  - Projection layers standardizing representations:
    - Node 1 (ECG): Linear(128, 128)
    - Node 2 (Text): Linear(128, 128)
    - Node 3 (Tabular): Linear(64, 128)
  - Constructed Graph Tensor: $(B, 3, 128)$ with 3 fully connected modality nodes.
- **Graph Attention Layers**:
  - GAT Layer 1: in_features=128, out_features=128, heads=4, concat=False.
  - GAT Layer 2: in_features=128, out_features=128, heads=1, concat=False.
- **Classification Dense Network**:
  - Flattened Representation: Reshape $(B, 3 \times 128 = 384)$.
  - Hidden Layer: Linear(384, 128) → ReLU → Dropout(0.3).
  - Output Layer: Linear(128, 2) → Binary classification logits.

---

## Controlled Baseline Comparisons

Alternative architectures (e.g., Early Concatenation MLP, Decision Trees / XGBoost on tabular data alone) are permitted **strictly** as documented baseline benchmark comparisons to validate the superior performance of the primary GAT architecture. They MUST NOT replace the core GAT model in the production pipeline.
