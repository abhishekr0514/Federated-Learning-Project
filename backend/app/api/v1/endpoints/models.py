"""Model status and weights registry router skeleton."""

from fastapi import APIRouter

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/status")
async def model_status():
    """Retrieve model registry and inference readiness status."""
    return {
        "inference_enabled": False,
        "status": "pending_ml_implementation",
        "target_architecture": {
            "ecg": "1D CNN + BiLSTM (128d)",
            "text": "BioClinicalBERT (128d)",
            "tabular": "Neural Tabular Encoder (64d)",
            "fusion": "Graph Attention Network (GAT)",
        },
        "message": "GAT multimodal model training and inference pipelines are currently unexposed.",
    }
