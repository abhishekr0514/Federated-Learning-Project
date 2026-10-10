"""System performance and evaluation metrics router skeleton."""

from fastapi import APIRouter
from backend.app.core.errors import FeatureNotImplementedError

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/evaluation")
async def get_evaluation_metrics():
    """Retrieve global model evaluation metrics.
    
    This endpoint returns 501 until empirical evaluation metrics are calculated on independent test sets without metric fabrication.
    """
    raise FeatureNotImplementedError(
        "Empirical evaluation metrics are pending model training and test split evaluation. "
        "Fabricating performance metrics is strictly prohibited."
    )
