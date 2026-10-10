"""Document and clinical text management router skeleton."""

from fastapi import APIRouter
from backend.app.core.errors import FeatureNotImplementedError

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/")
async def list_documents():
    """List clinical report document metadata."""
    raise FeatureNotImplementedError("Clinical document management pending Document contract implementation.")


@router.post("/upload")
async def upload_document():
    """Upload new clinical report document for local processing."""
    raise FeatureNotImplementedError("Document upload pending implementation.")
