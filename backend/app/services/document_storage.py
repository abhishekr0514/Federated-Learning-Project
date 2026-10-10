"""Protected Document Storage Service.

PRIVACY & SECURITY ARCHITECTURE:
1. Protected Storage Boundary: Original documents and masked clinical texts are stored
   strictly within an isolated, server-side storage hierarchy outside frontend/public and web roots.
2. Opaque Referencing: Internal object keys and filenames are generated exclusively from
   opaque UUIDs. Never use patient names, MRNs, phone numbers, or emails as keys.
3. Production Storage Contract: This local filesystem implementation provides the
   development interface. Production deployments must use enterprise object storage
   (e.g., S3/MinIO) featuring Server-Side Encryption (SSE-KMS), strict IAM access controls,
   and immutable audit logging.
"""

import os
import uuid
from pathlib import Path
from typing import Optional, Union

from backend.app.core.config import settings


class DocumentStorage:
    """Protected local filesystem storage abstraction for medical documents."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            # Default to storage/protected_documents at project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            self.base_dir = (project_root / "storage" / "protected_documents").resolve()

        # Enforce that storage is NEVER inside frontend or public assets
        base_str = str(self.base_dir).lower()
        if "frontend" in base_str or "public" in base_str or "static" in base_str:
            raise ValueError(
                "DocumentStorage location violation: storage directory must never be "
                "within frontend, public, or static asset paths."
            )

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_dir(self, hospital_id: uuid.UUID, doc_id: uuid.UUID) -> Path:
        """Resolve isolated folder for a given hospital and document UUID."""
        doc_dir = self.base_dir / str(hospital_id) / str(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    def store_original(
        self,
        hospital_id: uuid.UUID,
        doc_id: uuid.UUID,
        ext: str,
        content: bytes,
    ) -> str:
        """Store original uploaded binary content.
        
        Returns an opaque internal storage URI: protected://hospitals/{hospital_id}/docs/{doc_id}/original.{clean_ext}
        """
        clean_ext = ext.lstrip(".").lower()
        target_dir = self._resolve_dir(hospital_id, doc_id)
        target_file = target_dir / f"original.{clean_ext}"
        target_file.write_bytes(content)

        # Return standardized local reference URI
        return f"protected://hospitals/{hospital_id}/docs/{doc_id}/original.{clean_ext}"

    def store_masked(
        self,
        hospital_id: uuid.UUID,
        doc_id: uuid.UUID,
        masked_text: str,
    ) -> str:
        """Store privacy-masked text representation.
        
        Returns an opaque internal storage URI: protected://hospitals/{hospital_id}/docs/{doc_id}/masked.txt
        """
        target_dir = self._resolve_dir(hospital_id, doc_id)
        target_file = target_dir / "masked.txt"
        target_file.write_text(masked_text, encoding="utf-8")

        return f"protected://hospitals/{hospital_id}/docs/{doc_id}/masked.txt"

    def _uri_to_path(self, local_uri: str) -> Path:
        """Translate protected URI into local filesystem Path."""
        prefix = "protected://hospitals/"
        if not local_uri.startswith(prefix):
            # Fallback if raw path passed
            p = Path(local_uri).resolve()
            if not str(p).startswith(str(self.base_dir)):
                raise PermissionError("Access outside protected document storage boundary is forbidden.")
            return p

        rel_part = local_uri[len(prefix):]
        # Format: {hospital_id}/docs/{doc_id}/{filename}
        parts = rel_part.split("/")
        if len(parts) != 4 or parts[1] != "docs":
            raise ValueError(f"Malformed protected storage URI: {local_uri}")

        hospital_id_str, _, doc_id_str, filename = parts
        # Prevent directory traversal attacks
        if ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid storage file reference")

        target_path = (self.base_dir / hospital_id_str / doc_id_str / filename).resolve()
        if not str(target_path).startswith(str(self.base_dir)):
            raise PermissionError("Path traversal detected outside storage root.")

        return target_path

    def retrieve_for_authorized_processing(self, local_uri: str) -> bytes:
        """Retrieve original document bytes strictly for authorized processing workers."""
        target_path = self._uri_to_path(local_uri)
        if not target_path.exists():
            raise FileNotFoundError(f"Protected document not found at storage reference")
        return target_path.read_bytes()

    def retrieve_masked(self, local_uri: str) -> str:
        """Retrieve sanitized masked text representation."""
        target_path = self._uri_to_path(local_uri)
        if not target_path.exists():
            raise FileNotFoundError(f"Protected masked document not found at storage reference")
        return target_path.read_text(encoding="utf-8")

    def exists(self, local_uri: str) -> bool:
        """Check if document exists in protected storage."""
        try:
            target_path = self._uri_to_path(local_uri)
            return target_path.exists()
        except Exception:
            return False

    def delete(self, local_uri: str) -> bool:
        """Remove file from protected storage."""
        try:
            target_path = self._uri_to_path(local_uri)
            if target_path.exists():
                target_path.unlink()
                return True
            return False
        except Exception:
            return False


# Global default instance
default_document_storage = DocumentStorage()
