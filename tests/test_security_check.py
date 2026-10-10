"""Test suite verifying absence of hardcoded secrets in source files."""

import os
import re
from pathlib import Path

# Pattern to detect suspicious hardcoded secret assignments
SUSPICIOUS_PATTERNS = [
    r'(?i)api[_-]?key\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']',
    r'(?i)private[_-]?key\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']',
    r'(?i)aws[_-]?secret\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']',
]

PROJECT_ROOT = Path(__file__).parent.parent


def test_no_hardcoded_secrets_in_source():
    """Scan Python source files for potential hardcoded secret patterns."""
    for root, _, files in os.walk(PROJECT_ROOT):
        # Exclude git, venv, and cache directories
        if any(ignored in root for ignored in [".git", "venv", ".venv", "__pycache__", "node_modules", ".pytest_cache"]):
            continue

        for file in files:
            if file.endswith(".py") and file != "test_security_check.py":
                file_path = Path(root) / file
                content = file_path.read_text(encoding="utf-8")
                
                for pattern in SUSPICIOUS_PATTERNS:
                    match = re.search(pattern, content)
                    assert match is None, f"Potential hardcoded secret found in {file_path}: {match.group(0)}"
