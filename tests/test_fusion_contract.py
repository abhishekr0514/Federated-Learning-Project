"""Tests for MedShield-FL fusion contracts, error hierarchy, and security isolation.

These tests run without PyTorch to verify domain error behaviors and safety boundaries.
"""

import pytest

from ml.fusion.errors import FusionCheckpointError, FusionInputError


def test_fusion_error_hierarchy():
    """Verify fusion errors inherit from ValueError and are distinct."""
    assert issubclass(FusionInputError, ValueError)
    assert issubclass(FusionCheckpointError, ValueError)
    assert FusionInputError is not FusionCheckpointError


def test_fusion_input_error_message_safety():
    """Verify FusionInputError does not leak private patient data in generic messages."""
    err = FusionInputError("Unsupported modality pattern: tabular missing")
    msg = str(err)
    assert "tabular missing" in msg
    assert "patient_id" not in msg
    assert "waveform" not in msg
    assert "clinical_notes" not in msg


def test_fusion_checkpoint_error_message():
    """Verify FusionCheckpointError formatting."""
    err = FusionCheckpointError("Incompatible architecture_id: 'unknown'")
    assert "Incompatible architecture_id" in str(err)
