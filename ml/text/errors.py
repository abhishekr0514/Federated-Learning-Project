"""Structured errors for clinical text encoding (no clinical content in messages)."""


class TextInputValidationError(ValueError):
    """Missing, empty, or whitespace-only text input."""


class MaskingFailedError(RuntimeError):
    """Privacy masking or post-mask validation failed — encoding blocked."""


class ClinicalTextModelUnavailableError(RuntimeError):
    """Tokenizer or BioClinicalBERT weights could not be loaded."""


class ClinicalTextEncoderOutputError(RuntimeError):
    """Model output shape or numeric validity check failed."""
