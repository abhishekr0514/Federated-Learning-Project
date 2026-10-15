"""Tabular schema and preprocessing errors."""


class TabularSchemaError(ValueError):
    """Feature names, order, version, or dimension mismatch."""


class TabularRowValidationError(ValueError):
    """Invalid or excluded row (e.g. implausible vitals)."""


class TabularTransformError(ValueError):
    """Non-finite or invalid transformed values."""
