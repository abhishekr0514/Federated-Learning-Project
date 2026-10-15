"""Multimodal fusion input and checkpoint errors."""


class FusionInputError(ValueError):
    """Invalid embeddings, masks, or unsupported modality combination."""


class FusionCheckpointError(ValueError):
    """Checkpoint metadata or weights incompatible with the fusion architecture."""
