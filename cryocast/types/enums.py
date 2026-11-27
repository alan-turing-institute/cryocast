from enum import StrEnum


class BetaSchedule(StrEnum):
    """Enum for diffusion beta schedule types."""

    LINEAR = "linear"
    COSINE = "cosine"


class DiffMode(StrEnum):
    """Enum for difference-panel computation modes.

    - SIGNED: target - prediction (can be +/-, so symmetric colour scale around 0)
    - ABSOLUTE: |target - prediction| (>= 0, sequential scale)
    - SMAPE: |pred - target| / ((|pred|+|target|)/2) (>= 0, sequential scale)
    """

    SIGNED = "signed"
    ABSOLUTE = "absolute"
    SMAPE = "smape"


class Hemisphere(StrEnum):
    """Enum for hemispheres."""

    NORTH = "north"
    SOUTH = "south"


class MaskType(StrEnum):
    """Enum for types of masking."""

    ACTIVE = "active"
    LAND = "land"
    NONE = "none"


class RangeRestriction(StrEnum):
    """Enum for bounded output types."""

    CLAMP = "clamp"
    NONE = "none"
    SIGMOID = "sigmoid"
    TANH = "tanh"


class RolloutSpace(StrEnum):
    """Enum for where EncodeProcessDecode performs its forecast loop."""

    LATENT = "latent"
    PHYSICAL = "physical"


class SkipAnchorType(StrEnum):
    """Enum for the anchor that a finaliser skip connection adds to each forecast step.

    - INITIAL: every step uses the last observed frame as its anchor
    - PREVIOUS: each step uses the output of the previous step as its anchor
    """

    INITIAL = "initial"
    PREVIOUS = "previous"


class SkipConnectionType(StrEnum):
    """Enum for finaliser skip connection types."""

    ADDITIVE = "additive"
    CONVOLUTIONAL = "convolutional"
    GATED = "gated"
    NONE = "none"
