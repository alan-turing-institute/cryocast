from pickle import UnpicklingError


class PlottingError(RuntimeError): ...


class VideoRenderError(PlottingError): ...


class InvalidArrayError(PlottingError, ValueError): ...


class CheckpointError(RuntimeError):
    """An error related to loading or saving a model checkpoint."""


class OutdatedCheckpointError(UnpicklingError, CheckpointError):
    """A checkpoint that must be upgraded before it can be loaded safely."""


class UntrustedCheckpointError(OutdatedCheckpointError):
    """A checkpoint that can only be upgraded by trusting it to run arbitrary code."""


class CheckpointUpgradeError(CheckpointError):
    """A checkpoint that could not be upgraded into a safely loadable format."""
