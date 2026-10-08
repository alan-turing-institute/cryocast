from pickle import UnpicklingError


class PlottingError(RuntimeError): ...


class VideoRenderError(PlottingError): ...


class InvalidArrayError(PlottingError, ValueError): ...


class OutdatedCheckpointError(UnpicklingError):
    """A checkpoint that must be upgraded before it can be loaded safely."""
