from imageio_ffmpeg import get_ffmpeg_exe
from matplotlib import rcParams

from .dataset_media_writer import DatasetMediaWriter
from .media_publisher import MediaPublisher


def register_animation_backends() -> None:
    """Register the ImageIO FFMPEG animation backend."""
    rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()


__all__ = [
    "DatasetMediaWriter",
    "MediaPublisher",
    "register_animation_backends",
]
