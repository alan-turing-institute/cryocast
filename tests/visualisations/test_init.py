from matplotlib import rcParams

from icenet_mp.visualisations import register_animation_backends


def test_register_animation_backends_sets_ffmpeg_path() -> None:
    """Register a non-empty path to the bundled imageio-ffmpeg executable."""
    register_animation_backends()

    assert rcParams["animation.ffmpeg_path"]
