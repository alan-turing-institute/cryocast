from matplotlib import rcParams

from cryocast.visualisations import register_animation_backends


class TestRegisterAnimationBackends:
    """Tests for register_animation_backends."""

    def test_sets_ffmpeg_path(self) -> None:
        """Register a non-empty path to the bundled imageio-ffmpeg executable."""
        register_animation_backends()

        assert rcParams["animation.ffmpeg_path"]
