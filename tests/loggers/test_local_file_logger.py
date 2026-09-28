import json
import logging
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

from icenet_mp.loggers import LocalFileLogger


def test_logger_metadata_uses_configured_name_and_directory(tmp_path: Path) -> None:
    """Expose stable Lightning logger metadata without creating files eagerly."""
    logger = LocalFileLogger(str(tmp_path), name="offline")

    assert logger.name == "offline"
    assert logger.version == "0"
    assert logger.save_dir == str(tmp_path)
    assert not tmp_path.joinpath("metrics.jsonl").exists()


def test_log_hyperparams_is_a_noop(tmp_path: Path) -> None:
    """Accept and discard hyperparameters without touching the filesystem."""
    logger = LocalFileLogger(str(tmp_path))

    logger.log_hyperparams({"lr": 1e-3}, "unused_arg", extra="unused_kwarg")

    assert list(tmp_path.iterdir()) == []


def test_log_metrics_appends_json_lines_and_coerces_values(tmp_path: Path) -> None:
    """Persist one JSON record per logging call with the optional training step."""
    logger = LocalFileLogger(str(tmp_path))

    logger.log_metrics({"loss": 1, "accuracy": 0.75}, step=4)
    logger.log_metrics({"loss": 0.5})

    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert records == [
        {"step": 4, "loss": 1.0, "accuracy": 0.75},
        {"step": None, "loss": 0.5},
    ]


def test_log_image_uses_unique_sanitised_filenames(tmp_path: Path) -> None:
    """Keep repeated image keys distinct and safe for local filesystem paths."""
    logger = LocalFileLogger(str(tmp_path))
    first = MagicMock()
    second = MagicMock()

    logger.log_image("validation/output static:sic", [first])
    logger.log_image("validation/output static:sic", [second])

    first_path = first.save.call_args.args[0]
    second_path = second.save.call_args.args[0]
    assert isinstance(first_path, Path)
    assert isinstance(second_path, Path)
    assert first_path.name == "00000_validation_output_static_sic_0.png"
    assert second_path.name == "00001_validation_output_static_sic_0.png"
    assert first_path.parent == tmp_path / "images"


def test_log_image_respects_explicit_step(tmp_path: Path) -> None:
    """Use a supplied logging step in the persisted image filename."""
    logger = LocalFileLogger(str(tmp_path))
    image = MagicMock()

    logger.log_image("forecast", [image], step=23)

    assert image.save.call_args.args[0].name == "00023_forecast_0.png"


def test_log_image_skips_non_image_objects(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    """Warn and continue when an object does not provide an image save method."""
    logger = LocalFileLogger(str(tmp_path))

    with caplog.at_level(logging.WARNING):
        logger.log_image("bad/image", [object()])

    assert "Cannot save non-image object" in caplog.text


def test_log_video_rewinds_and_writes_requested_format(tmp_path: Path) -> None:
    """Rewind each buffer before persisting it with the requested extension."""
    logger = LocalFileLogger(str(tmp_path))
    video = BytesIO(b"video-bytes")
    video.seek(5)

    logger.log_video("forecast/video", [video], step=7, format=["gif"])

    output = tmp_path / "videos" / "00007_forecast_video_0.gif"
    assert output.read_bytes() == b"video-bytes"
    assert video.tell() == len(b"video-bytes")


def test_log_video_defaults_to_mp4_and_increments_call_index(tmp_path: Path) -> None:
    """Use MP4 by default and assign a new call index to repeated keys."""
    logger = LocalFileLogger(str(tmp_path))

    logger.log_video("forecast", [BytesIO(b"one")])
    logger.log_video("forecast", [BytesIO(b"two")])

    assert (tmp_path / "videos" / "00000_forecast_0.mp4").read_bytes() == b"one"
    assert (tmp_path / "videos" / "00001_forecast_0.mp4").read_bytes() == b"two"
