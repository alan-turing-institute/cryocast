"""A Lightning logger that writes images, videos, and metrics to local files.

Implements the subset of the `WandbLogger` interface used by `MediaLoggingCallback`
(`log_image`/`log_video`) and Lightning's own metric logging (`log_metrics`), so a
training/evaluation job can produce local, human-inspectable artefacts (loss curves,
prediction plots) without network access or a W&B account -- e.g. in CI. Enable it
by selecting the `local_files` logger configuration.
"""

import json
import logging
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers.logger import Logger

from icenet_mp.utils import sanitise_filename

log = logging.getLogger(__name__)


class LocalFileLogger(Logger):
    """Write metrics, images, and videos to plain files under `save_dir`."""

    def __init__(
        self, save_dir: str, name: str = "local_files", **_kwargs: Any
    ) -> None:
        """Write metrics/images/videos to local files instead of an external service.

        Args:
            save_dir: Directory to write `metrics.jsonl`, `images/`, and `videos/` to.
            name: Logger name, returned by the `name` property.
            _kwargs: Ignored. Absorbs `job_type`/`project` etc. passed by
                `ModelService.build_trainer`, which are only meaningful for W&B.

        """
        super().__init__()
        self._save_dir = Path(save_dir)
        self._name = name
        self._metrics_path = self._save_dir / "metrics.jsonl"
        self._image_call_count = 0
        self._video_call_count = 0
        log.info("Run data is logged locally in %s", self._save_dir)

    @property
    def name(self) -> str:
        """Return the logger name."""
        return self._name

    @property
    def version(self) -> str:
        """Return a fixed version, since this logger does not manage run versioning."""
        return "0"

    @property
    def save_dir(self) -> str:
        """Return the root directory this logger writes to."""
        return str(self._save_dir)

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Do nothing; hyperparameters are already saved as `model_config.yaml`."""
        del params, args, kwargs

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Append a JSON line of metrics to `metrics.jsonl`."""
        self._save_dir.mkdir(parents=True, exist_ok=True)
        record = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        with self._metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def log_image(
        self,
        key: str,
        images: list[Any],
        step: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Save each image in `images` as a PNG under `save_dir/images`.

        Every call gets its own, uniquely-numbered file (like W&B's step-indexed media
        timeline) rather than overwriting by `key` alone. `MediaPublisher` reuses the
        same `key` (date + variable) on every validation epoch, since the underlying
        dates don't change, so using `key` alone silently keeps only the last epoch.
        """
        call_idx = step if step is not None else self._image_call_count
        self._image_call_count += 1
        image_dir = self._save_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(images):
            if not hasattr(image, "save"):
                log.warning("Cannot save non-image object for key '%s'.", key)
                continue
            image.save(image_dir / sanitise_filename(f"{call_idx:05d}_{key}_{idx}.png"))

    def log_video(
        self,
        key: str,
        videos: list[Any],
        step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Save each video buffer in `videos` under `save_dir/videos`.

        See `log_image` for why calls are uniquely numbered rather than keyed on `key`
        alone.
        """
        call_idx = step if step is not None else self._video_call_count
        self._video_call_count += 1
        video_dir = self._save_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        formats = kwargs.get("format") or ["mp4"] * len(videos)
        for idx, (video, video_format) in enumerate(zip(videos, formats, strict=True)):
            video.seek(0)
            (
                video_dir
                / sanitise_filename(f"{call_idx:05d}_{key}_{idx}.{video_format}")
            ).write_bytes(video.read())
