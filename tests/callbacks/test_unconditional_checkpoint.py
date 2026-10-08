from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cryocast.callbacks.unconditional_checkpoint import UnconditionalCheckpoint


class TestDirpath:
    """Tests for the dirpath property."""

    def test_setter_ignores_falsy_value(self, tmp_path: Path) -> None:
        """Round-trip a truthy dirpath, and leave it unchanged when set to a falsy value."""
        callback = UnconditionalCheckpoint()
        callback.dirpath = tmp_path
        assert callback.dirpath == str(tmp_path)
        callback.dirpath = None
        assert callback.dirpath == str(tmp_path)


class TestSaveUnconditionally:
    """Tests for save_unconditionally."""

    def test_uses_log_dir_and_broadcast(
        self, tmp_path: Path, mock_trainer: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fall back to the trainer's log_dir and broadcast the resolved path."""
        callback = UnconditionalCheckpoint()
        format_checkpoint_name = MagicMock(return_value="epoch=2-step=7.ckpt")
        monkeypatch.setattr(
            callback.impl, "format_checkpoint_name", format_checkpoint_name
        )
        mock_trainer.current_epoch = 2
        mock_trainer.global_step = 7
        mock_trainer.log_dir = str(tmp_path)
        mock_trainer.strategy.broadcast.side_effect = lambda value: value

        callback.save_unconditionally(mock_trainer)

        format_checkpoint_name.assert_called_once()
        expected = tmp_path / "epoch=2-step=7.ckpt"
        mock_trainer.strategy.broadcast.assert_called_once_with(str(expected))
        mock_trainer.save_checkpoint.assert_called_once_with(
            expected, weights_only=False
        )

    def test_prefers_explicit_dirpath(
        self, tmp_path: Path, mock_trainer: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prefer an explicitly configured dirpath over the trainer's log_dir."""
        callback = UnconditionalCheckpoint()
        explicit = tmp_path / "checkpoints"
        callback.dirpath = explicit
        monkeypatch.setattr(
            callback.impl, "format_checkpoint_name", MagicMock(return_value="last.ckpt")
        )
        mock_trainer.current_epoch = 0
        mock_trainer.global_step = 0
        mock_trainer.log_dir = str(tmp_path / "logs")
        mock_trainer.strategy.broadcast.side_effect = lambda value: value

        callback.save_unconditionally(mock_trainer)

        mock_trainer.save_checkpoint.assert_called_once_with(
            explicit / "last.ckpt", weights_only=False
        )

    def test_uses_absolute_checkpoint_name_as_is(
        self, tmp_path: Path, mock_trainer: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not prepend a directory when the checkpoint name is already absolute."""
        callback = UnconditionalCheckpoint()
        absolute_checkpoint = tmp_path / "absolute.ckpt"
        monkeypatch.setattr(
            callback.impl,
            "format_checkpoint_name",
            MagicMock(return_value=str(absolute_checkpoint)),
        )
        mock_trainer.current_epoch = 0
        mock_trainer.global_step = 0
        mock_trainer.log_dir = str(tmp_path / "logs")
        mock_trainer.strategy.broadcast.side_effect = lambda value: value

        callback.save_unconditionally(mock_trainer)

        mock_trainer.save_checkpoint.assert_called_once_with(
            absolute_checkpoint, weights_only=False
        )

    def test_uses_relative_name_when_no_dirpath_available(
        self, mock_trainer: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fall back to the raw relative checkpoint name when no directory is available."""
        callback = UnconditionalCheckpoint()
        monkeypatch.setattr(
            callback.impl, "format_checkpoint_name", MagicMock(return_value="last.ckpt")
        )
        mock_trainer.current_epoch = 0
        mock_trainer.global_step = 0
        mock_trainer.log_dir = None
        mock_trainer.strategy.broadcast.side_effect = lambda value: value

        callback.save_unconditionally(mock_trainer)

        mock_trainer.save_checkpoint.assert_called_once_with(
            Path("last.ckpt"), weights_only=False
        )


class TestOnTrainEnd:
    """Tests for on_train_end."""

    def test_disabled_by_default_does_not_save(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Do not save a checkpoint when on_train_end is not opted in."""
        callback = UnconditionalCheckpoint(on_train_end=False)
        save = MagicMock()
        monkeypatch.setattr(callback, "save_unconditionally", save)

        callback.on_train_end(mock_trainer, mock_module)

        save.assert_not_called()

    def test_enabled_saves_unconditionally(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save a checkpoint when on_train_end is opted in."""
        callback = UnconditionalCheckpoint(on_train_end=True)
        save = MagicMock()
        monkeypatch.setattr(callback, "save_unconditionally", save)

        callback.on_train_end(mock_trainer, mock_module)

        save.assert_called_once_with(mock_trainer)
