import logging
from pathlib import Path

import numpy as np
import pytest

from icenet_mp.visualisations.land_mask import LandMask


class TestLandMaskConstruction:
    def test_no_path_leaves_cache_empty(self) -> None:
        """LandMask(None) should not raise, and apply_to returns arrays unchanged."""
        land_mask = LandMask(None)
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.testing.assert_array_equal(land_mask.apply_to(data), data)

    def test_nonexistent_path_leaves_cache_empty(self, tmp_path: Path) -> None:
        """A path that doesn't exist should be skipped silently, no load attempted."""
        missing_path = tmp_path / "does_not_exist.npy"
        land_mask = LandMask(missing_path)
        assert land_mask._cache == {}

        data = np.ones((4, 4), dtype=np.float32)
        np.testing.assert_array_equal(land_mask.apply_to(data), data)

    def test_successful_load_populates_cache(self, tmp_path: Path) -> None:
        """A valid .npy file at the given path should be loaded into the cache.

        Mask semantics per apply_to's implementation (np.where(~mask, nan, data)):
        True positions are kept, False positions become NaN.
        """
        mask_path = tmp_path / "land_mask.npy"
        mask_array = np.array(
            [[True, False], [False, True]],
        )
        np.save(mask_path, mask_array)

        land_mask = LandMask(mask_path)

        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = land_mask.apply_to(data)
        expected = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_failed_load_invalid_file_logs_warning_and_does_not_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed .npy file should be swallowed with a warning, not raised."""
        bad_path = tmp_path / "corrupt.npy"
        bad_path.write_bytes(b"not a valid numpy file")

        with caplog.at_level(logging.WARNING):
            land_mask = LandMask(bad_path)

        assert land_mask._cache == {}
        assert "Failed to load land mask" in caplog.text

    def test_failed_load_via_monkeypatched_np_load(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An OSError from np.load should also be swallowed with a warning."""
        mask_path = tmp_path / "land_mask.npy"
        np.save(mask_path, np.array([[True, False]]))

        def raise_oserror(*_args: object, **_kwargs: object) -> np.ndarray:
            msg = "disk read failed"
            raise OSError(msg)

        monkeypatch.setattr(np, "load", raise_oserror)

        with caplog.at_level(logging.WARNING):
            land_mask = LandMask(mask_path)

        assert land_mask._cache == {}
        assert "Failed to load land mask" in caplog.text


class TestLandMaskApplyTo:
    def test_true_positions_kept_false_positions_become_nan(self) -> None:
        """The mask is inverted internally: True positions are kept, False -> NaN."""
        land_mask = LandMask(None)
        mask_array = np.array([[True, True, False, False], [True, True, False, False]])
        land_mask.add_mask(mask_array)

        data = np.ones((2, 4), dtype=np.float32) * 5.0
        result = land_mask.apply_to(data)

        assert np.all(result[:, :2] == 5.0)
        assert np.all(np.isnan(result[:, 2:]))

    def test_all_true_mask_leaves_array_unchanged(self) -> None:
        land_mask = LandMask(None)
        land_mask.add_mask(np.ones((3, 3), dtype=bool))
        data = np.full((3, 3), 7.0, dtype=np.float32)
        result = land_mask.apply_to(data)
        np.testing.assert_array_equal(result, data)

    def test_all_false_mask_produces_all_nan(self) -> None:
        land_mask = LandMask(None)
        land_mask.add_mask(np.zeros((3, 3), dtype=bool))
        data = np.full((3, 3), 7.0, dtype=np.float32)
        result = land_mask.apply_to(data)
        assert np.all(np.isnan(result))

    def test_no_matching_mask_returns_array_unchanged(self) -> None:
        """A mask registered for a different shape must not be applied cross-shape."""
        land_mask = LandMask(None)
        land_mask.add_mask(np.ones((3, 3), dtype=bool))

        data = np.full((4, 4), 9.0, dtype=np.float32)
        result = land_mask.apply_to(data)
        np.testing.assert_array_equal(result, data)

    def test_apply_to_uses_trailing_two_dims_for_3d_array(self) -> None:
        """apply_to should key on data_array.shape[-2:], ignoring a leading time dim."""
        land_mask = LandMask(None)
        mask_array = np.array([[True, False], [False, True]])
        land_mask.add_mask(mask_array)

        data = np.full((3, 2, 2), 2.0, dtype=np.float32)
        result = land_mask.apply_to(data)

        assert result.shape == (3, 2, 2)
        for t in range(3):
            assert result[t, 0, 0] == 2.0
            assert np.isnan(result[t, 0, 1])
            assert np.isnan(result[t, 1, 0])
            assert result[t, 1, 1] == 2.0

    def test_debug_log_for_unmatched_shape_logged_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The 'no land mask' debug message should only be logged once per shape."""
        land_mask = LandMask(None)
        data = np.ones((5, 5), dtype=np.float32)

        with caplog.at_level(logging.DEBUG):
            land_mask.apply_to(data)
            land_mask.apply_to(data)

        matching_records = [
            record
            for record in caplog.records
            if "No land mask associated" in record.message
        ]
        assert len(matching_records) == 1
