import pytest
from anemoi.datasets.usage import store

from cryocast.compatibility.anemoi import patch_collect_analytics
from cryocast.compatibility.anemoi.patch_collect_analytics import _no_analytics


class TestPatchCollectAnalytics:
    def test_replaces_collect_analytics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list = []
        monkeypatch.setattr(store, "collect_analytics", lambda *a: calls.append(a))
        patch_collect_analytics()
        assert store.collect_analytics is _no_analytics
        store.collect_analytics("open_zarr_store")
        assert calls == []

    def test_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(store, "collect_analytics", store.collect_analytics)
        patch_collect_analytics()
        patch_collect_analytics()
        assert store.collect_analytics is _no_analytics
