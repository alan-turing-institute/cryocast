"""Verify compatibility registration and warning configuration without hardware."""

import warnings

import pytest

from cryocast import compatibility


def test_configure_external_libraries_calls_each_registration_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All patch/registration hooks are invoked in their prescribed order."""
    calls: list[str] = []
    hooks = [
        "patch_parameter_deepcopy",
        "register_accelerators",
        "register_animation_backends",
        "register_filters",
        "register_sources",
    ]
    for hook in hooks:

        def record(*, name: str = hook) -> None:
            calls.append(name)

        monkeypatch.setattr(compatibility, hook, record)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        compatibility.configure_external_libraries()
        warnings.warn(
            "Using padding='same' with even kernel lengths and odd dilation",
            UserWarning,
            stacklevel=2,
        )
    assert calls == hooks
    assert recorded == []


def test_configure_external_libraries_does_not_hide_unrelated_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility warning rule only targets its known Torch issue."""
    for hook in (
        "patch_parameter_deepcopy",
        "register_accelerators",
        "register_animation_backends",
        "register_filters",
        "register_sources",
    ):
        monkeypatch.setattr(compatibility, hook, lambda: None)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        compatibility.configure_external_libraries()
        warnings.warn("unrelated regression warning", UserWarning, stacklevel=2)

    assert len(recorded) == 1
    assert str(recorded[0].message) == "unrelated regression warning"


@pytest.mark.parametrize(
    "failed_hook",
    [
        "patch_parameter_deepcopy",
        "register_accelerators",
        "register_animation_backends",
        "register_filters",
        "register_sources",
    ],
)
def test_failed_registration_stops_setup_and_preserves_the_original_error(
    monkeypatch: pytest.MonkeyPatch, failed_hook: str
) -> None:
    """Registration failure must propagate before later hooks or warning changes."""
    hooks = [
        "patch_parameter_deepcopy",
        "register_accelerators",
        "register_animation_backends",
        "register_filters",
        "register_sources",
    ]
    calls: list[str] = []
    error = RuntimeError("registration failed")
    for hook in hooks:

        def record(*, name: str = hook) -> None:
            calls.append(name)
            if name == failed_hook:
                raise error

        monkeypatch.setattr(compatibility, hook, record)
    with warnings.catch_warnings():
        before = list(warnings.filters)
        with pytest.raises(RuntimeError, match="registration failed") as caught:
            compatibility.configure_external_libraries()
        assert warnings.filters == before
    assert calls == hooks[: hooks.index(failed_hook) + 1]
    assert caught.value is error
