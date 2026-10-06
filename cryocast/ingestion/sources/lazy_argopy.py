"""Lazy-loading wrapper for argopy.

Importing argopy at module level triggers a network request to ERDDAP servers. As this
can raise an exception, we only want to perform this import when argopy is actually
needed.

We therefore add a thin __getattr__ wrapper which imports the real argopy module.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argopy import DataFetcher
    from argopy.errors import NoData


def __getattr__(name: str) -> object:
    """Defer argopy imports until first access (PEP 562)."""
    if name == "DataFetcher":
        from argopy import DataFetcher  # noqa: PLC0415

        globals()["DataFetcher"] = DataFetcher
        return DataFetcher
    if name == "NoData":
        from argopy.errors import NoData  # noqa: PLC0415

        globals()["NoData"] = NoData
        return NoData
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["DataFetcher", "NoData"]
