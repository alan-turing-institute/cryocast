from collections.abc import Iterable, Iterator
from datetime import datetime as datetime_
from typing import Any

from earthkit.data.core.metadata import Metadata
from typing_extensions import override as override_

from .geographic_grid import GeographicGrid


class GeographicMetadata(Metadata):
    def __init__(self, metadata: Metadata, geography: GeographicGrid) -> None:
        """Initialise a GeographicMetadata from an EarthKit Metadata and a GeographicGrid."""
        self.metadata_ = metadata
        self.geography_ = geography

    @property
    @override_
    def geography(self) -> GeographicGrid:
        """Return the geography of the field."""
        return self.geography_

    @override_
    def __contains__(self, key: str) -> bool:
        """Return True if the key is in the metadata."""
        return key in self.metadata_

    @override_
    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the metadata keys."""
        return iter(self.metadata_)

    @override_
    def __len__(self) -> int:
        """Return the number of metadata keys."""
        return len(self.metadata_)

    @override_
    def __repr__(self) -> str:
        """Return a string representation of the GeographicMetadata, including the underlying metadata and geography."""
        return f"{self.__class__.__name__}({self.metadata_!r},{self.geography_!r})"

    @override_
    def _hide_internal_keys(self) -> Metadata:
        return self.metadata_._hide_internal_keys()

    @override_
    def as_namespace(self, namespace: str | None = None) -> dict[str, Any]:
        if hasattr(self.metadata_, "as_namespace"):
            return self.metadata_.as_namespace(namespace)
        if namespace is None or namespace in {"default", ""}:
            return {str(k): v for k, v in dict(self).items()}
        if namespace == "mars":
            return {}
        msg = f"Unsupported namespace '{namespace}'"
        raise ValueError(msg)

    @override_
    def base_datetime(self) -> datetime_:
        return self.metadata_.base_datetime()

    @override_
    def data_format(self) -> str:
        return self.metadata_.data_format()

    @override_
    def datetime(self) -> dict[str, datetime_]:
        return self.metadata_.datetime()

    @override_
    def describe_keys(self) -> list[str]:
        return self.metadata_.describe_keys()

    @override_
    def dump(self, **kwargs: Any) -> Any:
        return self.metadata_.dump(**kwargs)

    @override_
    def get(
        self,
        key: str,
        default: Any | None = None,
        *,
        astype: type | None = None,
        raise_on_missing: bool = False,
    ) -> Any:
        kwargs = {}
        if not raise_on_missing:
            kwargs["default"] = default
        result = self.metadata_.get(key, **kwargs)
        if astype is None:
            return result
        try:
            return astype(result)
        except Exception as exc:
            msg = f"Failed to convert metadata key '{key}' to type {astype}: {exc}"
            raise ValueError(msg) from exc

    @override_
    def index_keys(self) -> list[str]:
        return self.metadata_.index_keys()

    @override_
    def items(self) -> Iterable[tuple[str, Any]]:
        return self.metadata_.items()

    @override_
    def keys(self) -> Iterable[str]:
        return self.metadata_.keys()

    @override_
    def ls_keys(self) -> list[str]:
        return self.metadata_.ls_keys()

    @override_
    def namespaces(self) -> list[str]:
        return self.metadata_.namespaces()

    @override_
    def override(self, *args: Any, **kwargs: Any) -> "GeographicMetadata":
        return GeographicMetadata(
            self.metadata_.override(*args, **kwargs),
            self.geography_,
        )

    @override_
    def valid_datetime(self) -> datetime_:
        return self.metadata_.valid_datetime()
