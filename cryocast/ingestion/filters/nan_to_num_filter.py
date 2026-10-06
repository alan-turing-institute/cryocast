"""Filter that replaces all NaN values in the 'variables' list with the 'replace_with' value."""

import earthkit.data as ekd
import numpy as np
from anemoi.transform.filter import SingleFieldFilter


class NanToNumFilter(SingleFieldFilter):
    required_inputs = ("variables", "replace_with")

    def forward_select(self) -> dict[str, str | list[str] | tuple[str]]:
        """Select which fields to transform."""
        return {"param": list(self.variables)}

    def forward_transform(self, field: ekd.Field) -> ekd.Field:
        """A forward transform that replaces NaNs in the input field with 'replace_with'."""
        return self.new_field_from_numpy(
            np.nan_to_num(field.to_numpy().reshape(field.shape), nan=self.replace_with),
            template=field,
        )
