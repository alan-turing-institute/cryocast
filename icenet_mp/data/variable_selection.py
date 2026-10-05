from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property


class VariableSelection:
    """Validated view over a `variables` config section for one set of dataset groups.

    Turns the requested `variables.input`/`variables.target` mappings into per-group
    variable lists, checked against the configured dataset groups and each group's
    on-disk variable names. This class does no I/O itself: callers pass in each
    group's on-disk variable names (typically read from a `SingleDataset`).
    """

    def __init__(
        self,
        dataset_group_names: Iterable[str],
        input_variables: Mapping[str, Sequence[str]],
        target_variables: Mapping[str, Sequence[str]],
    ) -> None:
        """Store the requested `variables.input`/`variables.target` config sections."""
        self._dataset_group_names = frozenset(dataset_group_names)
        self._requested_input_variables: dict[str, list[str]] = {
            str(group_name): [str(v) for v in variable_names]
            for group_name, variable_names in input_variables.items()
        }
        self._requested_target_variables: dict[str, list[str]] = {
            str(group_name): [str(v) for v in variable_names]
            for group_name, variable_names in target_variables.items()
        }

    @cached_property
    def target_group_name(self) -> str:
        """Return the name of the single configured target variable group."""
        # Verify that exactly one target variable group is requested
        target_variable_groups = list(self._requested_target_variables.keys())
        if len(target_variable_groups) != 1:
            msg = (
                f"Expected exactly one target variable group, but found "
                f"{len(target_variable_groups)}: {target_variable_groups}."
            )
            raise ValueError(msg)
        # Verify that the requested group is a configured dataset group
        if target_variable_groups[0] not in self._dataset_group_names:
            available_ds_groups = (
                ", ".join(sorted(self._dataset_group_names)) or "<none>"
            )
            msg = (
                f"Target dataset group {target_variable_groups[0]!r} is not a "
                f"configured dataset group. Available groups: {available_ds_groups}."
            )
            raise ValueError(msg)
        return target_variable_groups[0]

    def filter_requested(
        self, variables_by_dataset: Mapping[str, Sequence[str]]
    ) -> dict[str, list[str]]:
        """Return the requested input variable names for each input dataset group.

        Args:
            variables_by_dataset: Mapping of dataset group name to the on-disk variable
                names for that group. This is typically read from a `SingleDataset` for
                each group.

        Returns:
            Mapping of dataset group name to the requested input variable names for that
            group, in the order they appear on-disk.

        """
        if not self._requested_input_variables:
            return {
                group_name: list(variable_names)
                for group_name, variable_names in variables_by_dataset.items()
            }
        verified: dict[str, list[str]] = {}
        for group_name, variable_names in self._requested_input_variables.items():
            # Verify that the requested group is a configured dataset group
            if group_name not in self._dataset_group_names:
                available_ds_groups = (
                    ", ".join(sorted(self._dataset_group_names)) or "<none>"
                )
                msg = (
                    f"Input dataset group {group_name!r} is not a configured dataset "
                    f"group. Available groups: {available_ds_groups}."
                )
                raise ValueError(msg)
            # Verify that the requested variable names exist in the dataset group
            available_variables = variables_by_dataset[group_name]
            for variable in variable_names:
                if variable not in available_variables:
                    available_ = ", ".join(sorted(available_variables)) or "<none>"
                    msg = (
                        f"Input variable {variable!r} was not found in dataset group "
                        f"{group_name!r}. Available variables: {available_}."
                    )
                    raise ValueError(msg)
            verified[group_name] = [
                v for v in available_variables if v in variable_names
            ]
        # Verify that every target variable is also requested as an input variable
        requested_inputs = self._requested_input_variables.get(
            self.target_group_name, []
        )
        for variable in self._requested_target_variables[self.target_group_name]:
            if variable not in requested_inputs:
                msg = (
                    f"Target variable {variable!r} in dataset group "
                    f"{self.target_group_name!r} must also be listed under "
                    f"variables.input.{self.target_group_name}."
                )
                raise ValueError(msg)
        return verified

    def target_variables(
        self, on_disk_target_variable_names: Sequence[str]
    ) -> list[str]:
        """Return the names of the variables to predict, in on-disk order.

        `on_disk_target_variable_names` must be the target group's already
        input-filtered, on-disk-ordered variable names (i.e. the target
        `SingleDataset.variable_names` after applying `variables_by_dataset`).
        """
        # An empty target selection means every variable in the target group
        requested_variables = self._requested_target_variables[self.target_group_name]
        if not requested_variables:
            return list(on_disk_target_variable_names)
        # Verify that the requested variable names exist in the dataset group
        for requested_variable in requested_variables:
            if requested_variable not in on_disk_target_variable_names:
                available_ = (
                    ", ".join(sorted(on_disk_target_variable_names)) or "<none>"
                )
                msg = (
                    f"Target variable {requested_variable!r} was not found in dataset "
                    f"group {self.target_group_name!r}. Available variables: "
                    f"{available_}."
                )
                raise ValueError(msg)
        return [v for v in on_disk_target_variable_names if v in requested_variables]
