import pytest

from icenet_mp.data.variable_selection import VariableSelection


class TestTargetGroupName:
    """Exactly one target group must be requested, and it must be a real dataset group."""

    def test_returns_the_single_target_group(self) -> None:
        selection = VariableSelection(
            dataset_group_names=["group1", "group2"],
            input_variables={"group1": ["a"], "group2": ["b"]},
            target_variables={"group1": ["a"]},
        )
        assert selection.target_group_name == "group1"

    def test_no_target_groups_raises(self) -> None:
        selection = VariableSelection(
            dataset_group_names=["group1"],
            input_variables={"group1": ["a"]},
            target_variables={},
        )
        with pytest.raises(ValueError, match="exactly one target variable group"):
            _ = selection.target_group_name

    def test_multiple_target_groups_raises(self) -> None:
        selection = VariableSelection(
            dataset_group_names=["group1", "group2"],
            input_variables={"group1": ["a"], "group2": ["b"]},
            target_variables={"group1": ["a"], "group2": ["b"]},
        )
        with pytest.raises(ValueError, match="exactly one target variable group"):
            _ = selection.target_group_name

    def test_target_group_not_a_dataset_group_raises(self) -> None:
        selection = VariableSelection(
            dataset_group_names=["group1"],
            input_variables={"group1": ["a"]},
            target_variables={"not-a-group": ["a"]},
        )
        with pytest.raises(ValueError, match="not-a-group") as exc_info:
            _ = selection.target_group_name
        assert "group1" in str(exc_info.value)


class TestRequestedVariableNames:
    """`variables.input` selections must be validated against on-disk variable names."""

    def test_empty_selection_returns_every_group_in_full(self) -> None:
        """No `variables.input` at all means every group's full on-disk list is used."""
        selection = VariableSelection(
            dataset_group_names=["group1", "group2"],
            input_variables={},
            target_variables={"group1": ["a"]},
        )
        result = selection.filter_requested({"group1": ["a", "b"], "group2": ["c"]})
        assert result == {"group1": ["a", "b"], "group2": ["c"]}

    def test_explicit_selection_returns_on_disk_order(self) -> None:
        """Returned lists follow on-disk order, not the order requested in config."""
        selection = VariableSelection(
            dataset_group_names=["group1"],
            input_variables={"group1": ["c", "a"]},
            target_variables={"group1": ["a"]},
        )
        result = selection.filter_requested({"group1": ["a", "b", "c"]})
        assert result == {"group1": ["a", "c"]}

    def test_explicit_selection_only_includes_requested_groups(self) -> None:
        """A group omitted from `variables.input` is absent from the result entirely."""
        selection = VariableSelection(
            dataset_group_names=["group1", "group2"],
            input_variables={"group2": ["b"]},
            target_variables={"group2": ["b"]},
        )
        result = selection.filter_requested({"group1": ["a"], "group2": ["b"]})
        assert result == {"group2": ["b"]}

    def test_unknown_input_group_raises(self) -> None:
        selection = VariableSelection(
            dataset_group_names=["group1"],
            input_variables={"not-a-group": ["a"]},
            target_variables={"group1": ["a"]},
        )
        with pytest.raises(ValueError, match="not-a-group") as exc_info:
            selection.filter_requested({"group1": ["a"]})
        assert "group1" in str(exc_info.value)

    def test_unknown_input_variable_raises(self) -> None:
        selection = VariableSelection(
            dataset_group_names=["group1"],
            input_variables={"group1": ["not-a-variable"]},
            target_variables={"group1": ["a"]},
        )
        with pytest.raises(ValueError, match="not-a-variable") as exc_info:
            selection.filter_requested({"group1": ["a"]})
        assert "a" in str(exc_info.value)

    def test_target_group_missing_from_inputs_raises(self) -> None:
        """The target group must appear in an explicit `variables.input` selection."""
        selection = VariableSelection(
            dataset_group_names=["group1", "group2"],
            input_variables={"group2": ["b"]},
            target_variables={"group1": ["a"]},
        )
        with pytest.raises(ValueError, match="must also be listed under") as exc_info:
            selection.filter_requested({"group1": ["a"], "group2": ["b"]})
        assert "variables.input.group1" in str(exc_info.value)

    def test_target_variable_missing_from_inputs_raises(self) -> None:
        """Each target variable must be in its group's `variables.input` list."""
        selection = VariableSelection(
            dataset_group_names=["group1"],
            input_variables={"group1": ["a"]},
            target_variables={"group1": ["b"]},
        )
        with pytest.raises(ValueError, match=r"'b'.*must also be listed under"):
            selection.filter_requested({"group1": ["a", "b"]})


class TestTargetVariables:
    """The target selection must be resolvable on disk; empty means all variables."""

    def test_returns_on_disk_order_filtered_to_requested(self) -> None:
        """Returned order follows the given on-disk list, not the request order."""
        selection = VariableSelection(
            dataset_group_names=["group1"],
            input_variables={"group1": ["a", "b", "c"]},
            target_variables={"group1": ["c", "a"]},
        )
        assert selection.target_variables(["a", "b", "c"]) == ["a", "c"]

    def test_empty_target_selection_returns_every_variable(self) -> None:
        """An empty target selection keeps every variable, in on-disk order."""
        selection = VariableSelection(
            dataset_group_names=["group1"],
            input_variables={"group1": ["a", "b"]},
            target_variables={"group1": []},
        )
        assert selection.target_variables(["a", "b"]) == ["a", "b"]

    def test_unknown_target_variable_raises(self) -> None:
        selection = VariableSelection(
            dataset_group_names=["group1"],
            input_variables={"group1": ["a"]},
            target_variables={"group1": ["not-a-variable"]},
        )
        with pytest.raises(ValueError, match="not-a-variable") as exc_info:
            selection.target_variables(["a"])
        assert "a" in str(exc_info.value)
