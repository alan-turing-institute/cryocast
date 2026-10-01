import pytest

from icenet_mp.types import (
    BetaSchedule,
    DiffMode,
    Hemisphere,
    MaskType,
    RangeRestriction,
    SkipConnectionType,
)


class TestEnums:
    """Tests for the icenet_mp.types StrEnum classes."""

    def test_beta_schedule_members(self) -> None:
        """Expose the expected BetaSchedule members."""
        assert {member.value for member in BetaSchedule} == {"linear", "cosine"}

    def test_diff_mode_members(self) -> None:
        """Expose the expected DiffMode members."""
        assert {member.value for member in DiffMode} == {
            "signed",
            "absolute",
            "smape",
        }

    def test_hemisphere_members(self) -> None:
        """Expose the expected Hemisphere members."""
        assert {member.value for member in Hemisphere} == {"north", "south"}

    def test_mask_type_members(self) -> None:
        """Expose the expected MaskType members."""
        assert {member.value for member in MaskType} == {"active", "land", "none"}

    @pytest.mark.parametrize(
        "member",
        [
            BetaSchedule.LINEAR,
            DiffMode.SIGNED,
            Hemisphere.NORTH,
            MaskType.LAND,
            RangeRestriction.SIGMOID,
            SkipConnectionType.GATED,
        ],
        ids=lambda member: f"{type(member).__name__}.{member.name}",
    )
    def test_member_behaves_as_its_string_value(
        self,
        member: BetaSchedule
        | DiffMode
        | Hemisphere
        | MaskType
        | RangeRestriction
        | SkipConnectionType,
    ) -> None:
        """Compare equal to, and behave as, its underlying string value."""
        assert member == member.value
        assert isinstance(member, str)

    def test_range_restriction_members(self) -> None:
        """Expose the expected RangeRestriction members."""
        assert {member.value for member in RangeRestriction} == {
            "clamp",
            "none",
            "sigmoid",
            "tanh",
        }

    def test_skip_connection_type_members(self) -> None:
        """Expose the expected SkipConnectionType members."""
        assert {member.value for member in SkipConnectionType} == {
            "additive",
            "convolutional",
            "gated",
            "none",
        }
