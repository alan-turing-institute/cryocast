"""Tests for physical-space rollout and residual (tendency) prediction.

Background — the two defects these options exist to fix, both in the historical
"latent" rollout path and both measured as real failures on 2026-07-29:

1. NO IDENTITY PATH. With the absolute parameterisation, reproducing the input
   requires pushing it through a downsampling CNN, a patch-embedding ViT and an
   upsampling CNN, so persistence is not representable. Measured on the toy: the
   model scores active-cell MAE 0.093 at lead 1 where copying the input scores
   0.034. An additive decoder skip connection during a physical rollout makes
   delta=0 exactly persistence.

2. UNCONSTRAINED LATENT FEEDBACK. `BaseProcessor.rollout` appends the processor's
   own output latent to its own input window, never re-encodes it, and never
   constrains it to the encoders' output distribution; in a multi-input model most
   of those fed-back channels are never supervised at all. The measured symptom is
   a near-fixed-point map whose forecast stops evolving with lead.
   `rollout_space="physical"` closes the loop in observation space.

The properties tested here, in order of importance:

* OFF is the default and byte-identical to the pre-feature model.
* A zero-output residual network reproduces persistence EXACTLY at every lead.
* The physical rollout genuinely advances the state (successive leads differ, and
  each lead's field depends on the previous prediction).
* No future information can enter through either option.
"""

from typing import Any, cast

import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models import EncodeProcessDecode

TARGET_GROUP = "sic-ssmis"
SEED = 1234


def _build_model(
    *,
    rollout_space: str = "latent",
    decoder_extra: dict[str, Any] | None = None,
    processor: dict[str, Any] | None = None,
    grid: int = 32,
    latent: int = 16,
    n_forecast_steps: int = 4,
    n_history_steps: int = 3,
    input_channels: int = 1,
    output_channels: int = 1,
    extra_inputs: list[DictConfig] | None = None,
    target_variable_indices: list[int] | None = None,
    seed: int = SEED,
) -> EncodeProcessDecode:
    input_spaces = [
        DictConfig(
            {"channels": input_channels, "name": TARGET_GROUP, "shape": (grid, grid)}
        ),
        *(extra_inputs or []),
    ]
    encoders = DictConfig(
        {
            "latent_space": (latent, latent),
            **{
                space["name"]: {
                    "_target_": "icenet_mp.models.encoders.CNNEncoder",
                    "n_layers": 1,
                }
                for space in input_spaces
            },
        }
    )
    decoder_payload: dict[str, Any] = {
        "_target_": "icenet_mp.models.decoders.CNNDecoder",
        "n_layers": 1,
    }
    decoder_payload.update(decoder_extra or {})
    decoder = DictConfig(decoder_payload)
    torch.manual_seed(seed)
    return EncodeProcessDecode(
        name="cnn-null-cnn",
        encoders=encoders,
        processor=DictConfig(
            processor or {"_target_": "icenet_mp.models.processors.NullProcessor"}
        ),
        decoder=decoder,
        rollout_space=rollout_space,
        hemisphere="north",
        input_spaces=input_spaces,
        n_forecast_steps=n_forecast_steps,
        n_history_steps=n_history_steps,
        output_space=DictConfig(
            {"channels": output_channels, "name": TARGET_GROUP, "shape": (grid, grid)}
        ),
        optimizer=DictConfig({}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
        loss=DictConfig({"_target_": "torch.nn.HuberLoss", "delta": 0.5}),
        # Required since #396: the per-forecast-day metrics BaseModel builds. Two
        # cheap ones, matching the `cfg_metrics` fixture used by main's own tests.
        metrics=["accuracy", "mae"],
        # Required since #405: which variable(s) of the target INPUT group are the
        # prediction target. output_space is single-channel throughout these tests,
        # so [0] satisfies the channel-count check; the feedback-channel tests pass
        # their own index so the two stay consistent.
        target_variable_indices=target_variable_indices or [0],
    )


def _inputs(
    model: EncodeProcessDecode,
    *,
    batch_size: int = 2,
    seed: int = 7,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        space.name: torch.rand(
            batch_size,
            model.n_history_steps,
            space.channels,
            *space.shape,
            generator=generator,
        )
        for space in model.input_spaces
    }


def _final_conv(model: EncodeProcessDecode) -> torch.nn.Conv2d:
    """The decoder's final convolution, with the cast mypy needs for indexing."""
    last = cast("torch.nn.Sequential", model.decoder.model)[-1]
    assert isinstance(last, torch.nn.Conv2d)
    return last


def _zero_decoder_output(model: EncodeProcessDecode) -> None:
    """Force the decoder to emit exactly zero, so the residual is zero everywhere."""
    last = _final_conv(model)
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()


class TestDefaultsOff:
    """Neither option may change anything unless explicitly switched on."""

    def test_defaults(self) -> None:
        model = _build_model()
        assert model.rollout_space == "latent"
        assert model.target_variable_indices == [0]

    def test_off_is_identical_to_absent(self) -> None:
        absent = _build_model()
        explicit = _build_model(rollout_space="latent")
        inputs = _inputs(absent)
        absent.eval()
        explicit.eval()
        with torch.no_grad():
            assert torch.equal(absent(inputs), explicit(dict(inputs)))

    def test_state_dicts_match(self) -> None:
        """The options add no parameters, so checkpoints stay interchangeable."""
        latent = _build_model()
        physical = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
        )
        assert set(latent.state_dict()) == set(physical.state_dict())
        assert sum(p.numel() for p in latent.parameters()) == sum(
            p.numel() for p in physical.parameters()
        )


class TestResidualIsExactlyPersistence:
    """The property that makes persistence reachable: zero tendency = persistence."""

    def test_zero_tendency_reproduces_persistence_at_every_lead(self) -> None:
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
        )
        _zero_decoder_output(model)
        inputs = _inputs(model)
        persistence = inputs[TARGET_GROUP][:, -1]  # newest observed frame
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
        assert prediction.shape[1] == model.n_forecast_steps
        for lead in range(model.n_forecast_steps):
            assert torch.equal(prediction[:, lead], persistence), (
                f"lead {lead + 1} is not exactly persistence"
            )

    def test_absolute_parameterisation_cannot_do_this(self) -> None:
        """Contrast: with a zeroed decoder the absolute path emits zeros, not the input.

        This is the failure in one line — the historical path's cheapest output is an
        empty field, and reproducing the input is a whole learning problem.
        """
        model = _build_model(rollout_space="physical")
        _zero_decoder_output(model)
        inputs = _inputs(model)
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
        assert torch.count_nonzero(prediction) == 0
        assert not torch.equal(prediction[:, 0], inputs[TARGET_GROUP][:, -1])

    def test_residual_output_is_bounded(self) -> None:
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
        )
        # a large positive bias would drive the sum far above 1 without the clamp
        bias = _final_conv(model).bias
        assert bias is not None
        with torch.no_grad():
            bias.fill_(5.0)
        model.eval()
        with torch.no_grad():
            prediction = model(_inputs(model))
        assert float(prediction.min()) >= 0.0
        assert float(prediction.max()) <= 1.0


def _small_tendency(model: EncodeProcessDecode, scale: float = 1e-3) -> None:
    """Give the zeroed tendency head a small non-zero weight.

    This is the regime a trained residual model lives in: a small signed correction on
    top of the previous state, well away from the [0, 1] clamp. At random init the
    tendency is large enough to saturate the clamp everywhere, which is precisely why
    the decoder's `zero_init_output` option exists.
    """
    generator = torch.Generator().manual_seed(99)
    final = _final_conv(model)
    with torch.no_grad():
        final.weight.copy_(torch.randn(final.weight.shape, generator=generator) * scale)
        if final.bias is not None:
            final.bias.zero_()


class TestPhysicalRolloutAdvancesTheState:
    def test_successive_leads_differ(self) -> None:
        """A non-zero tendency must produce a moving trajectory, not one field."""
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
            processor={
                "_target_": "icenet_mp.models.processors.VitProcessor",
                "patch_size": 4,
                "emb_dim": 32,
                "depth": 1,
                "heads": 2,
                "mlp_dim": 32,
                "dropout": 0.0,
            },
        )
        _small_tendency(model)
        model.eval()
        with torch.no_grad():
            prediction = model(_inputs(model))
        assert float(prediction.max()) < 1.0, "clamp saturated; not a valid test regime"
        for lead in range(1, model.n_forecast_steps):
            assert not torch.equal(prediction[:, lead], prediction[:, lead - 1])

    def test_zero_init_starts_at_persistence(self) -> None:
        """zero_init_output must place the whole trajectory exactly on persistence."""
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "zero_init_output": True,
                "skip_connection": {"method": "additive"},
            },
        )
        inputs = _inputs(model)
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
        persistence = inputs[TARGET_GROUP][:, -1]
        for lead in range(model.n_forecast_steps):
            assert torch.equal(prediction[:, lead], persistence)

    def test_later_leads_depend_on_the_previous_prediction(self) -> None:
        """Lead k+1 must be a function of the lead-k prediction, not of stale history.

        Two windows differing only in their OLDEST frame must give different final
        fields: the difference can only reach lead n by travelling through the
        intermediate predictions. Needs a processor that actually reads the whole
        window (NullProcessor does not), hence the ViT.
        """
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
            processor={
                "_target_": "icenet_mp.models.processors.VitProcessor",
                "patch_size": 4,
                "emb_dim": 32,
                "depth": 1,
                "heads": 2,
                "mlp_dim": 32,
                "dropout": 0.0,
            },
        )
        _small_tendency(model)
        model.eval()
        first = _inputs(model, seed=11)
        second = {key: value.clone() for key, value in first.items()}
        second[TARGET_GROUP][:, 0] = 0.0  # change only the OLDEST history frame
        with torch.no_grad():
            out_first = model(first)
            out_second = model(second)
        assert not torch.equal(out_first[:, -1], out_second[:, -1])

    def test_non_target_groups_are_held_at_last_observation(self) -> None:
        """An extra input group must not be hallucinated forward."""
        extra = DictConfig({"channels": 2, "name": "era5", "shape": (32, 32)})
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
            extra_inputs=[extra],
        )
        model.eval()
        inputs = _inputs(model)
        baseline = {key: value.clone() for key, value in inputs.items()}
        # Changing an OLDER era5 frame changes the encoding of the window, so the
        # forecast may move; changing nothing must be reproducible.
        with torch.no_grad():
            assert torch.equal(model(inputs), model(baseline))

    def test_feedback_writes_only_the_target_variable(self) -> None:
        """A multi-channel target group: the prediction is fed back into channel 1 only."""
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
            input_channels=3,
            target_variable_indices=[1],
        )
        _zero_decoder_output(model)
        inputs = _inputs(model)
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
        # zero tendency anchored on channel 1 => persistence of channel 1 at every lead
        expected = inputs[TARGET_GROUP][:, -1, 1:2]
        for lead in range(model.n_forecast_steps):
            assert torch.equal(prediction[:, lead], expected)

    def test_feedback_supports_non_contiguous_target_variables(self) -> None:
        """Target variables need not be neighbours: channels 0 and 2 of a 3-channel group."""
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
            input_channels=3,
            output_channels=2,
            target_variable_indices=[0, 2],
        )
        _zero_decoder_output(model)
        inputs = _inputs(model)
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
        # zero tendency anchored on channels [0, 2] => persistence of those channels
        # at every lead, exactly as in the contiguous case above.
        expected = inputs[TARGET_GROUP][:, -1, [0, 2]]
        for lead in range(model.n_forecast_steps):
            assert torch.equal(prediction[:, lead], expected)


class TestNoFutureLeak:
    def test_target_key_is_never_read(self) -> None:
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
        )
        model.eval()
        inputs = _inputs(model)
        with_truth = dict(inputs)
        with_truth["target"] = torch.ones(2, model.n_forecast_steps, 1, 32, 32)
        with torch.no_grad():
            assert torch.equal(model(inputs), model(with_truth))


class TestConfigValidation:
    def test_bad_rollout_space(self) -> None:
        with pytest.raises(ValueError, match="not a valid RolloutSpace"):
            _build_model(rollout_space="nonsense")

    def test_residual_accepts_a_bounded_decoder(self) -> None:
        """A bounded decoder works for a residual (additive-skip) physical model.

        With an additive skip connection BaseDecoder bounds SIGNED values
        symmetrically, so the tendency is never squashed into [0, 1].
        """
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "clamp",
                "skip_connection": {"method": "additive"},
            },
        )
        model.eval()
        with torch.no_grad():
            prediction = model(_inputs(model))
        assert float(prediction.min()) >= 0.0
        assert float(prediction.max()) <= 1.0


class TestAnchorSemantics:
    """Static and moving anchors are one mechanism differing only in the anchor frame.

    The static anchor (#405) and the moving anchor (#410) are the same additive skip
    connection; only the frame supplied as the anchor differs.

    Both tests below drive the decoder to emit a CONSTANT tendency c, which makes the
    two anchor choices analytically separable:
        static anchor  ->  output_k = clamp(observation + c)      (same for every lead)
        moving anchor  ->  output_k = clamp(output_{k-1} + c)     (accumulates with k)
    """

    @staticmethod
    def _constant_tendency(model: EncodeProcessDecode, value: float) -> None:
        """Make the decoder emit `value` everywhere, whatever its input."""
        final = _final_conv(model)
        with torch.no_grad():
            final.weight.zero_()
            assert final.bias is not None, "decoder's final conv needs a bias"
            final.bias.fill_(value)

    def test_moving_anchor_accumulates_across_leads(self) -> None:
        """rollout_space='physical': each lead corrects the PREVIOUS lead."""
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
        )
        step = 0.05
        self._constant_tendency(model, step)
        model.eval()
        inputs = _inputs(model, seed=5)
        with torch.no_grad():
            prediction = model(inputs)

        newest = inputs[TARGET_GROUP][:, -1]
        for lead in range(model.n_forecast_steps):
            expected = (newest + step * (lead + 1)).clamp(0.0, 1.0)
            torch.testing.assert_close(prediction[:, lead], expected)

    def test_static_anchor_does_not_accumulate(self) -> None:
        """The default latent path anchors EVERY lead on the newest observation."""
        model = _build_model(
            rollout_space="latent",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
        )
        step = 0.05
        self._constant_tendency(model, step)
        model.eval()
        inputs = _inputs(model, seed=5)
        with torch.no_grad():
            prediction = model(inputs)

        expected = (inputs[TARGET_GROUP][:, -1] + step).clamp(0.0, 1.0)
        for lead in range(model.n_forecast_steps):
            torch.testing.assert_close(prediction[:, lead], expected)


class TestTrainEvalParity:
    """training_step must score the SAME architecture that validation scores.

    Main's #422 overrode ``training_step`` in ``EncodeProcessDecode`` with an
    inlined LATENT path that never calls ``forward()``, while
    ``BaseModel.validation_step``/``test_step`` call ``self(batch)``. Without the
    ``rollout_space`` branch in ``training_step``, a physical-rollout model would
    TRAIN on the latent rollout and be VALIDATED on the physical one - two
    different architectures, with no error raised. These tests pin the routing.
    """

    @staticmethod
    def _batch(model: EncodeProcessDecode) -> dict[str, torch.Tensor]:
        batch = _inputs(model, seed=11)
        generator = torch.Generator().manual_seed(13)
        batch["target"] = torch.rand(
            2,
            model.n_forecast_steps,
            model.output_space.channels,
            *model.output_space.shape,
            generator=generator,
        )
        return batch

    def test_training_step_prediction_matches_forward_physical(self) -> None:
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
        )
        _small_tendency(model)
        model.eval()  # kill dropout so the two passes are deterministic
        batch = self._batch(model)
        with torch.no_grad():
            out = model.training_step(dict(batch), 0)
            expected = model(dict(batch))
        torch.testing.assert_close(out.prediction, expected)

    def test_training_step_loss_is_computed_on_the_physical_prediction(self) -> None:
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
        )
        _small_tendency(model)
        model.eval()
        batch = self._batch(model)
        with torch.no_grad():
            out = model.training_step(dict(batch), 0)
            expected_loss = model.loss(model(dict(batch)), batch["target"])
        torch.testing.assert_close(out.loss, expected_loss)

    def test_latent_path_is_unchanged_by_the_routing(self) -> None:
        """The default (latent) model still takes main's training_step path."""
        model = _build_model(rollout_space="latent")
        model.eval()
        batch = self._batch(model)
        with torch.no_grad():
            out = model.training_step(dict(batch), 0)
            expected = model(dict(batch))
        torch.testing.assert_close(out.prediction, expected)


class TestDecoderZeroInitOutput:
    """`zero_init_output` lives on the decoder (review: PR #410, C3)."""

    def test_zero_init_output_makes_residual_exactly_persistence(self) -> None:
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "zero_init_output": True,
                "skip_connection": {"method": "additive"},
            },
        )
        inputs = _inputs(model)
        persistence = inputs[TARGET_GROUP][:, -1]
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
        for lead in range(model.n_forecast_steps):
            assert torch.equal(prediction[:, lead], persistence)

    def test_default_is_off(self) -> None:
        model = _build_model(
            rollout_space="physical",
            decoder_extra={
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            },
        )
        final = _final_conv(model)
        assert final.weight.abs().max() > 0
