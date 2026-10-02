import logging

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from icenet_mp.losses import LeadTimeWeightedLoss
from icenet_mp.models import DDPM


class TestDDPM:
    @staticmethod
    def _make_model(cfg_loss: DictConfig, *, use_autoregressive: bool = True) -> DDPM:
        return DDPM(
            name="ddpm",
            hemisphere="north",
            input_spaces=[
                {"channels": 1, "name": "osisaf-north", "shape": (16, 16)},
                {"channels": 2, "name": "era5", "shape": (8, 8)},
            ],
            loss=cfg_loss,
            metrics=[],
            n_forecast_steps=2,
            n_history_steps=2,
            output_space={
                "channels": 1,
                "name": "osisaf-north",
                "shape": (16, 16),
            },
            optimizer={},
            scheduler={},
            lr_scheduler={},
            start_out_channels=4,
            timesteps=4,
            use_autoregressive=use_autoregressive,
        )

    @staticmethod
    def _make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
        return {
            "osisaf-north": torch.randn(batch_size, 2, 1, 16, 16),
            "era5": torch.randn(batch_size, 2, 2, 8, 8),
            "target": torch.randn(batch_size, 2, 1, 16, 16),
        }

    def test_forward_is_disabled(self, cfg_loss: DictConfig) -> None:
        model = self._make_model(cfg_loss)

        with pytest.raises(NotImplementedError, match="training_step"):
            model(self._make_batch())

    def test_prepare_inputs_combines_conditioning_channels(
        self, cfg_loss: DictConfig
    ) -> None:
        model = self._make_model(cfg_loss)

        conditioning = model.prepare_inputs(self._make_batch())

        assert conditioning.shape == (2, model.cond_channels, 16, 16)

    @pytest.mark.parametrize("use_autoregressive", [True, False])
    def test_training_step_shapes(
        self,
        cfg_loss: DictConfig,
        monkeypatch: pytest.MonkeyPatch,
        *,
        use_autoregressive: bool,
    ) -> None:
        model = self._make_model(cfg_loss, use_autoregressive=use_autoregressive)
        monkeypatch.setattr(
            model.model,
            "forward",
            lambda noisy, _timesteps, _conditioning: torch.zeros_like(noisy),
        )

        result = model.training_step(self._make_batch(), 0)

        expected_steps = 1 if use_autoregressive else model.n_forecast_steps
        assert result.prediction.shape == (2, expected_steps, 1, 16, 16)
        assert result.target.shape == result.prediction.shape
        assert result.loss.ndim == 0

    @pytest.mark.parametrize("use_autoregressive", [True, False])
    def test_training_supports_lead_time_weighted_loss(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        use_autoregressive: bool,
    ) -> None:
        """DDPM restores forecast time before a lead-time weighted loss call."""
        loss_cfg = OmegaConf.create(
            {"_target_": "torch.nn.MSELoss", "lead_time_exponent": 1.0}
        )
        model = self._make_model(loss_cfg, use_autoregressive=use_autoregressive)
        assert isinstance(model.loss_fn, LeadTimeWeightedLoss)
        monkeypatch.setattr(
            model.model,
            "forward",
            lambda noisy, _timesteps, _conditioning: torch.zeros_like(noisy),
        )

        result = model.training_step(self._make_batch(), 0)

        assert result.loss.ndim == 0
        assert torch.isfinite(result.loss)

    @pytest.mark.parametrize(
        ("test_use_autoregressive", "test_lead_time_exponent", "expect_warning"),
        [(True, 2.0, True), (False, 2.0, False), (True, None, False)],
        ids=["autoregressive-weighted", "direct-weighted", "autoregressive-unweighted"],
    )
    def test_warns_when_lead_time_weighting_ignored(
        self,
        caplog: pytest.LogCaptureFixture,
        *,
        test_use_autoregressive: bool,
        test_lead_time_exponent: float | None,
        expect_warning: bool,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="icenet_mp.models.ddpm"):
            self._make_model(
                OmegaConf.create(
                    {
                        "_target_": "torch.nn.MSELoss",
                        "lead_time_exponent": test_lead_time_exponent,
                    }
                ),
                use_autoregressive=test_use_autoregressive,
            )
        warned = any("has no effect" in r.getMessage() for r in caplog.records)
        assert warned is expect_warning

    def test_parallel_sample_runs_reverse_diffusion_loop(
        self,
        cfg_loss: DictConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = self._make_model(cfg_loss, use_autoregressive=False)
        timesteps: list[torch.Tensor] = []
        monkeypatch.setattr(
            model.model,
            "forward",
            lambda noisy, _timesteps, _conditioning: torch.zeros_like(noisy),
        )

        def p_sample(
            x: torch.Tensor, t: torch.Tensor, _pred_v: torch.Tensor
        ) -> torch.Tensor:
            timesteps.append(t.clone())
            return torch.zeros_like(x)

        monkeypatch.setattr(model.diffusion, "p_sample", p_sample)

        prediction = model.sample(self._make_batch())

        assert prediction.shape == (2, 2, 16, 16)
        assert torch.count_nonzero(prediction) == 0
        assert [int(t[0]) for t in timesteps] == [3, 2, 1, 0]

    def test_autoregressive_sample_slides_conditioning_windows(
        self,
        cfg_loss: DictConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = self._make_model(cfg_loss, use_autoregressive=True)
        batch = self._make_batch(batch_size=1)
        batch["osisaf-north"] = torch.stack(
            [
                torch.full((1, 1, 16, 16), 1.0),
                torch.full((1, 1, 16, 16), 2.0),
            ],
            dim=1,
        )
        batch["era5"] = torch.stack(
            [
                torch.full((1, 2, 8, 8), 10.0),
                torch.full((1, 2, 8, 8), 20.0),
            ],
            dim=1,
        )
        batch["era5_forecast"] = torch.stack(
            [
                torch.full((1, 2, 8, 8), 30.0),
                torch.full((1, 2, 8, 8), 40.0),
            ],
            dim=1,
        )
        conditioning_windows: list[dict[str, torch.Tensor]] = []
        reverse_steps: list[torch.Tensor] = []

        def prepare_inputs(current_batch: dict[str, torch.Tensor]) -> torch.Tensor:
            conditioning_windows.append(
                {key: value.clone() for key, value in current_batch.items()}
            )
            return torch.zeros((1, model.cond_channels, 16, 16))

        def p_sample(
            x: torch.Tensor, t: torch.Tensor, _pred_v: torch.Tensor
        ) -> torch.Tensor:
            reverse_steps.append(t.clone())
            return torch.zeros_like(x)

        monkeypatch.setattr(model, "prepare_inputs", prepare_inputs)
        monkeypatch.setattr(
            model.model,
            "forward",
            lambda noisy, _timesteps, _conditioning: torch.zeros_like(noisy),
        )
        monkeypatch.setattr(model.diffusion, "p_sample", p_sample)

        prediction = model.sample(batch)

        assert prediction.shape == (1, 2, 16, 16)
        assert torch.count_nonzero(prediction) == 0
        assert len(reverse_steps) == model.timesteps * model.n_forecast_steps
        assert len(conditioning_windows) == model.n_forecast_steps
        assert torch.all(conditioning_windows[0]["osisaf-north"][:, 0] == 1)
        assert torch.all(conditioning_windows[0]["osisaf-north"][:, 1] == 2)
        assert torch.all(conditioning_windows[1]["osisaf-north"][:, 0] == 2)
        assert torch.count_nonzero(conditioning_windows[1]["osisaf-north"][:, 1]) == 0
        assert torch.all(conditioning_windows[0]["era5"][:, 0] == 10)
        assert torch.all(conditioning_windows[0]["era5"][:, 1] == 20)
        assert torch.all(conditioning_windows[1]["era5"][:, 0] == 20)
        assert torch.all(conditioning_windows[1]["era5"][:, 1] == 30)

    @pytest.mark.parametrize("step_name", ["validation_step", "test_step"])
    def test_evaluation_step_shapes(
        self,
        cfg_loss: DictConfig,
        monkeypatch: pytest.MonkeyPatch,
        *,
        step_name: str,
    ) -> None:
        model = self._make_model(cfg_loss)
        batch = self._make_batch()
        monkeypatch.setattr(
            model,
            "sample",
            lambda _batch: torch.zeros((2, model.n_forecast_steps, 16, 16)),
        )

        result = getattr(model, step_name)(batch, 0)

        assert result.prediction.shape == (2, model.n_forecast_steps, 1, 16, 16)
        assert result.target.shape == result.prediction.shape
        assert result.loss.ndim == 0
