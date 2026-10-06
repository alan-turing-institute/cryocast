from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def cfg_decoder() -> DictConfig:
    """Test configuration for a decoder."""
    return DictConfig({"_target_": "cryocast.models.decoders.NaiveLinearDecoder"})


@pytest.fixture
def cfg_encoders() -> DictConfig:
    """Test configuration for an encoder."""
    return DictConfig(
        {
            "latent_space": (64, 64),
            "test-input": {
                "_target_": "cryocast.models.encoders.NaiveLinearEncoder",
            },
            "target": {
                "_target_": "cryocast.models.encoders.NaiveLinearEncoder",
            },
        }
    )


@pytest.fixture
def cfg_input_space() -> DictConfig:
    """Test configuration for an input space."""
    return DictConfig(
        {
            "channels": 4,
            "name": "test-input",
            "shape": (16, 16),
        }
    )


@pytest.fixture
def cfg_loss() -> DictConfig:
    """Test configuration for a loss function."""
    return OmegaConf.create({"_target_": "torch.nn.HuberLoss", "delta": 0.5})


@pytest.fixture
def cfg_lr_scheduler() -> DictConfig:
    """Test configuration for a scheduler's Lightning `lr_scheduler_config` wrapper."""
    return DictConfig({"frequency": 1, "interval": "epoch"})


@pytest.fixture
def cfg_metrics() -> list[dict[str, Any]]:
    """Test configuration for a model's `metrics` list."""
    return [
        {
            "name": "accuracy",
            "_target_": "cryocast.metrics.IceNetAccuracyPerForecastDay",
        },
        {"name": "mae", "_target_": "cryocast.metrics.MAEPerForecastDay"},
        {"name": "rmse", "_target_": "cryocast.metrics.RMSEPerForecastDay"},
        {
            "name": "sieerror",
            "_target_": "cryocast.metrics.SeaIceExtentErrorPerForecastDay",
        },
        {
            "name": "iiee",
            "_target_": "cryocast.metrics.IntegratedIceEdgeErrorPerForecastDay",
        },
        {
            "name": "diiee",
            "_target_": "cryocast.metrics.DistanceAveragedIceEdgeErrorPerForecastDay",
        },
        {
            "name": "centroid_error",
            "_target_": "cryocast.metrics.CentroidErrorPerForecastDay",
        },
        {
            "name": "fss_neighbourhood_size_1",
            "_target_": "cryocast.metrics.FractionalSkillScorePerForecastDay",
            "neighbourhood_size": 1,
        },
        {
            "name": "fss_neighbourhood_size_5",
            "_target_": "cryocast.metrics.FractionalSkillScorePerForecastDay",
            "neighbourhood_size": 5,
        },
        {
            "name": "fss_neighbourhood_size_15",
            "_target_": "cryocast.metrics.FractionalSkillScorePerForecastDay",
            "neighbourhood_size": 15,
        },
        {"name": "ssim", "_target_": "cryocast.metrics.SSIMPerForecastDay"},
    ]


@pytest.fixture
def cfg_optimizer() -> DictConfig:
    """Test configuration for an optimizer."""
    return DictConfig({"_target_": "torch.optim.AdamW", "lr": 5e-4})


@pytest.fixture
def cfg_output_space() -> DictConfig:
    """Test configuration for an output space."""
    return DictConfig(
        {
            "channels": 1,
            "name": "target",
            "shape": (16, 16),
        }
    )


@pytest.fixture
def cfg_processor() -> DictConfig:
    """Test configuration for a processor."""
    return DictConfig({"_target_": "cryocast.models.processors.NullProcessor"})


@pytest.fixture
def cfg_scheduler() -> DictConfig:
    """Test configuration for a scheduler."""
    return DictConfig(
        {
            "_target_": "torch.optim.lr_scheduler.LinearLR",
            "start_factor": 0.2,
            "end_factor": 0.8,
        }
    )
