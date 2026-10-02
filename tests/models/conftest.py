import pytest
from omegaconf import DictConfig


@pytest.fixture
def cfg_decoder() -> DictConfig:
    """Test configuration for a decoder."""
    return DictConfig({"_target_": "icenet_mp.models.decoders.NaiveLinearDecoder"})


@pytest.fixture
def cfg_encoders() -> DictConfig:
    """Test configuration for an encoder."""
    return DictConfig(
        {
            "latent_space": (64, 64),
            "test-input": {
                "_target_": "icenet_mp.models.encoders.NaiveLinearEncoder",
            },
            "target": {
                "_target_": "icenet_mp.models.encoders.NaiveLinearEncoder",
            },
        }
    )


@pytest.fixture
def cfg_processor() -> DictConfig:
    """Test configuration for a processor."""
    return DictConfig({"_target_": "icenet_mp.models.processors.NullProcessor"})
