import torch
from omegaconf import DictConfig

from icenet_mp.models import EncodeProcessDecode
from icenet_mp.models.common import LatentFusion


def test_encode_process_decode_uses_attention_fusion() -> None:
    """Use attention fusion in the end-to-end model path."""
    input_spaces = [
        DictConfig({"channels": 2, "name": "input-a", "shape": (16, 16)}),
        DictConfig({"channels": 3, "name": "target", "shape": (16, 16)}),
    ]
    output_space = DictConfig({"channels": 1, "name": "target", "shape": (16, 16)})
    encoders = DictConfig(
        {
            "latent_space": (16, 16),
            "input-a": {
                "_target_": "icenet_mp.models.encoders.NaiveLinearEncoder",
            },
            "target": {
                "_target_": "icenet_mp.models.encoders.NaiveLinearEncoder",
            },
        }
    )
    fusion = DictConfig(
        {
            "_target_": "icenet_mp.models.common.LatentFusion",
            "mode": "attention",
            "temperature": 1.0,
        }
    )
    model = EncodeProcessDecode(
        name="attention-fusion-test",
        encoders=encoders,
        fusion=fusion,
        processor=DictConfig({"_target_": "icenet_mp.models.processors.NullProcessor"}),
        decoder=DictConfig(
            {"_target_": "icenet_mp.models.decoders.NaiveLinearDecoder"}
        ),
        hemisphere="north",
        input_spaces=input_spaces,
        loss=DictConfig({"_target_": "torch.nn.HuberLoss"}),
        lr_scheduler=DictConfig({}),
        metrics=[],
        n_forecast_steps=2,
        n_history_steps=3,
        output_space=output_space,
        optimizer=DictConfig({}),
        scheduler=DictConfig({}),
        target_variable_indices=[0],
    )
    inputs = {
        "input-a": torch.randn(2, 3, 2, 16, 16),
        "target": torch.randn(2, 3, 3, 16, 16),
    }

    latent = model._encode_inputs(inputs)
    result = model(inputs)

    assert isinstance(model.fusion, LatentFusion)
    assert model.fusion.mode == "attention"
    assert model.processor.data_space.channels == 5
    assert latent.shape == (2, 3, 5, 16, 16)
    assert result.shape == (2, 2, 1, 16, 16)
