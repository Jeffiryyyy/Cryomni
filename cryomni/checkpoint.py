from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .protein_swin_mae3d import SwinTransformer_MAE3D_New


DEFAULT_CONFIG = {
    "patch_size": [4, 4, 4],
    "embed_dim": 576,
    "depths": [10, 10, 4, 2],
    "num_heads": [18, 36, 72, 144],
    "window_size": [4, 4, 4],
    "resolution": 64,
    "masking_prob": 0.75,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "decoder_dropout": 0.0,
    "stochastic_depth_prob": 0.0,
    "expand_dim": True,
}

_MODEL_CONFIG_KEYS = set(DEFAULT_CONFIG) | {
    "out_channels",
    "input_ch_dim",
    "masking_strategy",
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        return dict(DEFAULT_CONFIG)

    with Path(config_path).open("r") as f:
        config = json.load(f)

    merged = dict(DEFAULT_CONFIG)
    merged.update({k: v for k, v in config.items() if k in _MODEL_CONFIG_KEYS})
    return merged


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, dict):
        checkpoint = checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if hasattr(value, "shape")
    }


def load_model(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> SwinTransformer_MAE3D_New:
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        model = SwinTransformer_MAE3D_New.from_pretrained(str(checkpoint_path))
        return model

    model = SwinTransformer_MAE3D_New(**load_config(config_path))
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)

    return model
