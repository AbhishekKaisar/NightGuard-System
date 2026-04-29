import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .fusion_engine.unet import FusionUNet


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Backward compatibility for older torch versions.
        return torch.load(path, map_location=map_location)


class LowLightEnsemble(nn.Module):
    def __init__(self, dce_model, kind_model, retinex_model, restormer_model):
        super(LowLightEnsemble, self).__init__()

        self.dce = dce_model
        self.kind = kind_model
        self.retinex = retinex_model
        self.restormer = restormer_model

        # Initialize the U-Net expecting 12 channels (4 models * 3 RGB channels)
        self.fusion_unet = FusionUNet(n_channels=12, n_classes=3)

        # FREEZE THE BASE MODELS: We only want to train the U-Net to fuse them
        self._freeze_model(self.dce)
        self._freeze_model(self.kind)
        self._freeze_model(self.retinex)
        self._freeze_model(self.restormer)

    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def _pad_to_multiple(x, multiple=8):
        _, _, h, w = x.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return padded, h, w

    def forward(self, x):
        x_padded, original_h, original_w = self._pad_to_multiple(x, multiple=16)
        with torch.no_grad():
            # ZeroDCE returns (img1, final_img, r). We want the final_img at index 1.
            out_dce = self.dce(x_padded)[1]

            # KinD and RetinexNet return multiple maps. We grab the last one.
            out_kind = self.kind(x_padded)[-1]
            out_retinex = self.retinex(x_padded)[-1]

            # Restormer returns a single tensor directly
            out_restormer = self.restormer(x_padded)

        fused_features = torch.cat([out_dce, out_kind, out_retinex, out_restormer], dim=1)
        final_output = self.fusion_unet(fused_features)
        return final_output[:, :, :original_h, :original_w]


def load_base_weights(ensemble, cfg, device):
    """Load pretrained weights for all frozen base models in the ensemble."""
    def _must_exist(weight_path, key):
        if not weight_path:
            raise ValueError(f"Missing config value for weights.{key}")
        if not isinstance(weight_path, str) or not weight_path.strip():
            raise ValueError(f"Invalid config value for weights.{key}: {weight_path!r}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found for weights.{key}: {weight_path}")

    def _load_model_state(module, state, key):
        try:
            module.load_state_dict(state)
        except Exception as exc:
            raise RuntimeError(f"Failed loading weights.{key}: {exc}") from exc

    for key in [
        "zero_dce",
        "restormer",
        "kind_decom",
        "kind_restore",
        "kind_illum",
        "retinex_decom",
        "retinex_relight",
    ]:
        _must_exist(cfg['weights'][key], key)

    _load_model_state(
        ensemble.dce,
        _safe_torch_load(cfg['weights']['zero_dce'], map_location=device),
        "zero_dce",
    )

    ckpt = _safe_torch_load(cfg['weights']['restormer'], map_location=device)
    if 'params' in ckpt:
        _load_model_state(ensemble.restormer, ckpt['params'], "restormer")
    elif 'state_dict' in ckpt:
        _load_model_state(ensemble.restormer, ckpt['state_dict'], "restormer")
    else:
        _load_model_state(ensemble.restormer, ckpt, "restormer")

    _load_model_state(
        ensemble.kind.decom_net,
        _safe_torch_load(cfg['weights']['kind_decom'], map_location=device),
        "kind_decom",
    )
    _load_model_state(
        ensemble.kind.restore_net,
        _safe_torch_load(cfg['weights']['kind_restore'], map_location=device),
        "kind_restore",
    )
    _load_model_state(
        ensemble.kind.illum_net,
        _safe_torch_load(cfg['weights']['kind_illum'], map_location=device),
        "kind_illum",
    )

    _load_model_state(
        ensemble.retinex.DecomNet,
        _safe_torch_load(cfg['weights']['retinex_decom'], map_location=device),
        "retinex_decom",
    )
    _load_model_state(
        ensemble.retinex.RelightNet,
        _safe_torch_load(cfg['weights']['retinex_relight'], map_location=device),
        "retinex_relight",
    )

    print("All base model weights loaded successfully.")
