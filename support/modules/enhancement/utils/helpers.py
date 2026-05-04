import os
import torch

def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _resolve_path(project_root, value):
    if value is None:
        return None
    if os.path.isabs(value):
        return value
    return os.path.join(project_root, value)

def _resolve_weight_paths(cfg, project_root):
    weight_keys = [
        "zero_dce",
        "kind_decom",
        "kind_restore",
        "kind_illum",
        "retinex_decom",
        "retinex_relight",
        "restormer",
    ]
    for key in weight_keys:
        cfg["weights"][key] = _resolve_path(project_root, cfg["weights"][key])
