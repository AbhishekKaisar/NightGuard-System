import argparse
import yaml
import torch
import os
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms

from models.ensemble import LowLightEnsemble, load_base_weights
from models.base_models import ZeroDCE, KinD, RetinexNet, Restormer
from utils.helpers import _safe_torch_load, _resolve_path, _resolve_weight_paths


def enhance_image(input_path, output_path, ensemble_model, device, tile_size=512, tile_overlap=32):
    # Load and format the image
    img = Image.open(input_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Enhance
    print(f"Processing {input_path}...")
    with torch.no_grad():
        b, c, h, w = img_tensor.shape
        if h <= tile_size and w <= tile_size:
            enhanced_tensor = ensemble_model(img_tensor)
        else:
            stride = tile_size - tile_overlap
            h_idx_list = list(range(0, h - tile_size, stride)) + [max(0, h - tile_size)]
            w_idx_list = list(range(0, w - tile_size, stride)) + [max(0, w - tile_size)]
            
            # Remove duplicates while preserving order
            h_idx_list = list(dict.fromkeys(h_idx_list))
            w_idx_list = list(dict.fromkeys(w_idx_list))
            
            out_tensor = torch.zeros_like(img_tensor)
            weight_tensor = torch.zeros_like(img_tensor)
            
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_tensor[..., h_idx:h_idx+tile_size, w_idx:w_idx+tile_size]
                    out_patch = ensemble_model(in_patch)
                    
                    patch_h, patch_w = out_patch.shape[-2:]
                    
                    y_ramp = torch.min(torch.arange(patch_h, device=device), torch.arange(patch_h - 1, -1, -1, device=device))
                    x_ramp = torch.min(torch.arange(patch_w, device=device), torch.arange(patch_w - 1, -1, -1, device=device))
                    weight_y = torch.clamp(y_ramp.float() + 1, max=tile_overlap) / tile_overlap
                    weight_x = torch.clamp(x_ramp.float() + 1, max=tile_overlap) / tile_overlap
                    weight = weight_y.unsqueeze(1) * weight_x.unsqueeze(0)
                    weight = weight.unsqueeze(0).unsqueeze(0)
                    
                    out_tensor[..., h_idx:h_idx+patch_h, w_idx:w_idx+patch_w] += out_patch * weight
                    weight_tensor[..., h_idx:h_idx+patch_h, w_idx:w_idx+patch_w] += weight
            
            enhanced_tensor = out_tensor / weight_tensor

    # Save
    save_image(enhanced_tensor, output_path)
    print(f"Saved enhanced image to {output_path}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(project_root, "configs", "config.yaml")

    parser = argparse.ArgumentParser(description="Enhance a single dark image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the low-light image.")
    parser.add_argument("--output", type=str, default="enhanced_result.png", help="Path to save the result.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config file.")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size for patch-based inference.")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Overlap between tiles.")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config_root = os.path.dirname(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _resolve_weight_paths(cfg, project_root)

    device = torch.device(cfg['hardware']['device'] if torch.cuda.is_available() else "cpu")

    dce = ZeroDCE()
    kind = KinD()
    retinex = RetinexNet()
    restormer = Restormer(LayerNorm_type='BiasFree')
    # Initialize and load models
    ensemble = LowLightEnsemble(dce, kind, retinex, restormer).to(device)
    load_base_weights(ensemble, cfg, device)

    weight_path = os.path.join(
        _resolve_path(project_root, cfg['weights']['save_dir']),
        cfg['weights']['save_name']
    )
    if os.path.exists(weight_path):
        ensemble.fusion_unet.load_state_dict(_safe_torch_load(weight_path, map_location=device))
    else:
        raise FileNotFoundError(
            f"Fusion checkpoint not found at '{weight_path}'. Train first or update configs/config.yaml."
        )
    ensemble.eval()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    enhance_image(args.input, args.output, ensemble, device, args.tile_size, args.tile_overlap)
