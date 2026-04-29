import yaml
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.ensemble import LowLightEnsemble, load_base_weights
from models.base_models import ZeroDCE, KinD, RetinexNet, Restormer
from utils.dataloader import PairedLowLightDataset, SIDDataset
from utils.metrics import batch_psnr, batch_ssim
from utils.helpers import _safe_torch_load, _resolve_path, _resolve_weight_paths

def _build_test_sets(cfg, project_root):
    test_sets = []

    # LOL paired test set
    lol_low = _resolve_path(project_root, cfg['data'].get('test_low_light'))
    lol_gt = _resolve_path(project_root, cfg['data'].get('test_ground_truth'))
    if lol_low and lol_gt and os.path.isdir(lol_low) and os.path.isdir(lol_gt):
        test_sets.append(("lol", PairedLowLightDataset(lol_low, lol_gt, is_training=False)))

    # SID test set via split list
    sid_low = _resolve_path(project_root, cfg['data'].get('sid_test_low_light'))
    sid_gt = _resolve_path(project_root, cfg['data'].get('sid_test_ground_truth'))
    sid_list = _resolve_path(project_root, cfg['data'].get('sid_test_list'))
    if sid_low and sid_gt and sid_list:
        if os.path.isdir(sid_low) and os.path.isdir(sid_gt) and os.path.isfile(sid_list):
            test_sets.append(
                (
                    "sid",
                    SIDDataset(
                        sid_low,
                        sid_gt,
                        is_training=False,
                        split_file=sid_list,
                        raw_half_size=True,
                    ),
                )
            )

    if not test_sets:
        raise FileNotFoundError(
            "No test dataset found. Check LOL/SID test paths in configs/config.yaml."
        )
    return test_sets


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _resolve_weight_paths(cfg, project_root)

    device = torch.device(cfg['hardware']['device'] if torch.cuda.is_available() else "cpu")

    dce = ZeroDCE()
    kind = KinD()
    retinex = RetinexNet()
    restormer = Restormer(LayerNorm_type='BiasFree')

    ensemble = LowLightEnsemble(dce, kind, retinex, restormer).to(device)
    load_base_weights(ensemble, cfg, device)

    save_path = os.path.join(
        _resolve_path(project_root, cfg['weights']['save_dir']),
        cfg['weights']['save_name']
    )
    if os.path.exists(save_path):
        ensemble.fusion_unet.load_state_dict(_safe_torch_load(save_path, map_location=device))
        print("Loaded trained fusion weights successfully.")
    else:
        print("Warning: No trained weights found. Evaluating with random U-Net weights.")

    ensemble.eval()

    test_sets = _build_test_sets(cfg, project_root)
    output_root = os.path.join(project_root, "results", "test_outputs_patch")
    os.makedirs(output_root, exist_ok=True)

    overall_total_psnr = 0.0
    overall_total_ssim = 0.0
    overall_count = 0

    tile_size = 512
    tile_overlap = 256

    with torch.no_grad():
        for split_name, test_dataset in test_sets:
            num_workers = cfg.get('training', {}).get('num_workers', 4)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=1, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False
            )
            split_total_psnr = 0.0
            split_total_ssim = 0.0
            split_output_dir = os.path.join(output_root, split_name)
            os.makedirs(split_output_dir, exist_ok=True)

            print(f"Evaluating {split_name.upper()} split with {len(test_dataset)} images using Patch Inference...")
            for i, (low_img, gt_img) in enumerate(test_loader):
                low_img, gt_img = low_img.to(device), gt_img.to(device)

                b, c, h, w = low_img.shape
                if h > 1000 or w > 1000:
                    stride = tile_size - tile_overlap
                    h_idx_list = list(range(0, h - tile_size, stride)) + [max(0, h - tile_size)]
                    w_idx_list = list(range(0, w - tile_size, stride)) + [max(0, w - tile_size)]
                    
                    # Remove duplicates while preserving order
                    h_idx_list = list(dict.fromkeys(h_idx_list))
                    w_idx_list = list(dict.fromkeys(w_idx_list))
                    
                    out_tensor = torch.zeros_like(low_img)
                    weight_tensor = torch.zeros_like(low_img)
                    
                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = low_img[..., h_idx:h_idx+tile_size, w_idx:w_idx+tile_size]
                            out_patch = ensemble(in_patch)
                            
                            patch_h, patch_w = out_patch.shape[-2:]
                            
                            y_ramp = torch.min(torch.arange(patch_h, device=device), torch.arange(patch_h - 1, -1, -1, device=device))
                            x_ramp = torch.min(torch.arange(patch_w, device=device), torch.arange(patch_w - 1, -1, -1, device=device))
                            weight_y = torch.clamp(y_ramp.float() + 1, max=tile_overlap) / tile_overlap
                            weight_x = torch.clamp(x_ramp.float() + 1, max=tile_overlap) / tile_overlap
                            weight = weight_y.unsqueeze(1) * weight_x.unsqueeze(0)
                            weight = weight.unsqueeze(0).unsqueeze(0)
                            
                            out_tensor[..., h_idx:h_idx+patch_h, w_idx:w_idx+patch_w] += out_patch * weight
                            weight_tensor[..., h_idx:h_idx+patch_h, w_idx:w_idx+patch_w] += weight
                            
                            # Memory Management: Free VRAM after each patch
                            torch.cuda.empty_cache()
                    
                    enhanced_img = out_tensor / weight_tensor
                else:
                    enhanced_img = ensemble(low_img)
                    torch.cuda.empty_cache()

                split_total_psnr += batch_psnr(enhanced_img, gt_img)
                split_total_ssim += batch_ssim(enhanced_img, gt_img)

                save_image(enhanced_img, os.path.join(split_output_dir, f"output_{i}.png"))
                
                torch.cuda.empty_cache()

            split_count = len(test_loader)
            split_avg_psnr = split_total_psnr / split_count
            split_avg_ssim = split_total_ssim / split_count
            print(f"  {split_name.upper()} PSNR: {split_avg_psnr:.2f} dB")
            print(f"  {split_name.upper()} SSIM: {split_avg_ssim:.4f}")

            overall_total_psnr += split_total_psnr
            overall_total_ssim += split_total_ssim
            overall_count += split_count

    final_psnr = overall_total_psnr / overall_count
    final_ssim = overall_total_ssim / overall_count
    print("=== Final Evaluation (All Enabled Test Splits) ===")
    print(f"Average PSNR: {final_psnr:.2f} dB")
    print(f"Average SSIM: {final_ssim:.4f}")

if __name__ == "__main__":
    main()
