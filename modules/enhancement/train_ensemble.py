import os
from multiprocessing import freeze_support

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from models.ensemble import LowLightEnsemble, load_base_weights
from models.base_models import ZeroDCE, KinD, RetinexNet, Restormer
from utils.dataloader import PairedLowLightDataset, SIDDataset
from utils.losses import FusionLoss
from utils.image_utils import save_sample_images
from utils.metrics import batch_psnr
from utils.seed import set_global_seed, seed_worker
from utils.helpers import _resolve_path, _resolve_weight_paths

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "configs", "config.yaml")

    # 1. Load Configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _resolve_weight_paths(cfg, project_root)

    device = torch.device(cfg['hardware']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = cfg.get("training", {}).get("seed", 42)
    set_global_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    # 2. Initialize and Load Base Models
    print("Loading Base Models...")
    dce = ZeroDCE()
    kind = KinD()
    retinex = RetinexNet()
    restormer = Restormer(LayerNorm_type='BiasFree')

    ensemble = LowLightEnsemble(dce, kind, retinex, restormer).to(device)
    load_base_weights(ensemble, cfg, device)

    # 4. Setup Data
    print("Loading LOL Dataset...")
    lol_dataset = PairedLowLightDataset(
        _resolve_path(project_root, cfg['data']['lol_train_low_light']),
        _resolve_path(project_root, cfg['data']['lol_train_ground_truth']),
        patch_size=cfg['training']['patch_size'],
        is_training=True
    )

    print("Loading SID Dataset...")
    sid_dataset = SIDDataset(
        _resolve_path(project_root, cfg['data']['sid_train_low_light']),
        _resolve_path(project_root, cfg['data']['sid_train_ground_truth']),
        patch_size=cfg['training']['patch_size'],
        is_training=True,
        split_file=_resolve_path(project_root, cfg['data'].get('sid_train_list'))
    )

    print("Combining Datasets...")
    train_dataset = ConcatDataset([lol_dataset, sid_dataset])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
        persistent_workers=True if cfg['training'].get('num_workers', 0) > 0 else False
    )

    print("Loading Validation Dataset...")

    val_datasets = []
    val_lol_low = _resolve_path(project_root, cfg['data'].get('val_low_light'))
    val_lol_gt = _resolve_path(project_root, cfg['data'].get('val_ground_truth'))
    val_lol_loader = None
    if val_lol_low and val_lol_gt and os.path.isdir(val_lol_low) and os.path.isdir(val_lol_gt):
        val_lol_dataset = PairedLowLightDataset(val_lol_low, val_lol_gt, is_training=False)
        val_lol_loader = DataLoader(
            val_lol_dataset, 
            batch_size=1, 
            shuffle=False,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True,
            persistent_workers=True if cfg['training'].get('num_workers', 0) > 0 else False
        )

    val_sid_low = _resolve_path(project_root, cfg['data'].get('sid_val_low_light'))
    val_sid_gt = _resolve_path(project_root, cfg['data'].get('sid_val_ground_truth'))
    val_sid_list = _resolve_path(project_root, cfg['data'].get('sid_val_list'))
    val_sid_loader = None
    if val_sid_low and val_sid_gt and val_sid_list:
        if os.path.isdir(val_sid_low) and os.path.isdir(val_sid_gt) and os.path.isfile(val_sid_list):
            val_sid_dataset = SIDDataset(
                val_sid_low,
                val_sid_gt,
                is_training=False,
                split_file=val_sid_list,
                eval_resize=cfg['training']['patch_size'],
            )
            val_sid_loader = DataLoader(
                val_sid_dataset, 
                batch_size=1, 
                shuffle=False,
                num_workers=cfg['training']['num_workers'],
                pin_memory=True,
                persistent_workers=True if cfg['training'].get('num_workers', 0) > 0 else False
            )

    if val_lol_loader is None and val_sid_loader is None:
        raise FileNotFoundError("No validation datasets found. Check val paths in configs/config.yaml")

    optimizer = optim.Adam(ensemble.fusion_unet.parameters(), lr=cfg['training']['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
    criterion = FusionLoss(
        l1_weight=cfg['loss']['l1_weight'],
        perceptual_weight=cfg['loss']['perceptual_weight']
    ).to(device)

    num_epochs = cfg['training']['epochs']
    save_dir = _resolve_path(project_root, cfg['weights']['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, cfg['weights']['save_name'])
    best_psnr = 0.0

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    print("Starting U-Net Fusion Training...")
    for epoch in range(num_epochs):
        ensemble.eval()
        ensemble.fusion_unet.train()
        epoch_loss = 0
        epoch_psnr = 0
        sample_batch = None

        for _, (low_img, gt_img) in enumerate(train_loader):
            low_img, gt_img = low_img.to(device), gt_img.to(device)

            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                enhanced_img = ensemble(low_img)
                loss = criterion(enhanced_img, gt_img)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_psnr += batch_psnr(enhanced_img.detach(), gt_img.detach())
            if sample_batch is None:
                sample_batch = (low_img.detach(), enhanced_img.detach(), gt_img.detach())

        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)

        ensemble.eval()
        val_lol_psnr = None
        val_sid_psnr = None

        with torch.no_grad():
            if val_lol_loader is not None:
                lol_psnr_sum = 0.0
                for low_img, gt_img in val_lol_loader:
                    low_img, gt_img = low_img.to(device), gt_img.to(device)
                    enhanced_img = ensemble(low_img)
                    lol_psnr_sum += batch_psnr(enhanced_img, gt_img)
                val_lol_psnr = lol_psnr_sum / len(val_lol_loader)

            if val_sid_loader is not None:
                sid_psnr_sum = 0.0
                for low_img, gt_img in val_sid_loader:
                    low_img, gt_img = low_img.to(device), gt_img.to(device)
                    enhanced_img = ensemble(low_img)
                    sid_psnr_sum += batch_psnr(enhanced_img, gt_img)
                val_sid_psnr = sid_psnr_sum / len(val_sid_loader)

        # Use average of available metrics for best model tracking
        available_psnrs = []
        if val_lol_psnr is not None: available_psnrs.append(val_lol_psnr)
        if val_sid_psnr is not None: available_psnrs.append(val_sid_psnr)
        
        if not available_psnrs:
            raise RuntimeError("No validation metrics computed for this epoch.")
        avg_val_psnr = sum(available_psnrs) / len(available_psnrs)

        status_parts = [
            f"Epoch [{epoch + 1}/{num_epochs}]",
            f"Loss: {avg_loss:.4f}",
            f"Train PSNR: {avg_psnr:.2f} dB",
        ]
        if val_lol_psnr is not None:
            status_parts.append(f"LOL Val PSNR: {val_lol_psnr:.2f} dB")
        if val_sid_psnr is not None:
            status_parts.append(f"SID Val PSNR: {val_sid_psnr:.2f} dB")
        print(" | ".join(status_parts))

        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(ensemble.fusion_unet.state_dict(), save_path)
            print(f"  -> New best model saved (PSNR: {best_psnr:.2f} dB)")

        if (epoch + 1) % cfg['training']['save_frequency'] == 0 and sample_batch is not None:
            low_img, enhanced_img, gt_img = sample_batch
            save_sample_images(low_img, enhanced_img, gt_img, epoch + 1)

        scheduler.step()

    print(f"Training Complete. Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Best model saved to {save_path}")


if __name__ == "__main__":
    freeze_support()
    main()
