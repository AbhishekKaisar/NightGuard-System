import torch
import torch.nn.functional as F
import math


def calculate_psnr(enhanced_tensor, ground_truth_tensor):
    """
    Calculates the Peak Signal-to-Noise Ratio.
    Expects tensors in the range [0.0, 1.0].
    """
    mse = torch.mean((enhanced_tensor - ground_truth_tensor) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def batch_psnr(enhanced_batch, ground_truth_batch):
    """Calculates average PSNR for an entire batch."""
    mse = torch.mean((enhanced_batch - ground_truth_batch) ** 2, dim=[1, 2, 3])
    psnr = torch.zeros_like(mse)
    non_zero = mse > 0
    psnr[non_zero] = 20 * torch.log10(1.0 / torch.sqrt(mse[non_zero]))
    psnr[~non_zero] = float('inf')
    return psnr.mean().item()


def _gaussian_window(window_size, sigma=1.5):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(1) * g.unsqueeze(0)


def calculate_ssim(img1, img2, window_size=11):
    """SSIM for (C, H, W) tensors in [0, 1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    channels = img1.size(1)
    window = _gaussian_window(window_size).to(img1.device, img1.dtype)
    window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def batch_ssim(enhanced_batch, ground_truth_batch):
    """Calculates average SSIM for an entire batch."""
    return calculate_ssim(enhanced_batch, ground_truth_batch)
