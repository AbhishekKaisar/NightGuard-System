import os
import torch
from torchvision.utils import save_image


def save_sample_images(low_light, enhanced, ground_truth, epoch, save_dir="results/samples/"):
    """
    Saves a side-by-side comparison of the Input, Enhanced, and Ground Truth images.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Grab the first image in the batch
    low_img = low_light[0].cpu()
    enh_img = enhanced[0].cpu().detach()
    gt_img = ground_truth[0].cpu()

    # Stack them horizontally (Input | Enhanced | Ground Truth)
    comparison_grid = torch.cat([low_img, enh_img, gt_img], dim=2)

    save_path = os.path.join(save_dir, f"epoch_{epoch}_comparison.png")
    save_image(comparison_grid, save_path)
    print(f"Saved sample image to {save_path}")


def load_single_image_to_tensor(image_path, device):
    """
    Loads a single image from disk and prepares it for the model.
    Used primarily for inference.py.
    """
    from PIL import Image
    import torchvision.transforms as transforms

    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()

    # Add a batch dimension (C, H, W) -> (1, C, H, W) and move to GPU
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor