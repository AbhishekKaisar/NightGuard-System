import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import yaml

from models.ensemble import LowLightEnsemble, load_base_weights
from models.base_models import ZeroDCE, KinD, RetinexNet, Restormer
from utils.helpers import _resolve_weight_paths

# 2. Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dce = ZeroDCE()
kind = KinD()
retinex = RetinexNet()
restormer = Restormer(LayerNorm_type='BiasFree')

ensemble = LowLightEnsemble(dce, kind, retinex, restormer).to(device)

# Load base model weights to ensure the ensemble functions correctly
project_root = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(project_root, "configs", "config.yaml")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _resolve_weight_paths(cfg, project_root)
    load_base_weights(ensemble, cfg, device)

# 3. Weight Loading
fusion_weight_path = os.path.join(project_root, "weights", "best_ensemble", "unet_fusion_best.pth")
if os.path.exists(fusion_weight_path):
    ensemble.fusion_unet.load_state_dict(torch.load(fusion_weight_path, map_location=device))
else:
    print(f"Warning: Could not find fusion weights at {fusion_weight_path}")

ensemble.eval()

# 4. Inference Function
def enhance_image(img):
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = ensemble(img_tensor)
        
    output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
    output_img = transforms.ToPILImage()(output_tensor)
    return output_img

# 5. Gradio Interface
app = gr.Interface(
    fn=enhance_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Low-Light Image Enhancement",
    description="Upload a low-light image to enhance its visibility using a Deep Learning Ensemble.",
)

# 6. Launch
if __name__ == "__main__":
    app.launch()
