"""
NightGuard: Low-Light Surveillance Detection Pipeline
CSE468 - Computer Vision Project | Group 8

Integrates all four modules:
  1. Low-Light Enhancement (Deep Learning Ensemble)
  2. Face Detection (YOLOv8n-face)
  3. Human Detection (YOLOv8n)
  4. Vehicle Detection (YOLOv8n)

Usage:
  python main_pipeline.py --input samples/test.jpg --output results/output.jpg
"""

import argparse
import os
import cv2
import numpy as np
import yaml
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import onnxruntime as ort

from modules.enhancement.models.ensemble import LowLightEnsemble, load_base_weights
from modules.enhancement.models.base_models import ZeroDCE, KinD, RetinexNet, Restormer
from modules.enhancement.utils.helpers import _resolve_path, _resolve_weight_paths


# ─── Initialization ────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ONNX session globally for CPU inference
onnx_session = None
if not torch.cuda.is_available():
    onnx_path = "onnx_weights/ensemble_fp16.onnx"
    if os.path.exists(onnx_path):
        onnx_session = ort.InferenceSession(onnx_path)
    else:
        print(f"Warning: ONNX model not found at {onnx_path}. Run export_onnx.py first.")

# 1. Enhancement Ensemble (skip heavy PyTorch loading if ONNX is available on CPU)
ensemble = None
if onnx_session is None:
    dce = ZeroDCE()
    kind = KinD()
    retinex = RetinexNet()
    restormer = Restormer(LayerNorm_type='BiasFree')

    ensemble = LowLightEnsemble(dce, kind, retinex, restormer).to(device)

    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "enhancement")
    config_path = os.path.join(project_root, "configs", "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _resolve_weight_paths(cfg, project_root)
        load_base_weights(ensemble, cfg, device)

        fusion_weight_path = os.path.join(
            _resolve_path(project_root, cfg['weights']['save_dir']),
            cfg['weights']['save_name']
        )

        if os.path.exists(fusion_weight_path):
            ensemble.fusion_unet.load_state_dict(torch.load(fusion_weight_path, map_location=device))
        else:
            print(f"Warning: Could not find fusion weights at {fusion_weight_path}")
    else:
        print(f"Warning: Could not find config at {config_path}")

    ensemble.eval()

# 2. YOLO Detection Models
print("Loading YOLO detection models...")
yolo_general = YOLO("yolov8n.pt")

face_weight = os.path.join(os.path.dirname(__file__), "modules", "face_detection", "yolov8n-face.pt")
yolo_face = YOLO(face_weight) if os.path.exists(face_weight) else yolo_general

vehicle_finetuned = os.path.join(os.path.dirname(__file__), "maisha_weights", "yolo_finetune_exp2_best.pt")
yolo_vehicle_ft = YOLO(vehicle_finetuned) if os.path.exists(vehicle_finetuned) else None
print("All models loaded successfully.")


# ─── Enhancement Function ────────────────────────────────────────────────────

def enhance_image_with_ensemble(cv2_image, tile_size=256, tile_overlap=32):
    """Enhances a low-light image using the Deep Learning Ensemble with patch-based inference."""
    color_converted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted)
    
    transform = transforms.ToTensor()
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        b, c, h, w = img_tensor.shape
        if h <= tile_size and w <= tile_size:
            output_tensor = ensemble(img_tensor)
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
            
            output_tensor = out_tensor / weight_tensor
        
    output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
    output_np = output_tensor.permute(1, 2, 0).numpy()
    output_np = (output_np * 255.0).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    return output_bgr


def enhance_image_clahe(img):
    """Lightweight CLAHE enhancement (fast, no GPU needed)."""
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
    gamma = 0.7
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(enhanced, table)

def run_onnx_inference(cv2_image, session, tile_size=256, tile_overlap=32):
    """Enhances a low-light image using the ONNX model with patch-based inference."""
    color_converted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted)
    
    transform = transforms.ToTensor()
    img_tensor = transform(pil_img).unsqueeze(0).numpy().astype(np.float16)
    
    b, c, h, w = img_tensor.shape
    if h <= tile_size and w <= tile_size:
        inputs = {session.get_inputs()[0].name: img_tensor}
        output_tensor = session.run(None, inputs)[0]
    else:
        stride = tile_size - tile_overlap
        h_idx_list = list(range(0, h - tile_size, stride)) + [max(0, h - tile_size)]
        w_idx_list = list(range(0, w - tile_size, stride)) + [max(0, w - tile_size)]
        
        # Remove duplicates
        h_idx_list = list(dict.fromkeys(h_idx_list))
        w_idx_list = list(dict.fromkeys(w_idx_list))
        
        out_tensor = np.zeros_like(img_tensor)
        weight_tensor = np.zeros_like(img_tensor)
        
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_tensor[..., h_idx:h_idx+tile_size, w_idx:w_idx+tile_size]
                
                inputs = {session.get_inputs()[0].name: in_patch}
                out_patch = session.run(None, inputs)[0]
                
                patch_h, patch_w = out_patch.shape[-2:]
                
                y_ramp = np.minimum(np.arange(patch_h), np.arange(patch_h - 1, -1, -1))
                x_ramp = np.minimum(np.arange(patch_w), np.arange(patch_w - 1, -1, -1))
                weight_y = np.clip(y_ramp + 1, a_min=None, a_max=tile_overlap) / tile_overlap
                weight_x = np.clip(x_ramp + 1, a_min=None, a_max=tile_overlap) / tile_overlap
                weight = weight_y[:, None] * weight_x[None, :]
                weight = weight[None, None, :, :]
                
                out_tensor[..., h_idx:h_idx+patch_h, w_idx:w_idx+patch_w] += out_patch * weight
                weight_tensor[..., h_idx:h_idx+patch_h, w_idx:w_idx+patch_w] += weight
        
        output_tensor = out_tensor / weight_tensor
        
    output_tensor = np.clip(np.squeeze(output_tensor, axis=0), 0, 1)
    output_np = np.transpose(output_tensor, (1, 2, 0))
    output_np = (output_np * 255.0).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    return output_bgr

def smart_enhance(img):
    """
    Routes enhancement based on hardware capabilities and checks for over-exposure.
    """
    # 1. Hardware Routing
    if torch.cuda.is_available():
        # GPU: Run standard PyTorch ensemble (existing logic)
        enhanced = enhance_image_with_ensemble(img)
    else:
        # CPU: Run ONNX FP16 version for fast computation
        if onnx_session is not None:
            print("  [Notice] Running optimized ONNX FP16 Ensemble.")
            enhanced = run_onnx_inference(img, onnx_session)
        else:
            print("  [Notice] ONNX session not available. Falling back to PyTorch CPU.")
            enhanced = enhance_image_with_ensemble(img)
    
    # 2. Exposure Safety Check (Grayscale / IR Over-exposure)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 200:
        print("  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.")
        return enhance_image_clahe(img)
        
    return enhanced


# ─── Detection Functions ─────────────────────────────────────────────────────

def detect_faces(img, conf=0.3):
    """Detect faces using YOLOv8n-face model (falls back to yolov8n person class)."""
    results = yolo_face(img, conf=conf, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        detections.append({"bbox": (x1, y1, x2, y2), "conf": score, "label": "Face"})
    return detections


def detect_humans(img, conf=0.4, blur_ksize=(3, 3)):
    """Detect humans using YOLOv8n (COCO class 0 = person)."""
    if blur_ksize:
        img_to_process = cv2.GaussianBlur(img, blur_ksize, 0)
    else:
        img_to_process = img
        
    results = yolo_general(img_to_process, classes=[0], conf=conf, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        detections.append({"bbox": (x1, y1, x2, y2), "conf": score, "label": "Human"})
    return detections


def detect_vehicles(img, conf=0.4):
    """Detect vehicles using both fine-tuned and pretrained, pick best results."""
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    def _run_model(model, results, names):
        dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) * (y2 - y1) > img_area * 0.6:
                continue
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = names.get(cls_id, "Vehicle")
            dets.append({"bbox": (x1, y1, x2, y2), "conf": score, "label": label})
        return dets

    # Try pretrained YOLOv8n (reliable on general images)
    vehicle_classes = [2, 3, 5, 7]
    pretrained_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    results_pre = yolo_general(img, classes=vehicle_classes, conf=conf, verbose=False)
    dets_pre = _run_model(yolo_general, results_pre, pretrained_names)

    # Try fine-tuned model if available
    if yolo_vehicle_ft is not None:
        finetuned_names = {0: "Car", 1: "Bus", 2: "Bicycle", 3: "Motorcycle"}
        results_ft = yolo_vehicle_ft(img, conf=conf, verbose=False)
        dets_ft = _run_model(yolo_vehicle_ft, results_ft, finetuned_names)

        # Pick whichever found more vehicles (fine-tuned catches harder cases)
        if len(dets_ft) > len(dets_pre):
            return dets_ft

    return dets_pre


# ─── Visualization ───────────────────────────────────────────────────────────

# Color scheme: Green = Face, Blue = Human, Red = Vehicle
COLORS = {
    "Face": (0, 255, 0),
    "Human": (255, 200, 0),
    "Car": (0, 0, 255),
    "Motorcycle": (0, 0, 255),
    "Bus": (0, 0, 255),
    "Truck": (0, 0, 255),
}


def draw_results(img, detections):
    """Draw color-coded bounding boxes on the image."""
    output = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]
        color = COLORS.get(label, (255, 255, 255))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(output, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return output


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def process_single_image(input_path, output_path=None, show=False):
    """Run the full NightGuard detection pipeline on a single image."""
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image '{input_path}'")
        return None, []

    print(f"Loaded: {input_path} ({img.shape[1]}x{img.shape[0]})")

    # Step 1: Enhance
    print("\n[1/4] Enhancing low-light image...")
    h, w = img.shape[:2]
    if max(h, w) > 1080:
        scale = 1080 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    enhanced = smart_enhance(img)

    # Step 2: Detect faces
    print("[2/4] Detecting faces...")
    faces = detect_faces(enhanced, conf=0.3)
    print(f"      Found {len(faces)} face(s)")

    # Step 3: Detect humans
    print("[3/4] Detecting humans...")
    humans = detect_humans(enhanced, conf=0.4)
    print(f"      Found {len(humans)} human(s)")

    # Step 4: Detect vehicles
    print("[4/4] Detecting vehicles...")
    vehicles = detect_vehicles(enhanced, conf=0.4)
    print(f"      Found {len(vehicles)} vehicle(s)")

    # Combine and draw results
    all_detections = faces + humans + vehicles
    result = draw_results(enhanced, all_detections)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  NightGuard Detection Summary")
    print(f"{'='*50}")
    for det in all_detections:
        print(f"  {det['label']:12s} | Confidence: {det['conf']:.2f} | "
              f"Box: {det['bbox']}")
    if not all_detections:
        print("  No objects detected.")
    print(f"{'='*50}")

    # Save output
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"\nResult saved to: {output_path}")

    # Show output
    if show:
        cv2.imshow("NightGuard Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result, all_detections


def run_pipeline(input_path, output_path=None, show=False):
    """Run the full NightGuard detection pipeline on a single image or directory."""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    if os.path.isdir(input_path):
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(valid_extensions)]
        if not image_files:
            print(f"No valid images found in directory: {input_path}")
            return []
            
        print(f"Found {len(image_files)} images in '{input_path}'. Starting batch processing...")
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            
        results = []
        for img_name in image_files:
            img_in_path = os.path.join(input_path, img_name)
            img_out_path = os.path.join(output_path, img_name) if output_path else None
            res = process_single_image(img_in_path, img_out_path, show)
            if res is not None:
                results.append(res)
        return results

    elif os.path.isfile(input_path):
        img_out_path = output_path
        if output_path and (os.path.isdir(output_path) or not output_path.lower().endswith(valid_extensions)):
            os.makedirs(output_path, exist_ok=True)
            img_out_path = os.path.join(output_path, os.path.basename(input_path))
            
        return process_single_image(input_path, img_out_path, show)
        
    else:
        print(f"Error: Input path '{input_path}' does not exist.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NightGuard Low-Light Detection Pipeline")
    parser.add_argument("--input", required=True, help="Path to input image or directory containing images")
    parser.add_argument("--output", default=None, help="Path to save output image (or directory if input is a directory)")
    parser.add_argument("--show", action="store_true", help="Display result in a window")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.show)
