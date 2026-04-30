"""
NightGuard: Low-Light Surveillance Detection Pipeline
CSE468 - Computer Vision Project | Group 8

Integrates all four modules:
  1. Low-Light Enhancement (CLAHE + Gamma Correction)
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
from ultralytics import YOLO


# ─── Enhancement ─────────────────────────────────────────────────────────────

def enhance_image(img):
    """
    Enhance a low-light image using CLAHE on the L channel (LAB color space)
    followed by gamma correction. Lightweight alternative to the deep learning
    ensemble in modules/enhancement/ (which requires pretrained weights).
    """
    # Step 1: Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

    # Step 2: CLAHE on L channel
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    # Step 3: Gamma correction (brighten dark regions)
    gamma = 0.7
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)

    return enhanced


# ─── Detection Functions ─────────────────────────────────────────────────────

def detect_faces(img, conf=0.3):
    """Detect faces using YOLOv8n-face model (falls back to yolov8n person class)."""
    face_weight = os.path.join(os.path.dirname(__file__), "modules", "face_detection", "yolov8n-face.pt")
    if os.path.exists(face_weight):
        model = YOLO(face_weight)
    else:
        print("      (yolov8n-face.pt not found, using yolov8n person detection as fallback)")
        model = YOLO("yolov8n.pt")
    results = model(img, conf=conf, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        detections.append({"bbox": (x1, y1, x2, y2), "conf": score, "label": "Face"})
    return detections


def detect_humans(img, conf=0.4):
    """Detect humans using YOLOv8n (COCO class 0 = person)."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    model = YOLO("yolov8n.pt")
    results = model(blurred, classes=[0], conf=conf, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        detections.append({"bbox": (x1, y1, x2, y2), "conf": score, "label": "Human"})
    return detections


def detect_vehicles(img, conf=0.4):
    """Detect vehicles using YOLOv8n (COCO classes: car, motorcycle, bus, truck)."""
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    vehicle_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

    model = YOLO("yolov8n.pt")
    results = model(img, classes=vehicle_classes, conf=conf, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = vehicle_names.get(cls_id, "Vehicle")
        detections.append({"bbox": (x1, y1, x2, y2), "conf": score, "label": label})
    return detections


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

def run_pipeline(input_path, output_path=None, show=False):
    """Run the full NightGuard detection pipeline on a single image."""
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image '{input_path}'")
        return

    print(f"Loaded: {input_path} ({img.shape[1]}x{img.shape[0]})")

    # Step 1: Enhance
    print("\n[1/4] Enhancing low-light image...")
    enhanced = enhance_image(img)

    # Step 2: Detect faces
    print("[2/4] Detecting faces...")
    faces = detect_faces(enhanced)
    print(f"      Found {len(faces)} face(s)")

    # Step 3: Detect humans
    print("[3/4] Detecting humans...")
    humans = detect_humans(enhanced)
    print(f"      Found {len(humans)} human(s)")

    # Step 4: Detect vehicles
    print("[4/4] Detecting vehicles...")
    vehicles = detect_vehicles(enhanced)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NightGuard Low-Light Detection Pipeline")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default=None, help="Path to save output image")
    parser.add_argument("--show", action="store_true", help="Display result in a window")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.show)
