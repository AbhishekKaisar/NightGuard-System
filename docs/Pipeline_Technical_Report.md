# NightGuard: Technical Pipeline Report

**CSE468 - Computer Vision Project | Group 8**
**Supervised by Dr. Mohammad Shifat-E-Rabbi, North South University**

---

## 1. Problem Statement

Standard object detection models (YOLO, RT-DETR, etc.) are trained on well-lit images and struggle significantly when applied to low-light CCTV footage. Nighttime surveillance cameras produce dark, noisy frames where people, faces, and vehicles become hard to detect. NightGuard solves this by applying image enhancement before detection, dramatically improving confidence scores across all detection tasks.

---

## 2. System Architecture

```
                    ┌─────────────────────┐
                    │   Input: Raw CCTV    │
                    │   (Low-Light Frame)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  STAGE 1: ENHANCE   │
                    │                     │
                    │ Deep Learning Ens.  │
                    │ (Zero-DCE, KinD,    │
                    │ RetinexNet,         │
                    │ Restormer) + U-Net  │
                    └──────────┬──────────┘
                               │
                       Enhanced Frame
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
┌─────────▼─────────┐ ┌───────▼────────┐ ┌────────▼────────┐
│  STAGE 2a: FACE   │ │ STAGE 2b: HUMAN│ │ STAGE 2c: VEHICLE│
│                   │ │                │ │                  │
│  YOLOv8n-face     │ │ Gaussian Blur  │ │ Pretrained YOLO  │
│  conf >= 0.3      │ │ + YOLOv8n      │ │ + Fine-tuned YOLO│
│                   │ │ class=0 (person)│ │ Smart selection  │
│                   │ │ conf >= 0.4    │ │ conf >= 0.4      │
└─────────┬─────────┘ └───────┬────────┘ └────────┬────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  STAGE 3: FUSE &    │
                    │  VISUALIZE          │
                    │                     │
                    │  Color-coded boxes: │
                    │  Green  = Face      │
                    │  Blue   = Human     │
                    │  Red    = Vehicle   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Output: Annotated  │
                    │  Detection Image    │
                    └─────────────────────┘
```

---

## 3. Stage 1: Low-Light Enhancement

**Owner: Anindya Saha Ani (ID: 2221105042)**

### 3.1 Problem

Low-light CCTV images suffer from:
- **Low contrast** — objects blend with the dark background
- **High noise** — sensor noise amplified in dark conditions
- **Poor color fidelity** — colors appear washed out or shifted

### 3.2 Approach (Integrated Pipeline)

The `main_pipeline.py` integrates a **Deep Learning Ensemble** to perform low-light enhancement. This approach uses multiple state-of-the-art enhancement models and a fusion network to selectively combine their strengths.

| Base Model | Strength |
|-----------|----------|
| Zero-DCE | Dynamic range curve estimation |
| KinD | Retinex-based illumination/reflectance decomposition |
| RetinexNet | Illumination mapping with edge preservation |
| Restormer | Vision Transformer for global context and color constancy |

These four frozen models each produce a 3-channel RGB output. The outputs are concatenated into a **12-channel tensor** and fed into a **U-Net Fusion Engine** that learns pixel-by-pixel which model provides the best restoration.

To handle arbitrary high-resolution images efficiently without exceeding GPU memory, the pipeline uses **patch-based inference**:
- The image is divided into `512x512` overlapping patches (e.g., 32-pixel overlap).
- Each patch is independently enhanced by the ensemble.
- The enhanced patches are blended back together using a linearly decaying weight map at the borders to prevent visible seams.

### 3.3 Previous Classical Approach

During prototyping, a lightweight 3-step classical enhancement (Fast Non-Local Means Denoising, CLAHE in LAB color space, and Gamma Correction) was used. While fast and computationally cheap, it often resulted in unnatural colors and amplified noise compared to the current Deep Learning Ensemble.

### 3.4 Impact on Detection

| Metric | Without Enhancement | With Enhancement |
|--------|-------------------|------------------|
| Human detection confidence | 0.41 | 0.73-0.82 |
| Face detection confidence | 0.42-0.65 | 0.70-0.90 |
| Vehicle detection confidence | 0.557 (pretrained) | 0.88-0.93 |

---

## 4. Stage 2a: Face Detection

**Owner: Midhat Bin Shazzad (ID: 2222560642)**

### 4.1 Model

**YOLOv8n-face** — a YOLOv8 Nano model with weights specifically trained for face detection.

### 4.2 How It Works

```python
face_weight = "modules/face_detection/yolov8n-face.pt"
model = YOLO(face_weight)
results = model(enhanced_img, conf=0.3, verbose=False)
```

1. Takes the **enhanced image** as input (not the raw dark image)
2. Runs YOLOv8 inference with a **0.3 confidence threshold** (lower than human/vehicle detection because faces are smaller and harder to detect)
3. Returns bounding box coordinates `(x1, y1, x2, y2)` and confidence score for each face

### 4.3 Preprocessing (Standalone Module)

Midhat's full module (`modules/face_detection/`) adds a **dual-input smart selector**:

```
Original Image ──→ detect() ──→ raw_dets (confidence scores)
                                                              ──→ Compare ──→ Best result
Enhanced Image ──→ detect() ──→ enh_dets (confidence scores)
```

- Runs detection on **both** raw and enhanced images
- Calculates average confidence for each set
- Applies a **1.05x bias** toward the enhanced result
- Selects whichever gives higher average confidence

### 4.4 Results

| Condition | Confidence Range |
|-----------|-----------------|
| Raw (low-light) | 0.42 - 0.65 |
| Enhanced (Deep Learning Ensemble) | 0.70 - 0.90 |

---

## 5. Stage 2b: Human Detection

**Owner: Abhishek Kaisar Abhoy (ID: 2221140042)**

### 5.1 Model

**YOLOv8n (Nano)** — pretrained on COCO dataset, filtered to **class 0 (person)** only.

### 5.2 How It Works

```python
blurred = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
model = YOLO("yolov8n.pt")
results = model(blurred, classes=[0], conf=0.4, verbose=False)
```

1. **Gaussian Blur** (5x5 kernel) — suppresses high-frequency noise that causes false positives in dark regions. The Gaussian function weights center pixels more than edge pixels, smoothing noise while preserving human silhouettes (which are larger structures)

2. **YOLOv8 Inference** — the CNN processes the image through:
   - **Backbone:** CSPDarknet extracts multi-scale features
   - **Neck:** PANet fuses features from different scales (detects both near and far humans)
   - **Head:** Predicts bounding boxes `(x, y, w, h)`, objectness scores, and class probabilities
   - `classes=[0]` filters output to person class only

3. **Non-Maximum Suppression (NMS)** — removes duplicate overlapping detections, keeping only the highest confidence box for each person

4. **Confidence threshold 0.4** — balances detection sensitivity with false positive prevention. Tested at 0.3 but produced false positives on reflective surfaces.

### 5.3 Classical Analysis (Standalone Module)

The full module (`modules/human_detection/`) also includes:

- **Sobel Edge Detection** — computes spatial gradients to visualize human contours:
  ```
  Gx = Sobel(image, dx=1, dy=0)  # horizontal edges
  Gy = Sobel(image, dx=0, dy=1)  # vertical edges
  Magnitude = sqrt(Gx^2 + Gy^2)  # combined edge map
  ```
- **Interactive Gamma Slider** — `ipywidgets` slider to dynamically adjust brightness and see real-time impact on YOLOv8 confidence scores

### 5.4 Why Gaussian Blur Matters

| With Blur | Without Blur |
|-----------|-------------|
| No false positives | False positives on car reflections/windows |
| Slightly lower confidence (0.73) | Higher confidence but wrong detections |
| Reliable across all test images | Unreliable on complex scenes |

While the enhancement step (Deep Learning Ensemble) handles significant noise reduction and restores details, Gaussian blur adds an extra layer of protection specifically for human detection, where false positives (e.g., detecting a car window reflection as a person) are worse than missed detections.

### 5.5 Results

| Condition | Confidence |
|-----------|-----------|
| Raw (low-light) | 0.41 |
| Enhanced (Deep Learning Ensemble without blur) | 0.77 |
| Enhanced (Deep Learning Ensemble + blur) | 0.73 - 0.82 |

---

## 6. Stage 2c: Vehicle Detection

**Owner: Maisha Tabassum (ID: 2222728042)**

### 6.1 Models

Two models working together with **smart selection**:

| Model | Type | Training |
|-------|------|----------|
| YOLOv8n (pretrained) | CNN | COCO dataset (general) |
| YOLOv8n (fine-tuned) | CNN | ExDark vehicle subset (2,320 images) |

### 6.2 How It Works

```python
# Step 1: Run pretrained model (reliable on general images)
model_pre = YOLO("yolov8n.pt")
results_pre = model_pre(img, classes=[2, 3, 5, 7], conf=0.4)  # car, motorcycle, bus, truck

# Step 2: Run fine-tuned model (better on dark images)
model_ft = YOLO("maisha_weights/yolo_finetune_exp2_best.pt")
results_ft = model_ft(img, conf=0.4)

# Step 3: Smart selection — pick whichever found more valid vehicles
```

### 6.3 Bounding Box Validation

A critical safeguard filters out bad detections:

```python
if (x2 - x1) * (y2 - y1) > img_area * 0.6:
    continue  # Skip — no single vehicle covers 60% of a frame
```

The fine-tuned model occasionally produces full-image bounding boxes on images outside its training distribution (e.g., top-down CCTV angles vs. the side-view ExDark training data). This filter prevents those from reaching the output.

### 6.4 Smart Model Selection Logic

```
Pretrained detections: [Car 0.88, Car 0.83, Car 0.61]  → 3 vehicles
Fine-tuned detections: []  (bad boxes filtered out)     → 0 vehicles

Result: Use pretrained (3 > 0) ✓
```

This ensures the pipeline always uses the better result — fine-tuned model for dark ExDark-style scenes, pretrained for general CCTV footage.

### 6.5 Vehicle Classes

| COCO ID | Class | Fine-tuned ID | Class |
|---------|-------|---------------|-------|
| 2 | Car | 0 | Car |
| 3 | Motorcycle | 1 | Bus |
| 5 | Bus | 2 | Bicycle |
| 7 | Truck | 3 | Motorcycle |

### 6.6 Results

| Model | Avg Confidence |
|-------|---------------|
| Pretrained YOLOv8n (baseline) | 0.557 |
| Fine-tuned YOLOv8n | 0.870 |
| RT-DETR baseline | 0.614 |
| RT-DETR fine-tuned | 0.893 |

---

## 7. Stage 3: Fusion and Visualization

### 7.1 Detection Fusion

All detections from the three parallel modules are combined into a single list:

```python
all_detections = faces + humans + vehicles
```

Each detection is a dictionary:
```python
{"bbox": (x1, y1, x2, y2), "conf": 0.82, "label": "Human"}
```

### 7.2 Color-Coded Visualization

| Detection Type | Color | RGB Value |
|---------------|-------|-----------|
| Face | Green | (0, 255, 0) |
| Human | Cyan/Blue | (255, 200, 0) |
| Vehicle (Car/Bus/Truck/Motorcycle) | Red | (0, 0, 255) |

Each bounding box includes:
- A colored rectangle around the detected object
- A filled label background with the class name and confidence score
- Text rendered in black for readability

---

## 8. End-to-End Test Results

### Test Image: x1080.jpg (Real CCTV Night Footage)

| Detection | Confidence | Accuracy |
|-----------|-----------|----------|
| Human | 0.82 | Correct — person near car |
| Car | 0.88 | Correct — white SUV |
| Car | 0.83 | Correct — white car |
| Car | 0.61 | Correct — dark pickup truck |

### Test Image: sample-face.png (IR Night Camera)

| Detection | Confidence | Accuracy |
|-----------|-----------|----------|
| Face | 0.75 | Correct — face with cap and beard |
| Human | 0.91 | Correct — full body |

### Test Image: cctvsample.png (Indoor CCTV)

| Detection | Confidence | Accuracy |
|-----------|-----------|----------|
| Face | 0.78 | Correct — clear face |
| Human | 0.93 | Correct — full body |

### Test Image: 2015_06281.jpg (ExDark Low-Light)

| Detection | Confidence | Accuracy |
|-----------|-----------|----------|
| Human | 0.73 | Correct |
| Human | 0.70 | Correct |
| Human | 0.58 | Correct |
| Human | 0.45 | Correct |

### Test Image: 2015_06282.jpg (Dark Street)

| Detection | Confidence | Accuracy |
|-----------|-----------|----------|
| Car | 0.93 | Correct |
| Car | 0.88 | Correct |
| Car | 0.70 | Correct |

---

## 9. Known Limitations

| Limitation | Reason | Potential Fix |
|-----------|--------|---------------|
| Heavily occluded persons missed | YOLOv8 Nano has limited feature capacity | Use YOLOv8-Large or RT-DETR |
| Top-down vehicle angles poorly detected | Training data is mostly side/front views | Fine-tune on aerial/CCTV vehicle datasets |
| Face detection fails in extreme darkness | Face features too small and degraded | Stronger enhancement or specialized IR face models |
| Fine-tuned model gives bad boxes on some images | Domain gap between ExDark and general CCTV | More diverse training data or domain adaptation |

---

## 10. Technology Stack

| Component | Technology |
|-----------|-----------|
| Enhancement | PyTorch (Deep Learning Ensemble: Zero-DCE, KinD, RetinexNet, Restormer, U-Net) |
| Detection Models | Ultralytics YOLOv8 Nano, RT-DETR |
| Deep Learning Framework | PyTorch |
| Legacy Enhancement | OpenCV (CLAHE, Gamma, Denoising) |
| Deployment | Gradio (web interface) |
| Language | Python 3.8+ |
| Hardware | CPU (no GPU required for inference, but GPU recommended for Ensemble) |

---

## 11. Team Contributions

| Member | ID | Module | Key Contribution |
|--------|-----|--------|-----------------|
| Anindya Saha Ani | 2221105042 | Enhancement | Deep learning ensemble (4 models + U-Net fusion), Legacy CLAHE pipeline |
| Midhat Bin Shazzad | 2222560642 | Face Detection | YOLOv8n-face with dual-input smart selector |
| Abhishek Kaisar Abhoy | 2221140042 | Human Detection | YOLOv8n with Gaussian blur preprocessing, pipeline integration |
| Maisha Tabassum | 2222728042 | Vehicle Detection | Fine-tuned YOLOv8n + RT-DETR on ExDark dataset |
