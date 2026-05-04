# NightGuard: Low-Light Surveillance Detection System

**CSE468 - Computer Vision Project | Group 8**
**Supervised by [Dr. Mohammad Shifat-E-Rabbi](https://ece.northsouth.edu/people/dr-mohammad-shifat-e-rabbi/), North South University**

---

## Overview

NightGuard is a real-time surveillance system designed to detect and identify objects in low-light CCTV footage. The system combines image enhancement techniques with deep learning-based detection to handle challenging nighttime conditions.

## Pipeline Architecture

```
Raw CCTV Frame (Low-Light)
        │
        ▼
┌──────────────────────────────┐
│  Low-Light Enhancement       │  ← Anindya (Module 1)
│  (DL Ensemble / CLAHE)      │
└──────────────┬───────────────┘
               │ Enhanced Frame
               ▼
┌──────────────────────────────────────────────┐
│            Parallel Detection                │
│                                              │
│  ┌────────────┐ ┌────────────┐ ┌───────────┐│
│  │   Face     │ │   Human    │ │  Vehicle  ││
│  │ Detection  │ │ Detection  │ │ Detection ││
│  │ (Midhat)   │ │ (Abhishek) │ │ (Maisha)  ││
│  └────────────┘ └────────────┘ └───────────┘│
└──────────────────────────────────────────────┘
               │
               ▼
         Fused Results
   (Bounding Boxes, Labels, Confidence Scores)
```

**[View Project Presentation](https://gamma.app/docs/NightGuard-A-RealTime-MultiClass-Object-Detection-System-for-LowL-2hfoz142lc5yf83?mode=present#card-y7dxb618h35dni2)**

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/AbhishekKaisar/NightGuard-System.git
cd NightGuard-System
pip install -r requirements.txt

# 2. Download required weights (not included in repo due to size)
#    - support/onnx_weights/    → from Google Drive (see Weights section below)
#    - support/maisha_weights/  → included in repo

# 3. Run the full pipeline on a sample image
python3 main.py --input data/x1080.jpg --output results/output.jpg
```

Or open `support/notebooks/NightGuard_Demo.ipynb` in Google Colab for an interactive demo.

---

## Team Members

| Name | ID | Role | Module Folder |
|------|----|------|---------------|
| Anindya Saha Ani | 2221105042 | Low-Light Enhancement Lead | `support/modules/enhancement/` |
| Midhat Bin Shazzad | 2222560642 | Face Detection Lead | `support/modules/face_detection/` |
| Abhishek Kaisar Abhoy | 2221140042 | Human Detection Lead | `support/modules/human_detection/` |
| Maisha Tabassum | 2222728042 | Vehicle Detection Lead | `support/modules/vehicle_detection/` |

---

## Project Structure

```
NightGuard-System/
├── main.py                 # Main pipeline — run this to detect
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── data/                   # Sample test images (datasets)
├── support/                # All supporting code and models
│   ├── modules/
│   │   ├── enhancement/    # Anindya — Deep learning ensemble
│   │   ├── face_detection/ # Midhat — YOLOv8n-face detection
│   │   ├── human_detection/# Abhishek — YOLOv8n human detection
│   │   └── vehicle_detection/ # Maisha — YOLOv8n + RT-DETR
│   ├── notebooks/          # Jupyter notebooks
│   ├── maisha_weights/     # Fine-tuned vehicle detection weights
│   ├── onnx_weights/       # ONNX models for fast CPU inference
│   ├── export_onnx.py      # ONNX export script
│   └── tune_pipeline.py    # Hyperparameter tuning script
├── others/                 # Presentations, reports, demo video
├── results/                # Output images and evaluation metrics
└── docs/                   # Technical documentation
```

## Modules

### 1. Low-Light Enhancement — Anindya Saha Ani
- **Deep Learning Ensemble:** Fuses four frozen base models (Zero-DCE, KinD, RetinexNet, Restormer Vision Transformer)
- **Meta-Learner:** U-Net Fusion Engine for dynamic spatial feature weighting
- **Optimized Inference:** ONNX Runtime FP16 on CPU, PyTorch on GPU — no GPU required
- **Exposure Safety Check:** Auto-fallback to CLAHE if the DL model over-exposes the image
- **Downscaling:** Images above 1080p are automatically downscaled before enhancement
- **Deployment:** Gradio web interface (`app.py`) for drag-and-drop inference

### 2. Face Detection — Midhat Bin Shazzad
- YOLOv8n with face detection weights
- CLAHE + Fast Non-Local Means Denoising preprocessing
- Dual-input smart selector (runs on both raw and enhanced, picks best confidence)
- Confidence improvement: ~0.42-0.65 (raw) → ~0.70-0.90 (enhanced)

### 3. Human Detection — Abhishek Kaisar Abhoy
- YOLOv8n pretrained model (person class)
- Gaussian blur preprocessing for noise suppression
- Sobel edge detection for structural analysis
- Interactive gamma correction slider for parameter tuning
- Confidence improvement: 0.41 (raw) → 0.77 (enhanced)

### 4. Vehicle Detection — Maisha Tabassum
- YOLOv8n baseline + fine-tuned on ExDark dataset (2,320 vehicle images)
- RT-DETR (transformer-based) fine-tuned for low-light vehicle detection
- Smart model selection: runs both pretrained and fine-tuned, picks best result
- Bounding box validation to filter bad detections
- Vehicle classes: Car, Bus, Bicycle, Motorcycle
- Best result: RT-DETR fine-tuned — 0.893 avg confidence

## Dataset

This project uses the **ExDark Dataset** — a collection of low-light images across 12 object categories. The dataset is not included in this repository due to size constraints.

**Download from Google Drive:** [ExDark Dataset](https://drive.google.com/drive/folders/15DIy444TvFkIGZi1AMNHUuYZcXZPtonQ?usp=sharing)

**Original Source:** [ExDark GitHub](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)

After downloading, place it in a `Dataset/` folder at the project root:
```
Dataset/
└── ExDark_Dataset/
    └── People/        # 609 low-light images
```

## Weights (Download Required)

Some weight files are too large for GitHub. Download from Google Drive and place in the correct folders:

| Folder | Contents | Download |
|--------|----------|----------|
| `onnx_weights/` | ONNX ensemble model for fast CPU inference | [Google Drive](https://drive.google.com/drive/folders/1pRGWh1ckUeqWEAiR02CNyIGJtGIrDiZG?usp=sharing) |
| `modules/enhancement/weights/` | Pretrained base model weights + U-Net fusion | [Google Drive](https://drive.google.com/drive/folders/1pRGWh1ckUeqWEAiR02CNyIGJtGIrDiZG?usp=sharing) |

> **Note:** `maisha_weights/` (vehicle detection) and `modules/face_detection/yolov8n-face.pt` are already included in the repo. `yolov8n.pt` auto-downloads on first run.

## Setup

```bash
# Clone the repository
git clone https://github.com/AbhishekKaisar/NightGuard-System.git
cd NightGuard-System

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Download weights from Google Drive (see Weights section above)
# Place onnx_weights/ and modules/enhancement/weights/ in the project root

# Run the pipeline
python3 main.py --input samples/x1080.jpg --output results/output.jpg
```

## Tech Stack

- **Deep Learning:** PyTorch, Ultralytics YOLOv8, RT-DETR, Restormer, RetinexNet, Zero-DCE, KinD
- **Optimized Inference:** ONNX Runtime (FP16 for CPU)
- **Computer Vision:** OpenCV
- **Deployment:** Gradio
- **Data Processing:** rawpy, NumPy, Pandas
- **Languages:** Python 3.8+
- **Environment:** CPU (ONNX) / GPU (PyTorch) / Google Colab

## License

This project is developed for academic purposes as part of the CSE468 course at North South University.
