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
│  (Deep Learning Ensemble)    │
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

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/AbhishekKaisar/NightGuard-System.git
cd NightGuard-System
pip install -r requirements.txt

# Run the full pipeline on a sample image
python main_pipeline.py --input samples/2015_06281.jpg --output results/output.jpg
```

Or open `notebooks/NightGuard_Demo.ipynb` in Google Colab for an interactive demo.

---

## Team Members

| Name | ID | Role | Module Folder |
|------|----|------|---------------|
| Anindya Saha Ani | 2221105042 | Low-Light Enhancement Lead | `modules/enhancement/` |
| Midhat Bin Shazzad | 2222560642 | Face Detection Lead | `modules/face_detection/` |
| Abhishek Kaisar Abhoy | 2221140042 | Human Detection Lead | `modules/human_detection/` |
| Maisha Tabassum | 2222728042 | Vehicle Detection Lead | `modules/vehicle_detection/` |

---

## Project Structure

```
NightGuard-System/
├── main_pipeline.py        # Integrated detection pipeline
├── export_onnx.py          # Script to export PyTorch ensemble to ONNX
├── tune_pipeline.py        # Pipeline hyperparameter tuning
├── modules/
│   ├── enhancement/        # Anindya — Deep learning ensemble
│   ├── face_detection/     # Midhat — YOLOv8n-face detection
│   ├── human_detection/    # Abhishek — YOLOv8n human detection
│   └── vehicle_detection/  # Maisha — YOLOv8n + RT-DETR vehicle detection
├── notebooks/
│   ├── NightGuard_Demo.ipynb           # Full pipeline demo
│   ├── Human_Detection_Module.ipynb
│   ├── FaceDetection.ipynb
│   ├── vehicle_detection_yolo_experiments.ipynb
│   ├── vehicle_detection_RT_DETR.ipynb
│   └── vehicle_detection_first_report.ipynb
├── samples/                # Sample test images
├── docs/                   # Project documentation & reports
├── onnx_weights/           # Exported ONNX models for fast CPU inference
├── results/                # Output images and evaluation metrics
└── requirements.txt        # Python dependencies
```

## Modules

### 1. Low-Light Enhancement — Anindya Saha Ani
- **Deep Learning Ensemble:** Fuses four frozen base models (Zero-DCE, KinD, RetinexNet, Restormer Vision Transformer)
- **Meta-Learner:** U-Net Fusion Engine for dynamic spatial feature weighting
- **Optimized Inference:** PyTorch GPU inference with ONNX Runtime FP16 fallback for fast CPU computation
- **Exposure Safety Check:** Fallback to CLAHE enhancement if the deep learning model over-exposes the image
- **Custom Dataloading:** Dual-dataset capability (LOL dataset + SID RAW sensor data)
- **Memory-Safe Evaluation:** Downscaling (max dimension 1080) and 256x256 patch-based inference for high-resolution images
- **Deployment:** Gradio web interface (`app.py`) for drag-and-drop inference

### 2. Face Detection — Midhat Bin Shazzad
- YOLOv8n with face detection weights
- CLAHE + Fast Non-Local Means Denoising preprocessing
- Dual-input smart selector (runs on both raw and enhanced, picks best confidence)
- Confidence improvement: ~0.42-0.65 (raw) → ~0.70-0.90 (enhanced)

### 3. Human Detection — Abhishek Kaisar Abhoy
- YOLOv8n pretrained model (person class)
- Gaussian blur (3x3 kernel) preprocessing for noise suppression
- Sobel edge detection for structural analysis
- Interactive gamma correction slider for parameter tuning
- Confidence improvement: 0.41 (raw) → 0.77 (enhanced)

### 4. Vehicle Detection — Maisha Tabassum
- YOLOv8n baseline + fine-tuned on ExDark dataset (2,320 vehicle images)
- RT-DETR (transformer-based) fine-tuned for low-light vehicle detection
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

### Enhancement Module Datasets

The enhancement module uses additional datasets for training:

*   **LOL (Low-Light) Dataset:** Standard RGB paired images (low-light vs. normal-light)
*   **SID (See-in-the-Dark) Dataset:** RAW sensor data (`.ARW` / `.RAF`) with short/long exposure pairs

**Enhancement Datasets & Weights:** [Google Drive Link](https://drive.google.com/drive/folders/1pRGWh1ckUeqWEAiR02CNyIGJtGIrDiZG?usp=sharing)

## Setup

```bash
# Clone the repository
git clone https://github.com/AbhishekKaisar/NightGuard-System.git
cd NightGuard-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For the enhancement module (requires PyTorch + GPU):
pip install -r modules/enhancement/requirements.txt
```

## Tech Stack

- **Deep Learning:** PyTorch, Ultralytics YOLOv8, RT-DETR, Restormer, RetinexNet, Zero-DCE, KinD, ONNX Runtime
- **Computer Vision:** OpenCV
- **Deployment:** Gradio
- **Data Processing:** rawpy, NumPy, Pandas
- **Languages:** Python 3.8+
- **Environment:** Local GPU / Google Colab

## License

This project is developed for academic purposes as part of the CSE468 course at North South University.

rposes as part of the CSE468 course at North South University.

