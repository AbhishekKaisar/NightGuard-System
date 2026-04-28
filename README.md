# NightGuard: Low-Light Surveillance Detection System

**CSE468 - Computer Vision Project | Group 8**

## Overview

NightGuard is a real-time surveillance system designed to detect and identify objects in low-light CCTV footage. The system combines image enhancement techniques with deep learning-based detection to handle challenging nighttime conditions.

## Pipeline Architecture

```
Raw CCTV Frame (Low-Light)
        │
        ▼
┌─────────────────────┐
│  Low-Light Enhancement  │  ← Module 1: Preprocessing
│  (CLAHE, Gamma, etc.)   │
└────────┬────────────┘
         │ Enhanced Frame
         ▼
┌────────────────────────────────────────────┐
│           Parallel Detection               │
│                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   Face   │  │  Human   │  │ Vehicle/ │ │
│  │Detection │  │Detection │  │  Plate   │ │
│  │(MTCNN/   │  │(YOLOv8)  │  │Detection │ │
│  │Haarcasc.)│  │          │  │(YOLOv8+  │ │
│  │          │  │          │  │ EasyOCR) │ │
│  └──────────┘  └──────────┘  └──────────┘ │
└────────────────────────────────────────────┘
         │
         ▼
   Fused Results
   (Bounding Boxes, Labels, Confidence Scores)
```

## Project Structure

```
NightGuard-System/
├── modules/
│   ├── enhancement/        # Low-light image enhancement
│   ├── face_detection/     # Face detection (Haarcascade + MTCNN)
│   ├── human_detection/    # Human detection (YOLOv8)
│   └── vehicle_detection/  # Vehicle & license plate detection
├── notebooks/              # Jupyter notebooks for each module
├── docs/                   # Project documentation & reports
├── results/                # Output images and evaluation metrics
├── requirements.txt        # Python dependencies
└── main_pipeline.py        # Integrated detection pipeline
```

## Modules

### 1. Low-Light Enhancement
- Contrast/brightness correction (alpha-beta adjustment)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gamma correction with interactive parameter tuning
- Noise reduction via Gaussian blur

### 2. Face Detection
- Haarcascade frontalface classifier (classical approach)
- MTCNN (Multi-task Cascaded Convolutional Networks)

### 3. Human Detection
- YOLOv8 pretrained model (person class)
- Gaussian blur preprocessing for noise suppression
- Sobel edge detection for structural analysis
- Confidence improvement: 0.41 (raw) → 0.77 (enhanced)

### 4. Vehicle & License Plate Detection
- YOLOv8 for vehicle detection in low-light
- EasyOCR for license plate text extraction
- CLAHE-enhanced detection pipeline

## Dataset

This project uses the **ExDark Dataset** — a collection of low-light images across 12 object categories. The dataset is not included in this repository due to size constraints.

**Download:** [ExDark Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)

After downloading, place it in a `Dataset/` folder at the project root.

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/NightGuard-System.git
cd NightGuard-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Team

| Role | Module |
|------|--------|
| Low-Light Enhancement Lead | Image preprocessing & enhancement |
| Face Detection Lead | Facial recognition pipeline |
| Human Detection Lead | Human figure detection (YOLOv8) |
| Vehicle Detection Lead | Vehicle & license plate detection |

## Tech Stack

- **Deep Learning:** Ultralytics YOLOv8, MTCNN
- **Computer Vision:** OpenCV
- **OCR:** EasyOCR
- **Languages:** Python 3.8+
- **Environment:** Google Colab / Local

## License

This project is developed for academic purposes as part of the CSE468 course.
