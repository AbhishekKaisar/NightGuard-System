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
│  (CLAHE, Gamma, etc.)       │
└──────────────┬───────────────┘
               │ Enhanced Frame
               ▼
┌──────────────────────────────────────────────┐
│            Parallel Detection                │
│                                              │
│  ┌────────────┐ ┌────────────┐ ┌───────────┐│
│  │   Face     │ │   Human    │ │  License  ││
│  │ Detection  │ │ Detection  │ │   Plate   ││
│  │ (Midhat)   │ │ (Abhishek) │ │ (Maisha)  ││
│  └────────────┘ └────────────┘ └───────────┘│
└──────────────────────────────────────────────┘
               │
               ▼
         Fused Results
   (Bounding Boxes, Labels, Confidence Scores)
```

---

## Team Members

| Name | ID | Role | Module Folder | Branch Name |
|------|----|------|---------------|-------------|
| Anindya Saha Ani | 2221105042 | Low-Light Enhancement Lead | `modules/enhancement/` | `feature/enhancement` |
| Midhat Bin Shazzad | 2222560642 | Face Detection Lead | `modules/face_detection/` | `feature/face-detection` |
| Abhishek Kaisar Abhoy | 2221140042 | Human Detection Lead | `modules/human_detection/` | `feature/human-detection` |
| Maisha Tabassum | 2222728042 | License Plate Detection | `modules/vehicle_detection/` | `feature/vehicle-detection` |

---

## Where to Put Your Work

Each member has a dedicated folder. **Put your code and notebook inside your folder only.**

```
modules/
├── enhancement/           ← Anindya: your enhancement code goes here
├── face_detection/        ← Midhat: your face detection code goes here
├── human_detection/       ← Abhishek: (already added)
└── vehicle_detection/     ← Maisha: your license plate code goes here
```

Also copy your `.ipynb` notebook into the `notebooks/` folder.

### Quick Steps for Each Member

```bash
# 1. Clone the repo
git clone https://github.com/AbhishekKaisar/NightGuard-System.git
cd NightGuard-System

# 2. Create your branch (use the branch name from the table above)
git checkout -b feature/<your-module-name>

# 3. Add your files into your module folder
#    Example for Anindya:
#    - Copy your code into modules/enhancement/
#    - Copy your notebook into notebooks/

# 4. Stage, commit, and push
git add .
git commit -m "Add <your-module-name> module"
git push origin feature/<your-module-name>

# 5. Go to GitHub and open a Pull Request to merge into main
```

> **Do NOT push dataset files or model weights (.pt, .h5) — they are too large for GitHub.**

---

## Project Structure

```
NightGuard-System/
├── modules/
│   ├── enhancement/        # Anindya — Low-light image enhancement
│   ├── face_detection/     # Midhat — Face detection (Haarcascade + MTCNN)
│   ├── human_detection/    # Abhishek — Human detection (YOLOv8)
│   └── vehicle_detection/  # Maisha — License plate detection
├── notebooks/              # All Jupyter notebooks
├── docs/                   # Project documentation & reports
├── results/                # Output images and evaluation metrics
├── requirements.txt        # Python dependencies
└── main_pipeline.py        # Final integrated pipeline (after merging)
```

## Modules

### 1. Low-Light Enhancement — Anindya Saha Ani
- Contrast/brightness correction (alpha-beta adjustment)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gamma correction with interactive parameter tuning
- Noise reduction via Gaussian blur

### 2. Face Detection — Midhat Bin Shazzad
- Haarcascade frontalface classifier (classical approach)
- MTCNN (Multi-task Cascaded Convolutional Networks)

### 3. Human Detection — Abhishek Kaisar Abhoy
- YOLOv8 pretrained model (person class)
- Gaussian blur preprocessing for noise suppression
- Sobel edge detection for structural analysis
- Confidence improvement: 0.41 (raw) → 0.77 (enhanced)

### 4. License Plate Detection — Maisha Tabassum
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
git clone https://github.com/AbhishekKaisar/NightGuard-System.git
cd NightGuard-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Tech Stack

- **Deep Learning:** Ultralytics YOLOv8, MTCNN
- **Computer Vision:** OpenCV
- **OCR:** EasyOCR
- **Languages:** Python 3.8+
- **Environment:** Google Colab / Local

## License

This project is developed for academic purposes as part of the CSE468 course at North South University.
