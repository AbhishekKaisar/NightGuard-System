# Low-Light Image Enhancement Ensemble

**Lead:** Anindya Saha Ani (ID: 2221105042)

## Project Overview
This repository contains the **Low-Light Image Enhancement Ensemble**, a deep learning module developed as part of the broader **NightGuard-System**. Located in the `modules/enhancement/` directory, this module serves as a critical preprocessing step designed to restore dark, noisy, and corrupted images. By significantly improving the visual fidelity of night-time CCTV footage, this system enables more accurate and robust downstream tasks, such as face, human, and vehicle detection algorithms.

The core of this system is a powerful deep learning ensemble that fuses the outputs of four state-of-the-art base models using a U-Net meta-learner to intelligently reconstruct the final image.

---

## Architecture
The enhancement pipeline leverages a two-stage architecture: independent feature extraction followed by dynamic spatial fusion.

### 1. Parallel Feature Extractors (Base Models)
The first stage consists of four distinct, frozen, pre-trained neural networks operating in parallel. Each model is selected for a specific strength in handling low-light degradation:
*   **Zero-DCE:** Focuses on dynamic range adjustment and curve estimation to globally brighten the image without overexposing well-lit areas.
*   **KinD:** Employs Retinex physics-based principles to explicitly decouple illumination from reflectance, aggressively suppressing color noise often found in under-exposed sensors.
*   **RetinexNet:** Specializes in detailed illumination mapping, ensuring structural edges are preserved during the brightening process.
*   **Restormer:** A Vision Transformer (ViT) that captures long-range global context and ensures color constancy across the image, utilizing a robust BiasFree LayerNorm architecture.

### 2. U-Net Fusion Engine
The outputs of the four base models (each producing a 3-channel RGB image) are concatenated into a 12-channel feature map. This tensor is fed into the **U-Net Fusion Engine**. Acting as a trainable meta-learner, the U-Net dynamically learns spatial feature weighting—deciding pixel-by-pixel which base model provided the most accurate restoration—to synthesize a single, high-quality, artifact-free output image.

---

## Datasets & Dataloading
The system is engineered with dual-dataset capability to handle both standard synthetic low-light images and raw sensor data from real-world cameras.

*   **LOL (Low-Light) Dataset:** Handles standard RGB paired images (low-light vs. normal-light), providing a baseline for general enhancement.
*   **SID (See-in-the-Dark) Dataset:** Implements a custom dataloading pipeline using the `rawpy` library to directly ingest `.ARW` (Sony) and `.RAF` (Fuji) RAW sensor data. The dataloader explicitly maps multiple short-exposure (dark) captures of a scene to a single, high-quality long-exposure ground truth using unique Scene IDs, allowing the network to learn extreme low-light recovery directly from the sensor data.

**Dataset and Pre-trained Model Weights:** [Google Drive Link...](https://drive.google.com/drive/folders/1pRGWh1ckUeqWEAiR02CNyIGJtGIrDiZG?usp=sharing)

---

## Training Pipeline (`train_ensemble.py`)
The U-Net Fusion Engine is trained via a supervised learning loop while keeping the base models completely frozen. 

*   **Optimization:** The ensemble learns to map the 12-channel concatenated tensor to the high-quality ground truth.
*   **Custom FusionLoss:** The training objective is guided by a hybrid loss function:
    *   **L1 Loss:** Ensures structural integrity and accurate color reproduction.
    *   **VGG16 Perceptual Loss:** Extracts high-level feature maps from a pre-trained VGG16 network to ensure the enhanced image possesses natural, human-pleasing visual fidelity, preventing the blurry or smoothed textures common in standard L1/L2 optimizations.

---

## Evaluation Strategies
Evaluating high-resolution DSLR or modern CCTV images (often exceeding 4K resolution) typically causes 6GB VRAM Out-of-Memory (OOM) errors during the forward pass. To solve this, the project implements two distinct evaluation strategies:

### 1. `evaluate.py` (Fast Benchmarking)
This script utilizes dynamic bilinear downscaling and interpolation. If an image exceeds 1000 pixels in either dimension, it is proportionally downscaled before passing through the network, allowing for rapid calculation of PSNR and SSIM benchmarks across large datasets without crashing the GPU.

### 2. `evaluate_patch.py` (Full Resolution Preservation)
For deployment scenarios where maintaining the original ultra-high resolution is critical, this script employs patch-based inference. It slices the high-resolution input into overlapping patches (e.g., 512x512 tiles with 256-pixel overlap), predicts each patch sequentially to strictly manage VRAM, and then applies a bilinear weight map to seamlessly stitch the patches back together. This prevents visible boundary seams or grid artifacts, evaluating the final metrics on the true, full-resolution output.

---

## Deployment (`app.py`)
To facilitate easy testing and demonstration of the enhancement module, a local web interface is provided. Powered by Gradio, the `app.py` script loads the trained ensemble and provides a drag-and-drop UI directly in the browser, allowing researchers and operators to instantly enhance dark images.

---

## Setup & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment (`venv` or `conda`).

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

2. **Launch the Gradio Web Interface:**
   Start the local web server for drag-and-drop testing.
   ```bash
   python app.py
   ```
   *Open the provided local URL (e.g., `http://127.0.0.1:7860`) in your browser.*

3. **Train the Fusion Model:**
   Begin supervised training of the U-Net on top of the pre-trained base models. Configure dataset paths in `configs/config.yaml` prior to running.
   ```bash
   python train_ensemble.py
   ```

4. **Evaluate Performance:**
   Calculate PSNR and SSIM on your test splits.
   *   *For fast, downscaled benchmarking:*
       ```bash
       python evaluate.py
       ```
   *   *For full-resolution, patch-based inference:*
       ```bash
       python evaluate_patch.py
       ```

---

## Research & Acknowledgements

**Academic Context**
*This project is developed for academic purposes as part of the CSE468 course at North South University.*

*   **Researcher:** Anindya Saha Ani
*   **Supervisor:** [Dr. Mohammad Shifat-E-Rabbi](https://ece.northsouth.edu/people/dr-mohammad-shifat-e-rabbi/)
*   **Institution:** North South University

*Acknowledgements to the authors of the Zero-DCE, KinD, RetinexNet, and Restormer architectures, as well as the creators of the LOL and SID datasets.*