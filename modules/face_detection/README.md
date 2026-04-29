# Face Detection Module

**Lead:** Midhat Bin Shazzad (ID: 2222560642)

Approach

Model: YOLOv8 with Face Detection Weightage.

Preprocessing:

Fast Non-Local Means Denoising (light noise reduction)
CLAHE (Contrast Limited Adaptive Histogram Equalization) for controlled contrast enhancement

Enhancement Strategy:

Mild low-light enhancement to preserve facial structure
Avoided aggressive gamma correction to prevent feature distortion

Detection Strategy:

Dual input pipeline (original + enhanced image)
Confidence-based smart selector to choose best detection output

Key Results
Condition	Detection Confidence
Raw (low-light)	            ~0.45–0.65
Enhanced (CLAHE + denoise)	~0.60–0.80
Smart Selection Output	    Best of both

