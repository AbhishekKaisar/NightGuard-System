# Human Detection Module

**Lead:** Abhishek Kaisar Abhoy

## Approach
- **Model:** YOLOv8 Nano (pretrained, person class)
- **Preprocessing:** Gaussian blur (5x5 kernel) for noise suppression
- **Edge Detection:** Sobel operator for structural analysis
- **Enhancement:** Gamma correction for low-light improvement

## Key Results
| Condition | Confidence Score |
|-----------|-----------------|
| Raw (low-light) | 0.41 |
| Enhanced (gamma corrected) | 0.77 |

## Files
- `Human_Detection_Module.ipynb` — Full notebook with analysis and results
