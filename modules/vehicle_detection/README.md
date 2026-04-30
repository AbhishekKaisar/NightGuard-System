## Vehicle Detection Module
**Lead:** Maisha Tabassum | ID: YOUR_ID

### Approach
- Model: YOLOv8n (fine-tuned) + RT-DETR (fine-tuned) on ExDark dataset
- Dataset: ExDark — Car, Bus, Bicycle, Motorbike (2,320 images)
- Experiments: Baseline → Fine-tuning → Regularization → LR Sweep

### Key Results
| Condition | Avg Confidence |
|---|---|
| Pretrained YOLOv8n (baseline) | 0.557 |
| Fine-tuned YOLOv8n (Exp 2) | 0.870 |
| Regularized YOLOv8n (Exp 3) | 0.845 |
| RT-DETR Baseline | 0.614 |
| RT-DETR Fine-tuned | 0.893 |

### Files
`Vehicle_Detection_Finetuning.ipynb` — YOLOv8 fine-tuning experiments  
`RTDETR_VehicleDetection.ipynb` — RT-DETR experiments  
`Week1_Analysis.ipynb` — Enhancement analysis (Zero-DCE vs CLAHE)

### Weights
Pre-trained weights available at:  
https://drive.google.com/drive/folders/1NZX5AOPGyUxuWpU38OSE_gmHzSadi2DZ?usp=sharing
