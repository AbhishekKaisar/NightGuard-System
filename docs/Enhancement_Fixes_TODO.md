# Enhancement Module — Issues & Fixes

**For: Anindya Saha Ani**
**Date: 2026-05-02**

---

## Issue 1: Over-Enhancement on Grayscale/IR CCTV Footage

### Problem

The DL ensemble over-brightens grayscale and IR camera images, producing washed-out output. Detection performance drops compared to the previous CLAHE approach.

**Test image:** `samples/x1080.jpg` (grayscale IR CCTV)

| Metric | CLAHE (before) | DL Ensemble (now) |
|--------|---------------|-------------------|
| Image quality | Natural contrast | Washed out / overexposed |
| Human confidence | 0.82 | 0.66 |
| Vehicles detected | 3 | 1 |

The ensemble was trained on LOL/SID datasets (color images), so it doesn't generalize well to grayscale IR footage.

### Fix Options (pick one or combine)

#### Option A: Auto-fallback to CLAHE

If the ensemble output is over-exposed, fall back to CLAHE automatically:

```python
def enhance_image(img):
    enhanced = enhance_image_with_ensemble(img)
    
    # Check if over-exposed (mean brightness > 200 on 0-255 scale)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 200:
        print("  (Ensemble over-exposed — falling back to CLAHE)")
        enhanced = enhance_image_clahe(img)
    
    return enhanced
```

#### Option B: Auto-detect GPU / CPU

```python
def enhance_image(img):
    if torch.cuda.is_available():
        return enhance_image_with_ensemble(img)
    else:
        return enhance_image_clahe(img)
```

#### Option C: Add a `--method` CLI flag

```python
parser.add_argument("--enhance", choices=["auto", "deep", "clahe"], default="auto",
                    help="Enhancement method: auto (GPU=deep, CPU=clahe), deep, or clahe")
```

#### Option D: Clamp the output

Add post-processing to prevent over-brightening:

```python
output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)

# Blend with original if over-exposed
gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
if gray.mean() > 200:
    alpha = 0.6  # blend ratio
    output_bgr = cv2.addWeighted(output_bgr, alpha, cv2_image, 1 - alpha, 0)
```

**Recommended:** Combine Option A + Option B. Use DL ensemble on GPU, CLAHE on CPU, and auto-fallback if over-exposed.

---

## Issue 2: Extremely Slow on CPU (~10 minutes per image)

### Problem

The pipeline loads 4 neural networks + U-Net fusion and runs patch-based inference. On a MacBook (CPU only), processing a 1920x1080 image takes ~10 minutes. This is too slow for the faculty demo on a laptop.

### Fix Options

#### Option 1: ONNX Export (Best — 3-5x faster on CPU)

Convert the full ensemble to ONNX format:

```python
import torch

# After loading and initializing the ensemble
ensemble.eval()
dummy_input = torch.randn(1, 3, 256, 256).to(device)

torch.onnx.export(
    ensemble,
    dummy_input,
    "modules/enhancement/weights/ensemble.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {2: "height", 3: "width"},
        "output": {2: "height", 3: "width"}
    },
    opset_version=17
)
```

Then use ONNX Runtime for inference:

```python
import onnxruntime as ort

session = ort.InferenceSession("modules/enhancement/weights/ensemble.onnx")
output = session.run(None, {"input": img_numpy})[0]
```

Add `onnxruntime` to `requirements.txt`:
```
onnxruntime>=1.17.0
```

#### Option 2: TorchScript (2-3x faster)

```python
ensemble.eval()
dummy_input = torch.randn(1, 3, 256, 256).to(device)
scripted = torch.jit.trace(ensemble, dummy_input)
scripted.save("modules/enhancement/weights/ensemble_scripted.pt")
```

Load and use:
```python
model = torch.jit.load("ensemble_scripted.pt")
output = model(img_tensor)
```

#### Option 3: Downscale Before Enhancement

Resize large images before the ensemble, then upscale:

```python
def enhance_image_with_ensemble(cv2_image, max_dim=640, tile_size=256, tile_overlap=32):
    h, w = cv2_image.shape[:2]
    
    # Downscale if too large
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(cv2_image, (int(w * scale), int(h * scale)))
    else:
        small = cv2_image
    
    # Run ensemble on smaller image (much faster)
    enhanced_small = _run_ensemble(small, tile_size, tile_overlap)
    
    # Upscale back to original size
    enhanced = cv2.resize(enhanced_small, (w, h))
    return enhanced
```

#### Option 4: Reduce Tile Size

In the current code, change tile_size from 256 to 128:

```python
# In main_pipeline.py, change the default:
def enhance_image_with_ensemble(cv2_image, tile_size=128, tile_overlap=16):
```

Smaller tiles = less memory per patch = faster, but may produce more visible seam artifacts.

#### Option 5: Half Precision (FP16)

```python
ensemble.half()  # Convert model to float16

# Before inference:
img_tensor = img_tensor.half()
```

This halves memory usage and speeds up computation, especially on Apple Silicon (M1/M2/M3).

### Recommended Approach

Combine these for the best result:

1. **Export to ONNX** — biggest single speedup
2. **Downscale to 640px** — reduces computation by ~6x for 1080p images
3. **Use FP16** — free speedup with minimal quality loss
4. **Keep CLAHE as CPU fallback** — guaranteed fast demo on any laptop

Expected result: ~30-60 seconds instead of 10 minutes on CPU.

---

## Issue 3: Missing CLAHE Fallback Function

### Problem

After your PR, the `enhance_image_clahe()` function was removed from `main_pipeline.py`. We need it back as a fallback.

### Fix

Add this function back to `main_pipeline.py`:

```python
def enhance_image_clahe(img):
    """Lightweight CLAHE enhancement (fast, no GPU needed)."""
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
    gamma = 0.7
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(enhanced, table)
```

---

## Summary Checklist

- [ ] Add `enhance_image_clahe()` back to `main_pipeline.py`
- [ ] Add auto-fallback when ensemble over-exposes (mean brightness > 200)
- [ ] Add auto GPU/CPU detection (ensemble on GPU, CLAHE on CPU)
- [ ] Export ensemble to ONNX for faster CPU inference
- [ ] Add downscale option for large images (max 640px)
- [ ] Test on all sample images: `x1080.jpg`, `sample-face.png`, `cctvsample.png`, `2015_06281.jpg`




**Test command:**
```bash
python3 main_pipeline.py --input samples/x1080.jpg --output results/test.jpg
```

Should produce natural-looking output and run in under 1 minute on CPU.
