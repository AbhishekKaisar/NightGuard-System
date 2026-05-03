(.venv) PS D:\CSE\NightGuard-System>  python main_pipeline.py --input samples/ --output results/              
All base model weights loaded successfully.
Loading YOLO detection models...
All models loaded successfully.
Found 8 images in 'samples/'. Starting batch processing...
Loaded: samples/2015_06255.jpg (640x410)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 2 face(s)
[3/4] Detecting humans...
      Found 3 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.68 | Box: (177, 80, 192, 100)
  Face         | Confidence: 0.38 | Box: (127, 125, 138, 140)
  Human        | Confidence: 0.87 | Box: (162, 75, 223, 263)
  Human        | Confidence: 0.77 | Box: (242, 83, 291, 211)
  Human        | Confidence: 0.60 | Box: (109, 122, 171, 192)
==================================================

Result saved to: results/2015_06255.jpg
Loaded: samples/2015_06260.jpg (640x448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 1 face(s)
[3/4] Detecting humans...
      Found 5 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.52 | Box: (342, 291, 354, 306)
  Human        | Confidence: 0.79 | Box: (323, 285, 368, 423)
  Human        | Confidence: 0.76 | Box: (558, 274, 639, 447)
  Human        | Confidence: 0.64 | Box: (375, 297, 416, 426)
  Human        | Confidence: 0.57 | Box: (279, 287, 309, 358)
  Human        | Confidence: 0.54 | Box: (215, 288, 237, 368)
==================================================

Result saved to: results/2015_06260.jpg
Loaded: samples/2015_06281.jpg (640x399)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 3 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Human        | Confidence: 0.81 | Box: (216, 149, 294, 356)
  Human        | Confidence: 0.68 | Box: (295, 153, 359, 348)
  Human        | Confidence: 0.46 | Box: (384, 128, 449, 365)
==================================================

Result saved to: results/2015_06281.jpg
Loaded: samples/2015_06282.jpg (640x427)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 1 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Human        | Confidence: 0.44 | Box: (301, 99, 367, 360)
  Car          | Confidence: 0.90 | Box: (162, 128, 302, 295)
==================================================

Result saved to: results/2015_06282.jpg
Loaded: samples/cctvsample.png (1992x1294)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 1 face(s)
[3/4] Detecting humans...
      Found 1 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.78 | Box: (1373, 442, 1493, 594)
  Human        | Confidence: 0.93 | Box: (1216, 384, 1624, 1294)
==================================================

Result saved to: results/cctvsample.png
Loaded: samples/sample-face.png (1446x842)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 1 face(s)
[3/4] Detecting humans...
      Found 1 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.75 | Box: (710, 380, 764, 450)
  Human        | Confidence: 0.91 | Box: (610, 351, 835, 840)
==================================================

Result saved to: results/sample-face.png
Loaded: samples/sample3.png (1050x648)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 1 face(s)
[3/4] Detecting humans...
      Found 1 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.73 | Box: (209, 287, 245, 334)
  Human        | Confidence: 0.87 | Box: (143, 266, 289, 632)
==================================================

Result saved to: results/sample3.png
Loaded: samples/x1080.jpg (1920x1080)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 1 human(s)
[4/4] Detecting vehicles...
      Found 3 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Human        | Confidence: 0.82 | Box: (1485, 663, 1688, 1076)
  Car          | Confidence: 0.88 | Box: (891, 130, 1147, 276)
  Car          | Confidence: 0.83 | Box: (629, 88, 812, 188)
  Car          | Confidence: 0.61 | Box: (739, 320, 1302, 715)
==================================================

Result saved to: results/x1080.jpg





(.venv) PS D:\CSE\NightGuard-System>  python main_pipeline.py --input data/tuning_set/ --output data/tuning_results_images_2/      
All base model weights loaded successfully.
Loading YOLO detection models...
All models loaded successfully.
Found 50 images in 'data/tuning_set/'. Starting batch processing...
Loaded: data/tuning_set/2015_02450.jpg (513x335)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 4 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.84 | Box: (309, 191, 393, 260)
  Car          | Confidence: 0.81 | Box: (173, 178, 238, 231)
  Car          | Confidence: 0.78 | Box: (0, 164, 174, 320)
  Car          | Confidence: 0.48 | Box: (281, 181, 354, 235)
==================================================

Result saved to: data/tuning_results_images_2/2015_02450.jpg
Loaded: data/tuning_set/2015_02467.jpg (508x338)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 3 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.92 | Box: (221, 221, 343, 321)
  Car          | Confidence: 0.87 | Box: (0, 211, 139, 335)
  Car          | Confidence: 0.85 | Box: (437, 223, 507, 283)
==================================================

Result saved to: data/tuning_results_images_2/2015_02467.jpg
Loaded: data/tuning_set/2015_02517.jpg (640x428)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 5 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.82 | Box: (482, 130, 639, 260)
  Car          | Confidence: 0.72 | Box: (390, 115, 475, 182)
  Car          | Confidence: 0.58 | Box: (158, 127, 227, 179)
  Car          | Confidence: 0.49 | Box: (70, 136, 187, 216)
  Car          | Confidence: 0.43 | Box: (235, 133, 260, 160)
==================================================

Result saved to: data/tuning_results_images_2/2015_02517.jpg
Loaded: data/tuning_set/2015_02678.JPG (2448x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_02678.JPG
Loaded: data/tuning_set/2015_02679.JPG (2448x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_02679.JPG
Loaded: data/tuning_set/2015_02680.JPG (2448x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_02680.JPG
Loaded: data/tuning_set/2015_02681.JPG (2448x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_02681.JPG
Loaded: data/tuning_set/2015_02682.JPG (2448x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.85 | Box: (299, 898, 2393, 1717)
==================================================

Result saved to: data/tuning_results_images_2/2015_02682.JPG
Loaded: data/tuning_set/2015_02684.JPG (2448x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_02684.JPG
Loaded: data/tuning_set/2015_02685.JPG (3264x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_02685.JPG
Loaded: data/tuning_set/2015_02686.JPG (3264x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_02686.JPG
Loaded: data/tuning_set/2015_02709.jpg (600x450)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 2 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.87 | Box: (81, 237, 232, 334)
  Car          | Confidence: 0.78 | Box: (209, 220, 274, 281)
==================================================

Result saved to: data/tuning_results_images_2/2015_02709.jpg
Loaded: data/tuning_set/2015_02742.jpg (710x479)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 2 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Human        | Confidence: 0.64 | Box: (51, 151, 76, 216)
  Human        | Confidence: 0.56 | Box: (220, 150, 245, 216)
  Car          | Confidence: 0.63 | Box: (378, 166, 556, 331)
==================================================

Result saved to: data/tuning_results_images_2/2015_02742.jpg
Loaded: data/tuning_set/2015_02743.jpg (600x450)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.46 | Box: (126, 235, 242, 305)
==================================================

Result saved to: data/tuning_results_images_2/2015_02743.jpg
Loaded: data/tuning_set/2015_02794.jpg (600x450)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Truck        | Confidence: 0.51 | Box: (50, 26, 507, 363)
==================================================

Result saved to: data/tuning_results_images_2/2015_02794.jpg
Loaded: data/tuning_set/2015_02799.jpg (800x600)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 5 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.89 | Box: (343, 259, 476, 326)
  Car          | Confidence: 0.86 | Box: (180, 290, 345, 360)
  Car          | Confidence: 0.66 | Box: (198, 356, 441, 465)
  Car          | Confidence: 0.64 | Box: (480, 240, 578, 295)
  Car          | Confidence: 0.64 | Box: (578, 234, 643, 279)
==================================================

Result saved to: data/tuning_results_images_2/2015_02799.jpg
Loaded: data/tuning_set/2015_02801.jpg (1600x1200)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 4 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.84 | Box: (709, 590, 914, 709)
  Truck        | Confidence: 0.55 | Box: (1401, 558, 1599, 887)
  Car          | Confidence: 0.53 | Box: (558, 576, 640, 661)
  Car          | Confidence: 0.41 | Box: (624, 578, 741, 676)
==================================================

Result saved to: data/tuning_results_images_2/2015_02801.jpg
Loaded: data/tuning_set/2015_02803.jpg (1092x731)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 3 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.79 | Box: (646, 211, 770, 338)
  Car          | Confidence: 0.64 | Box: (1002, 170, 1091, 244)
  Car          | Confidence: 0.57 | Box: (906, 178, 1013, 244)
==================================================

Result saved to: data/tuning_results_images_2/2015_02803.jpg
Loaded: data/tuning_set/2015_02808.jpg (1024x682)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_02808.jpg
Loaded: data/tuning_set/2015_02814.jpg (1032x774)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.88 | Box: (562, 345, 918, 564)
==================================================

Result saved to: data/tuning_results_images_2/2015_02814.jpg
Loaded: data/tuning_set/2015_02865.JPEG (500x328)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 3 face(s)
[3/4] Detecting humans...
      Found 1 human(s)
[4/4] Detecting vehicles...
      Found 2 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.59 | Box: (388, 74, 399, 86)
  Face         | Confidence: 0.45 | Box: (21, 53, 31, 65)
  Face         | Confidence: 0.35 | Box: (62, 71, 68, 80)
  Human        | Confidence: 0.54 | Box: (378, 69, 411, 141)
  Car          | Confidence: 0.76 | Box: (205, 83, 414, 192)
  Car          | Confidence: 0.59 | Box: (418, 106, 489, 136)
==================================================

Result saved to: data/tuning_results_images_2/2015_02865.JPEG
Loaded: data/tuning_set/2015_02868.JPEG (375x500)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 1 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Human        | Confidence: 0.79 | Box: (5, 234, 41, 335)
  Car          | Confidence: 0.42 | Box: (48, 259, 87, 328)
==================================================

Result saved to: data/tuning_results_images_2/2015_02868.JPEG
Loaded: data/tuning_set/2015_02870.JPEG (500x333)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 2 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.76 | Box: (364, 100, 433, 184)
  Car          | Confidence: 0.71 | Box: (180, 126, 381, 328)
==================================================

Result saved to: data/tuning_results_images_2/2015_02870.JPEG
Loaded: data/tuning_set/2015_02881.jpg (1024x768)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.65 | Box: (10, 244, 925, 598)
==================================================

Result saved to: data/tuning_results_images_2/2015_02881.jpg
Loaded: data/tuning_set/2015_02911.jpg (1024x696)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.65 | Box: (301, 217, 1022, 686)
==================================================

Result saved to: data/tuning_results_images_2/2015_02911.jpg
Loaded: data/tuning_set/2015_02915.jpg (768x1024)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 7 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.75 | Box: (499, 654, 626, 732)
  Car          | Confidence: 0.66 | Box: (642, 653, 750, 735)
  Car          | Confidence: 0.65 | Box: (719, 662, 767, 745)
  Car          | Confidence: 0.63 | Box: (420, 662, 505, 730)
  Car          | Confidence: 0.59 | Box: (78, 693, 345, 926)
  Car          | Confidence: 0.49 | Box: (620, 658, 672, 725)
  Car          | Confidence: 0.41 | Box: (68, 665, 197, 757)
==================================================

Result saved to: data/tuning_results_images_2/2015_02915.jpg
Loaded: data/tuning_set/2015_02916.jpg (1024x768)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 4 human(s)
[4/4] Detecting vehicles...
      Found 6 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Human        | Confidence: 0.81 | Box: (404, 537, 488, 673)
  Human        | Confidence: 0.67 | Box: (775, 509, 798, 578)
  Human        | Confidence: 0.45 | Box: (703, 498, 732, 554)
  Human        | Confidence: 0.42 | Box: (906, 515, 984, 719)
  Car          | Confidence: 0.88 | Box: (482, 522, 732, 634)
  Car          | Confidence: 0.88 | Box: (0, 534, 263, 651)
  Car          | Confidence: 0.87 | Box: (328, 510, 481, 583)
  Car          | Confidence: 0.65 | Box: (182, 503, 314, 569)
  Car          | Confidence: 0.62 | Box: (834, 509, 1023, 623)
  Car          | Confidence: 0.42 | Box: (959, 519, 1024, 609)
==================================================

Result saved to: data/tuning_results_images_2/2015_02916.jpg
Loaded: data/tuning_set/2015_02922.jpg (1023x678)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 4 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.92 | Box: (141, 429, 518, 594)
  Car          | Confidence: 0.89 | Box: (838, 445, 1023, 554)
  Car          | Confidence: 0.89 | Box: (560, 461, 832, 572)
  Car          | Confidence: 0.49 | Box: (0, 465, 64, 600)
==================================================

Result saved to: data/tuning_results_images_2/2015_02922.jpg
Loaded: data/tuning_set/2015_02947.jpg (600x450)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 3 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.92 | Box: (7, 177, 291, 303)
  Car          | Confidence: 0.89 | Box: (378, 164, 577, 242)
  Car          | Confidence: 0.89 | Box: (225, 168, 461, 268)
==================================================

Result saved to: data/tuning_results_images_2/2015_02947.jpg
Loaded: data/tuning_set/2015_02971.jpg (860x647)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 2 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.91 | Box: (309, 334, 477, 465)
  Car          | Confidence: 0.74 | Box: (12, 343, 324, 626)
==================================================

Result saved to: data/tuning_results_images_2/2015_02971.jpg
Loaded: data/tuning_set/2015_02987.jpg (600x900)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 2 human(s)
[4/4] Detecting vehicles...
      Found 2 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Human        | Confidence: 0.56 | Box: (420, 777, 474, 899)
  Human        | Confidence: 0.47 | Box: (21, 516, 43, 599)
  Car          | Confidence: 0.75 | Box: (261, 525, 352, 610)
  Car          | Confidence: 0.58 | Box: (298, 467, 363, 520)
==================================================

Result saved to: data/tuning_results_images_2/2015_02987.jpg
Loaded: data/tuning_set/2015_02997.jpg (646x492)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_02997.jpg
Loaded: data/tuning_set/2015_03007.png (1280x720)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_03007.png
Loaded: data/tuning_set/2015_06446.jpg (500x375)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 6 face(s)
[3/4] Detecting humans...
      Found 3 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.80 | Box: (213, 180, 244, 221)
  Face         | Confidence: 0.80 | Box: (219, 129, 245, 163)
  Face         | Confidence: 0.78 | Box: (251, 113, 278, 146)
  Face         | Confidence: 0.77 | Box: (183, 126, 208, 158)
  Face         | Confidence: 0.73 | Box: (151, 123, 177, 156)
  Face         | Confidence: 0.58 | Box: (293, 123, 322, 157)
  Human        | Confidence: 0.70 | Box: (205, 101, 310, 374)
  Human        | Confidence: 0.69 | Box: (80, 119, 213, 374)
  Human        | Confidence: 0.69 | Box: (291, 113, 411, 374)
==================================================

Result saved to: data/tuning_results_images_2/2015_06446.jpg
Loaded: data/tuning_set/2015_06567.jpg (640x480)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 4 face(s)
[3/4] Detecting humans...
      Found 6 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.80 | Box: (538, 155, 577, 204)
  Face         | Confidence: 0.78 | Box: (426, 202, 486, 275)
  Face         | Confidence: 0.67 | Box: (190, 175, 217, 211)
  Face         | Confidence: 0.52 | Box: (251, 139, 329, 238)
  Human        | Confidence: 0.92 | Box: (0, 134, 128, 478)
  Human        | Confidence: 0.86 | Box: (490, 139, 638, 412)
  Human        | Confidence: 0.77 | Box: (153, 202, 259, 479)
  Human        | Confidence: 0.76 | Box: (156, 90, 434, 479)
  Human        | Confidence: 0.65 | Box: (398, 170, 527, 477)
  Human        | Confidence: 0.48 | Box: (153, 161, 219, 317)
==================================================

Result saved to: data/tuning_results_images_2/2015_06567.jpg
Loaded: data/tuning_set/2015_06616.JPG (4320x3240)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 1 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.51 | Box: (2374, 1503, 2767, 1930)
==================================================

Result saved to: data/tuning_results_images_2/2015_06616.JPG
Loaded: data/tuning_set/2015_06622.JPG (3104x1746)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_06622.JPG
Loaded: data/tuning_set/2015_06677.jpg (4160x3120)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 3 face(s)
[3/4] Detecting humans...
      Found 4 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.79 | Box: (1391, 1143, 1686, 1536)
  Face         | Confidence: 0.74 | Box: (2796, 1208, 3064, 1547)
  Face         | Confidence: 0.51 | Box: (3374, 1293, 3552, 1517)
  Human        | Confidence: 0.87 | Box: (1825, 1193, 2800, 3117)
  Human        | Confidence: 0.84 | Box: (2659, 1199, 3523, 2930)
  Human        | Confidence: 0.52 | Box: (961, 1048, 1803, 3118)
  Human        | Confidence: 0.48 | Box: (209, 1306, 945, 3101)
==================================================

Result saved to: data/tuning_results_images_2/2015_06677.jpg
Loaded: data/tuning_set/2015_06684.jpg (4160x3120)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 2 face(s)
[3/4] Detecting humans...
      Found 2 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.74 | Box: (2284, 1265, 2460, 1486)
  Face         | Confidence: 0.71 | Box: (1856, 1381, 2039, 1597)
  Human        | Confidence: 0.63 | Box: (1472, 1294, 2387, 3118)
  Human        | Confidence: 0.49 | Box: (1486, 1234, 2734, 3074)
==================================================

Result saved to: data/tuning_results_images_2/2015_06684.jpg
Loaded: data/tuning_set/2015_06685.jpg (4160x3120)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 7 face(s)
[3/4] Detecting humans...
      Found 3 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.77 | Box: (2167, 1141, 2370, 1432)
  Face         | Confidence: 0.73 | Box: (1705, 1634, 1928, 1922)
  Face         | Confidence: 0.71 | Box: (1573, 1148, 1764, 1386)
  Face         | Confidence: 0.71 | Box: (1847, 1119, 2043, 1350)
  Face         | Confidence: 0.71 | Box: (1990, 1468, 2209, 1742)
  Face         | Confidence: 0.68 | Box: (2651, 1208, 2849, 1460)
  Face         | Confidence: 0.55 | Box: (2453, 1205, 2645, 1442)
  Human        | Confidence: 0.85 | Box: (894, 1535, 1994, 3117)
  Human        | Confidence: 0.68 | Box: (2495, 1181, 3324, 3117)
  Human        | Confidence: 0.52 | Box: (1774, 1059, 2106, 1561)
==================================================

Result saved to: data/tuning_results_images_2/2015_06685.jpg
Loaded: data/tuning_set/2015_06686.jpg (4160x3120)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 3 face(s)
[3/4] Detecting humans...
      Found 4 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.77 | Box: (1814, 1094, 2070, 1423)
  Face         | Confidence: 0.76 | Box: (3274, 937, 3523, 1242)
  Face         | Confidence: 0.66 | Box: (1375, 1141, 1597, 1431)
  Human        | Confidence: 0.86 | Box: (711, 1083, 1626, 3113)
  Human        | Confidence: 0.82 | Box: (1476, 1003, 2303, 3111)
  Human        | Confidence: 0.67 | Box: (2734, 916, 3700, 3109)
  Human        | Confidence: 0.60 | Box: (2186, 977, 2941, 3113)
==================================================

Result saved to: data/tuning_results_images_2/2015_06686.jpg
Loaded: data/tuning_set/2015_06694.jpg (1600x1200)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  No objects detected.
==================================================

Result saved to: data/tuning_results_images_2/2015_06694.jpg
Loaded: data/tuning_set/2015_06771.jpg (1024x768)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 1 face(s)
[3/4] Detecting humans...
      Found 5 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.35 | Box: (702, 383, 712, 396)
  Human        | Confidence: 0.85 | Box: (344, 381, 414, 579)
  Human        | Confidence: 0.84 | Box: (92, 386, 235, 700)
  Human        | Confidence: 0.79 | Box: (402, 365, 474, 663)
  Human        | Confidence: 0.62 | Box: (172, 377, 249, 697)
  Human        | Confidence: 0.54 | Box: (523, 367, 574, 532)
==================================================

Result saved to: data/tuning_results_images_2/2015_06771.jpg
Loaded: data/tuning_set/2015_06772.jpg (1024x681)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 2 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Human        | Confidence: 0.82 | Box: (331, 272, 411, 438)
  Human        | Confidence: 0.74 | Box: (407, 282, 481, 407)
==================================================

Result saved to: data/tuning_results_images_2/2015_06772.jpg
Loaded: data/tuning_set/2015_06774.jpg (768x1024)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 6 face(s)
[3/4] Detecting humans...
      Found 9 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.78 | Box: (639, 705, 671, 748)
  Face         | Confidence: 0.65 | Box: (561, 732, 589, 767)
  Face         | Confidence: 0.58 | Box: (91, 653, 105, 673)
  Face         | Confidence: 0.34 | Box: (375, 703, 386, 719)
  Face         | Confidence: 0.34 | Box: (170, 679, 183, 699)
  Face         | Confidence: 0.33 | Box: (173, 679, 187, 699)
  Human        | Confidence: 0.88 | Box: (596, 700, 712, 1024)
  Human        | Confidence: 0.86 | Box: (475, 699, 615, 1024)
  Human        | Confidence: 0.82 | Box: (188, 693, 277, 984)
  Human        | Confidence: 0.79 | Box: (149, 670, 215, 909)
  Human        | Confidence: 0.77 | Box: (73, 648, 133, 844)
  Human        | Confidence: 0.61 | Box: (35, 715, 89, 862)
  Human        | Confidence: 0.56 | Box: (358, 694, 412, 816)
  Human        | Confidence: 0.55 | Box: (615, 640, 665, 737)
  Human        | Confidence: 0.48 | Box: (290, 742, 340, 813)
==================================================

Result saved to: data/tuning_results_images_2/2015_06774.jpg
Loaded: data/tuning_set/2015_06775.jpg (576x1024)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
  [Notice] DL Ensemble over-exposed the image. Falling back to CLAHE.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 2 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.80 | Box: (0, 702, 164, 833)
  Car          | Confidence: 0.72 | Box: (431, 668, 576, 981)
==================================================

Result saved to: data/tuning_results_images_2/2015_06775.jpg
Loaded: data/tuning_set/2015_06792.jpg (600x416)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 1 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Human        | Confidence: 0.87 | Box: (321, 222, 373, 362)
==================================================

Result saved to: data/tuning_results_images_2/2015_06792.jpg
Loaded: data/tuning_set/2015_06799.jpg (3264x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 5 face(s)
[3/4] Detecting humans...
      Found 7 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.80 | Box: (2637, 507, 2913, 870)
  Face         | Confidence: 0.80 | Box: (1951, 605, 2172, 889)
  Face         | Confidence: 0.75 | Box: (2363, 564, 2562, 814)
  Face         | Confidence: 0.75 | Box: (1396, 668, 1628, 970)
  Face         | Confidence: 0.51 | Box: (3156, 595, 3262, 905)
  Human        | Confidence: 0.82 | Box: (1364, 499, 2141, 2440)
  Human        | Confidence: 0.79 | Box: (2584, 379, 3261, 2447)
  Human        | Confidence: 0.52 | Box: (0, 725, 428, 2410)
  Human        | Confidence: 0.51 | Box: (369, 516, 1272, 2447)
  Human        | Confidence: 0.48 | Box: (2237, 517, 2775, 2444)
  Human        | Confidence: 0.46 | Box: (861, 528, 1543, 2448)
  Human        | Confidence: 0.40 | Box: (1889, 454, 2460, 2423)
==================================================

Result saved to: data/tuning_results_images_2/2015_06799.jpg
Loaded: data/tuning_set/2015_06800.jpg (3264x2448)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 5 face(s)
[3/4] Detecting humans...
      Found 8 human(s)
[4/4] Detecting vehicles...
      Found 0 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Face         | Confidence: 0.71 | Box: (2378, 1006, 2494, 1162)
  Face         | Confidence: 0.63 | Box: (1074, 895, 1204, 1053)
  Face         | Confidence: 0.63 | Box: (1662, 889, 1797, 1050)
  Face         | Confidence: 0.47 | Box: (2755, 1012, 2887, 1176)
  Face         | Confidence: 0.33 | Box: (188, 775, 298, 934)
  Human        | Confidence: 0.81 | Box: (490, 821, 1057, 2441)
  Human        | Confidence: 0.77 | Box: (2457, 960, 3049, 2445)
  Human        | Confidence: 0.71 | Box: (1735, 894, 2169, 2422)
  Human        | Confidence: 0.63 | Box: (942, 783, 1292, 2438)
  Human        | Confidence: 0.59 | Box: (1183, 771, 1624, 2448)
  Human        | Confidence: 0.43 | Box: (1936, 974, 2278, 2407)
  Human        | Confidence: 0.43 | Box: (2147, 985, 2598, 2445)
  Human        | Confidence: 0.42 | Box: (1477, 808, 1814, 2429)
==================================================

Result saved to: data/tuning_results_images_2/2015_06800.jpg
Loaded: data/tuning_set/2017_07357.jpg (1024x648)

[1/4] Enhancing low-light image...
  [Notice] CPU detected. Running optimized ONNX FP16 Ensemble.
[2/4] Detecting faces...
      Found 0 face(s)
[3/4] Detecting humans...
      Found 0 human(s)
[4/4] Detecting vehicles...
      Found 1 vehicle(s)

==================================================
  NightGuard Detection Summary
==================================================
  Car          | Confidence: 0.76 | Box: (252, 370, 446, 479)
==================================================

Result saved to: data/tuning_results_images_2/2017_07357.jpg