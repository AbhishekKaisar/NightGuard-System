# Prototype Workflow

<aside>
💡

*This is a prototype workflow to help us divide the project roles and write our individual 2-page updates for this upcoming Monday. Even though we are submitting separately, our timelines need to align perfectly to show a cohesive team effort, especially if the faculty cross checks our reports.*

</aside>

<aside>
💡

**Please note:** this is a guide, not the actual report. Pick one of the four roles below, take ownership of that specific field and use the bullet points to write your own project update. It will also help us the take responsibilities on some specific fields which is the key component of any team work and wrap up the project.

</aside>

<aside>
💡

**How to use this document:**
This workflow is a roadmap to help us survive Monday's deadline. Here is the plan:

1. **Pick a Role:** Claim one of the four specific "Leads" below. Do your Code and Research. Align it with course materials!
2. **Write Your Report:** Use the bullet points in your section to write your *individual* 2-page update for Google Classroom.
3. **Stay Cohesive:** Even though we are submitting individually, our reports must fit together perfectly like puzzle pieces. This breakdown ensures we cover all the faculty’s requirements as a team without overlapping our work or triggering plagiarism flags.
</aside>

### **Member 1: Low-Light Enhancement Lead**

- **Week 1 (March 3 – March 9): Requirement Analysis & Research**
    - Analyzed the "Low-Light Image Enhancement for CCTV" requirement.
    - Researched classical OpenCV methods, specifically comparing basic Contrast/Brightness Correction against Histogram Equalization to see which would process faster for CCTV frames.
    - Investigated how to structure the output of the enhancement module so it can be fed simultaneously into both facial recognition and YOLO-based object detection pipelines.
- **Week 2 (March 10 – March 16): Initial Prototyping**
    - Wrote the first Python script using OpenCV (`cv2`) to load dark images.
    - Implemented a basic alpha/beta contrast adjustment function (`cv2.convertScaleAbs`).
    - *Current Status:* The script successfully brightens the image. Currently looking into how to reduce the visual noise that appears in the darkest areas before passing the frames to the YOLOv8 and MTCNN models.
        
        ## The 2-Week "40% Progress" Team Breakdown
        

### **Member 2: Face Detection Lead (Haarcascade & MTCNN)**

- **Week 1 (March 3 – March 9): Requirement Analysis & Research**
    - Focused on the circled "Face Detection" requirement from the proposal.
    - Studied how the Haarcascade classifier (`haarcascade_frontalface_default.xml`) works and why it requires grayscale images to function properly.
    - Researched the deep learning architecture of MTCNN as an advanced alternative to handle complex lighting and facial angles better than standard Haarcascade.
- **Week 2 (March 10 – March 16): Initial Prototyping**
    - Developed a standalone Python script to run Haarcascade on well-lit test images, successfully drawing bounding boxes around detected faces.
    - Began setting up the deep learning libraries (TensorFlow/PyTorch) required to run the more accurate MTCNN detector.
    - *Current Status:* The baseline Haarcascade model works perfectly on clear images. Resolving some library dependency warnings for MTCNN. Next step is to test both against Member 1's enhanced dark images.

### **Member 3: Human Detection Lead**

- **Week 1 (March 3 – March 9): Requirement Analysis & Research**
    - Researched human detection models used in modern surveillance systems.
    - Identified YOLO-based object detection architectures as the most feasible option for real-time detection tasks.
    - Compared YOLOv5, YOLOv7, and YOLOv8 for speed, pretrained model availability, and compatibility with Google Colab.
    - Determined that YOLOv8 pretrained models already contain a “person” class, eliminating the need for custom training.
- **Week 2 (March 10 – March 16): Initial Prototyping**
    - Installed the Ultralytics YOLOv8 framework and configured the Colab environment.
    - Implemented code to load pretrained YOLO weights and run inference on enhanced frames.
    - Integrated the detection pipeline so that frames processed by the enhancement module are passed into the YOLO model.
    - *Current Status:* Model successfully detects human figures in test frames. The next step is refining detection thresholds and evaluating performance on enhanced vs non-enhanced frames.
    ****

### **Member 4: License Plate Detection**

- **Week 1 (March 3 – March 9): Requirement Analysis & Research**
    - Explored automated license plate recognition systems used in traffic monitoring and intelligent surveillance applications.
    - Studied two-stage ALPR pipelines involving **plate localization followed by text recognition**.
    - Identified **YOLOv8** as a suitable model for detecting small objects such as license plates in images.
    - Evaluated the **UFPR-ALPR Dataset**, which provides annotated vehicle images with license plate bounding boxes and text labels.
    - Researched optical character recognition tools capable of extracting alphanumeric characters from cropped plate regions, selecting **EasyOCR** for its simplicity and multilingual support.
- **Week 2 (March 10 – March 16): Initial Prototyping**
    - Set up the YOLOv8 detection model to identify license plate regions within vehicle images.
    - Implemented code to crop detected plate areas for further processing.
    - Integrated EasyOCR to read characters from the extracted plate region.
    - Conducted initial tests using publicly available vehicle images to verify that plate detection and text extraction work correctly.
    - *Current Status:* Basic license plate detection and OCR pipeline is operational. Future work will evaluate how low-light enhancement improves plate visibility and recognition accuracy.
    - ***Issue: Dataset has to be curated***

—> OR 

**Low-Light Vehicle Detection Lead**

- **Week 1 (March 3 – March 9): Requirement Analysis & Research**
    - Explored vehicle detection in low-light surveillance scenarios, motivated by real-world CCTV use cases where subjects may not be visible by face but arrive in identifiable vehicles.
    - Studied the YOLO family of object detection models, identifying YOLOv8n as the most suitable for real-time detection on resource-constrained environments due to its lightweight architecture and pretrained weights.
    - Evaluated the ExDark dataset, which provides exclusively dark images across 12 object categories including Car, Bus, Bicycle, and Motorbike, each with bounding box annotations — confirming its suitability as a benchmark for low-light vehicle detection.
    - Researched classical and deep learning-based image enhancement methods, comparing CLAHE (Contrast Limited Adaptive Histogram Equalization) against Zero-DCE as preprocessing stages prior to detection.
    - Identified detection confidence score as the primary evaluation metric to measure whether image enhancement meaningfully improves vehicle detection performance.
- **Week 2 (March 10 – March 16): Initial Prototyping**
    - Set up the YOLOv8n pretrained model using the Ultralytics library within a shared Google Colab environment to run inference on ExDark vehicle images.
    - Implemented a CLAHE-based enhancement pipeline using OpenCV, converting images to LAB color space to apply contrast enhancement exclusively to the luminance channel without distorting color information.
    - Developed a side-by-side detection pipeline that runs YOLOv8n on both the original dark image and the CLAHE-enhanced version, logging confidence scores for each detected vehicle to quantify the impact of enhancement.
    - Conducted initial tests on Car and Bus categories from the ExDark dataset, confirming that the detection and enhancement pipeline executes correctly end-to-end.
    - *Current Status:* The baseline pipeline is operational. Next steps include expanding tests across all vehicle classes in ExDark, integrating Zero-DCE as an alternative enhancement stage, and computing mAP scores to formally compare detection performance before and after enhancement.

 Sonnet 4.6