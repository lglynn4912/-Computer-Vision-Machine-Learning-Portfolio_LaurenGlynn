# Computer Vision & Machine Learning Portfolio — Lauren Glynn

Welcome! This repository showcases my applied work in **machine learning and computer vision**.  
It spans **deep learning, classical CV, and applied pipelines**, demonstrating both breadth of fundamentals and depth in solving real-world problems.

---

## Featured Projects

### Fly Detection & Tracking (Capstone Project)
End-to-end ML pipeline for analyzing insect activity in ecological lab videos.  
- **Training**: feature extraction + Random Forest classifier, tracked with MLflow.  
- **Detection**: GPU-accelerated motion validation to detect moving insects only.  
- **Tracking**: aggregated detections into events (activity counts, durations, peaks).  
📂 [mtion_based_detection_YOLO_+_classicCV](./final_flycounter.py)

### YOLO-Style Detection
Experimented with YOLO detection for insect images.  
- Compared detection accuracy with classical CV methods.  
- Studied tradeoffs in speed vs performance.  
📂  [mtion_based_detection_YOLO_+_classicCV](./finalflydetector.py)

---

### Scene Classification with CNNs (MiniPlaces)
Built and trained a **LeNet-style CNN** from scratch on the MiniPlaces dataset (100 scene classes).  
- Designed CNN architecture with convolution, pooling, and FC layers.  
- Implemented training loop, checkpointing, and evaluation in PyTorch.  
- Applied data augmentation and optimization strategies.  
📂 [deeplearning_imageclassifier_from_scratch_PyTorch](./deeplearning_imageclassifier_from_scratch_PyTorch)

### Classical Computer Vision Implementations
Core CV algorithms implemented from scratch to strengthen fundamentals.  
- **Homography & Image Stitching**: panorama construction with RANSAC + blending  
  📂 [homography_stitching](./homography_stitching)  
- **Photometric Stereo**: surface normal & albedo recovery under varying lighting  
  📂 [photometric_stereo](./photometric_stereo)  
- **Optical Flow**: motion estimation visualized with arrow maps  
  📂 [optical_flow+object_tracking](./optical_flow+object_tracking)  
- **Object Tracking**: histogram-based target tracking across video frames  
  📂 [optical_flow+object_tracking](./optical_flow+object_tracking)  

---



---

## Additional Projects & Explorations
- **Insect Image-to-3D Reconstruction**  
  Experimented with single-image 3D reconstruction (DreamGaussian / Zero123) to generate coarse meshes from specimen photos.  
   Why: quick visualization for dataset QA & inspection in entomology workflows.  
   [View Repo](https://github.com/dashingzombie/insectclassifiers)  

---

## Tech Stack
- **Languages**: Python (NumPy, Pandas, Scikit-Learn, PyTorch)  
- **CV & DL**: OpenCV, TorchVision, YOLO frameworks  
- **Experiment Tracking**: MLflow  
- **Visualization**: Matplotlib, Seaborn  
- **Other Tools**: Git, Jupyter, Gradio  
