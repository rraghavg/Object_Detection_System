# 🚀 Real-Time Object Detection & Tracking System

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-supported-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-purple)
![Tests](https://img.shields.io/badge/Tests-20%2F20%20passed-brightgreen)

## 📋 Overview

A clean, modular, and interview-ready computer vision repository for real-time object detection and multi-object tracking. Built with **YOLOv8**, **OpenCV**, and **ByteTrack**, the system detects objects across video streams and image collections, associates detections across consecutive frames to maintain persistent object IDs, and evaluates performance using IoU, Precision, Recall, F1-score, mAP@50, and mAP@50:95.

---

## ✨ Features

* **Real-time Detection:** Powered by Ultralytics YOLOv8 with custom training and inference pipelines.
* **Persistent Object Tracking:** Uses ByteTrack algorithm for stable multi-object tracking across video frames with unique IDs.
* **Custom Dataset Support:** Configured for the Roboflow People Detection benchmark (17,401 images, 145K+ annotations).
* **Comprehensive Metrics:** Custom vectorized computation of IoU, Precision, Recall, F1, mAP@50, and mAP@50:95.
* **Automated Visualizations:** Scripts to plot training curves, confusion matrices, PR curves, and class distributions.
* **Thorough Test Suite:** 20 unit tests covering conversions, metrics, tracking state, and palette generation.

---

## 🏗 System Architecture

```text
                 ┌───────────────────────────┐
                 │  Input Image / Video / Cam│
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │  OpenCV Frame Processing  │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │       YOLOv8 Detector     │
                 └─────────────┬─────────────┘
                               │ Bounding Boxes, Class, Conf
                               ▼
                 ┌───────────────────────────┐
                 │   Confidence / NMS Filter │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │     ByteTrack Tracker     │
                 │   (Frame Association)     │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │  Visualization & Track IDs│
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │   Output Video / Metrics  │
                 └───────────────────────────┘
```

---

## 📁 Project Structure

```text
object-detection-yolo/
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git exclusion rules
├── LICENSE                            # MIT License
│
├── configs/
│   └── data.yaml                      # YOLO dataset configuration
│
├── data/                              # Dataset directories (preserved via .gitkeep)
│   ├── raw/                           # Raw input images and videos
│   ├── processed/                     # Preprocessed images and labels
│   └── dataset/                       # Train / validation / test splits
│
├── models/
│   └── README.md                      # Model checkpoint documentation
│
├── src/
│   ├── __init__.py                    # Package initialization (v1.0.0)
│   ├── train.py                       # YOLOv8 fine-tuning CLI
│   ├── detect.py                      # Modular detection CLI
│   ├── track.py                       # Video tracking CLI with persistent IDs
│   ├── evaluate.py                    # Evaluation CLI (IoU, Precision, Recall, mAP)
│   ├── inference.py                   # Unified CLI entry point
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── dataset_utils.py           # Parsing, validation, splitting, synthetic data
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py                 # ObjectTracker wrapper around ByteTrack
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                 # Vectorized IoU, PR curve, mAP@50, mAP@50:95
│       └── visualization.py           # Drawing bounding boxes, track IDs, charts
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   # Dataset stats & class distribution
│   ├── 02_model_training.ipynb        # Training walkthrough & loss curves
│   └── 03_model_evaluation.ipynb      # Precision-Recall & IoU analysis
│
├── outputs/
│   ├── detections/                    # Detection outputs
│   ├── tracking/                      # Annotated tracking video results
│   ├── metrics/                       # Evaluation reports and CSVs
│   └── plots/                         # Generated curves and plots
│
└── tests/
    ├── test_detection.py              # Box format and label conversion tests
    ├── test_metrics.py                # IoU, precision, recall, and mAP tests
    └── test_tracking.py               # Tracker state, reset, and palette tests
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rraghavg/Object_Detection_System.git
   cd Object_Detection_System
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify tests:**
   ```bash
   pytest tests/ -v
   ```

---

## 🚀 Usage Guide

### 1. Pretrained Baseline Detection
Test the out-of-the-box YOLOv8n detector on any image or folder:
```bash
python src/detect.py --source data/raw/images --model yolov8n.pt --output outputs/detections --conf 0.25
```

### 2. Model Training
Fine-tune YOLOv8 on your custom dataset configured in `configs/data.yaml`:
```bash
python src/train.py \
    --data configs/data.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --imgsz 640 \
    --batch 16 \
    --optimizer AdamW \
    --name yolov8n_custom
```

### 3. Multi-Object Tracking
Track objects in a video with ByteTrack frame association and persistent IDs:
```bash
python src/track.py \
    --source data/raw/videos/test.mp4 \
    --model models/best.pt \
    --output outputs/tracking/result.mp4 \
    --conf 0.25
```

### 4. Model Evaluation
Compute quantitative metrics against test set ground-truth annotations:
```bash
python src/evaluate.py \
    --data configs/data.yaml \
    --model models/best.pt \
    --split test \
    --conf 0.25 \
    --iou 0.5
```

### 5. Unified Inference Runner
Single-command interface for both detection and tracking:
```bash
# Detection mode
python src/inference.py --source data/sample.jpg --model models/best.pt --mode detect

# Tracking mode
python src/inference.py --source data/video.mp4 --model models/best.pt --output outputs/tracking/result.mp4 --mode track
```

---

## 📦 Dataset

The repository supports standard YOLO format datasets:
```text
class_id x_center y_center width height  (normalized 0-1)
```

The system is configured with the **People Detection (v12i)** benchmark from Roboflow Universe:
* **Total Images:** 17,401 (15,210 train, 1,431 valid, 760 test)
* **Total Annotations:** 145,885 bounding boxes
* **Average Density:** 8.4 objects per image

---

## 🧪 Testing

The test suite covers algorithmic building blocks without requiring GPU or heavyweight model downloads:
```bash
pytest tests/ -v
```

```text
tests/test_detection.py::test_yolo_to_xyxy_conversion PASSED
tests/test_detection.py::test_xyxy_to_yolo_conversion PASSED
tests/test_detection.py::test_roundtrip_conversion PASSED
tests/test_detection.py::test_load_yolo_labels PASSED
tests/test_detection.py::test_load_yolo_labels_empty PASSED
tests/test_detection.py::test_detection_output_format PASSED
tests/test_metrics.py::test_iou_complete_overlap PASSED
tests/test_metrics.py::test_iou_no_overlap PASSED
tests/test_metrics.py::test_iou_partial_overlap PASSED
tests/test_metrics.py::test_iou_matrix PASSED
tests/test_metrics.py::test_precision_recall_perfect PASSED
tests/test_metrics.py::test_precision_recall_no_detections PASSED
tests/test_metrics.py::test_precision_recall_false_positives PASSED
tests/test_metrics.py::test_precision_recall_missed_detections PASSED
tests/test_metrics.py::test_compute_ap PASSED
tests/test_metrics.py::test_compute_map PASSED
tests/test_tracking.py::test_tracker_initialization PASSED
tests/test_tracking.py::test_track_summary_structure PASSED
tests/test_tracking.py::test_tracker_reset PASSED
tests/test_tracking.py::test_color_palette_generation PASSED

============================== 20 passed in 6.27s ==============================
```

---

## 💡 Key Computer Vision Concepts

* **Single-Stage Detection (YOLO):** Predicts bounding boxes and class probabilities directly in a single forward pass, providing high FPS for real-time applications.
* **Intersection over Union (IoU):** Evaluates overlap quality between predicted bounding box \(B_p\) and ground-truth \(B_{gt}\):
  \[\text{IoU} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}\]
* **Detection vs. Tracking:** Detection locates objects independently in each frame; tracking associates detections temporally using motion dynamics and appearance to assign persistent IDs.
* **ByteTrack:** Maintains low-confidence detections rather than discarding them, utilizing both high and low similarity associations to recover tracked objects through occlusion and motion blur.

---

## 🛠 Technologies Used

* **Python 3.8+**
* **Ultralytics (YOLOv8)**
* **OpenCV**
* **ByteTrack / lapx**
* **PyTorch / Torchvision**
* **NumPy / Pandas / Matplotlib / Seaborn / Scikit-learn**
* **Pytest**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Raghav Gupta**
