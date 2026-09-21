# 🚀 Real-Time Object Detection & Tracking System

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-supported-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-purple)

## 📋 Overview

A robust, real-time object detection and tracking system built with YOLOv8 and OpenCV. This project focuses on high-performance inference, capable of tracking objects across video streams with advanced metrics and tracking algorithms (ByteTrack). 

## ✨ Features

* **Real-time Detection:** Powered by ultralytics YOLOv8 for blazing-fast inference.
* **Robust Tracking:** Uses ByteTrack algorithm for stable multi-object tracking across frames.
* **Custom Dataset Support:** Easily retrain the model on custom datasets.
* **Performance Metrics:** Built-in evaluation scripts to compute mAP, Precision, Recall, and IoU.
* **Extensible Architecture:** Modular codebase to swap out models, tracking algorithms, or datasets.

## 🏗 System Architecture

```text
+-------------------+       +--------------------+       +---------------------+
|   Video/Image     | ----> |   YOLOv8 Model     | ----> | ByteTrack Tracker   |
|   Input Stream    |       |   (Detection)      |       | (ID Assignment)     |
+-------------------+       +--------------------+       +---------------------+
                                                                |
                                                                v
                                                         +---------------------+
                                                         |  Output Rendering   |
                                                         |  (BBoxes & IDs)     |
                                                         +---------------------+
```

## 📦 Dataset

The system is configured to work with a custom dataset focusing on urban traffic or pedestrian scenarios:
* **Classes:** person (0), car (1), bicycle (2), motorcycle (3)
* **Size:** 500-2000 images
* **Format:** Standard YOLO format (`class_id x_center y_center width height` normalized 0-1)

## 🧠 Model

We utilize **YOLOv8n** (Nano) or **YOLOv8s** (Small) for optimal real-time performance. Transfer learning is employed to fine-tune pre-trained COCO weights onto our specific dataset.

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rraghavg/Object_Detection_System.git
   cd Object_Detection_System
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```text
YOLO Project/
├── data/                  # Dataset (images and YOLO labels)
├── models/                # Saved model weights (.pt files)
├── src/                   # Source code
│   ├── utils/             # Detection & metrics utilities
│   ├── tracking/          # Tracking modules (ByteTrack config)
│   └── inference.py       # Main inference script
├── tests/                 # Unit tests (pytest)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🚀 Quick Start

### Training

To fine-tune the YOLOv8 model on your custom dataset:
```bash
yolo task=detect mode=train model=yolov8n.pt data=data/dataset.yaml epochs=50 imgsz=640
```

### Detection

Run basic object detection on an image or video:
```bash
python src/inference.py --source data/sample.mp4 --weights models/best.pt --conf 0.5
```

### Tracking

Enable tracking to assign unique IDs to detected objects across frames:
```bash
python src/inference.py --source data/sample.mp4 --weights models/best.pt --track
```

### Evaluation

Evaluate the model to get performance metrics:
```bash
python -m pytest tests/
yolo task=detect mode=val model=models/best.pt data=data/dataset.yaml
```

## 📊 Results

*(Placeholder: Add your training curves, confusion matrices, and final metrics here)*

## 📈 Performance Metrics

| Model | mAP@50 | mAP@50-95 | Precision | Recall | FPS (CPU) | FPS (GPU) |
|-------|--------|-----------|-----------|--------|-----------|-----------|
| YOLOv8n | - | - | - | - | - | - |
| YOLOv8s | - | - | - | - | - | - |

## 🖼 Sample Outputs

*(Placeholder: Add sample GIFs or images showing detection and tracking results)*

## 🛠 Technologies Used

* **Python 3.8+**: Core programming language.
* **Ultralytics (YOLOv8)**: State-of-the-art object detection framework.
* **OpenCV**: Computer vision library for image/video I/O and processing.
* **ByteTrack**: Multi-object tracking algorithm integrated into Ultralytics.
* **Pytest**: Framework for writing unit tests.
* **NumPy/Pandas/Scikit-learn**: Data processing and metrics computation.

## 🔮 Future Improvements

* Integrate deeply with DeepSORT for appearance-based tracking.
* Add support for exporting models to ONNX and TensorRT for edge deployment.
* Implement a WebUI (using Streamlit or Gradio) for easier interactions.
* Support additional classes (e.g., buses, traffic lights).

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Raghav Gupta**

