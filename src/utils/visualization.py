"""
Visualization module for YOLO project.
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_color_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    """Generate visually distinct BGR colors for each class."""
    colors = []
    for i in range(num_classes):
        hue = i * (180 / num_classes)
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))
    return colors

def draw_detections(image: np.ndarray, detections: List[Dict], class_names: Dict[int, str], colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """Draw bounding boxes with class names and confidence scores on image using OpenCV."""
    img_copy = image.copy()
    if colors is None:
        colors = generate_color_palette(len(class_names))
        
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cls_id = det['class_id']
        conf = det.get('confidence', 1.0)
        
        color = colors[cls_id % len(colors)]
        label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return img_copy

def draw_tracks(image: np.ndarray, tracks: List[Dict], class_names: Dict[int, str], colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """Draw tracking boxes with persistent IDs."""
    img_copy = image.copy()
    if colors is None:
        colors = generate_color_palette(len(class_names))
        
    for track in tracks:
        x1, y1, x2, y2 = map(int, track['bbox'])
        cls_id = track['class_id']
        track_id = track.get('track_id', -1)
        
        color = colors[cls_id % len(colors)]
        label = f"ID:{track_id} {class_names.get(cls_id, str(cls_id))}"
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return img_copy

def plot_training_curves(results_csv: Path, output_dir: Path):
    """Plot training/validation loss curves from ultralytics results.csv."""
    if not results_csv.exists():
        return
        
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    plt.figure(figsize=(10, 6))
    if 'train/box_loss' in df.columns:
        plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    if 'val/box_loss' in df.columns:
        plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()

def plot_precision_recall_curve(precisions: List[float], recalls: List[float], class_names: List[str], output_path: Path):
    """Plot PR curve per class."""
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.', label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path):
    """Plot confusion matrix heatmap using seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_class_distribution(class_counts: Dict[int, int], output_path: Path):
    """Plot bar chart of class distribution."""
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(classes)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_map_curve(map_values: List[float], iou_thresholds: List[float], output_path: Path):
    """Plot mAP vs IoU threshold."""
    plt.figure(figsize=(8, 6))
    plt.plot(iou_thresholds, map_values, marker='o', linestyle='-')
    plt.xlabel('IoU Threshold')
    plt.ylabel('mAP')
    plt.title('mAP vs IoU Threshold')
    plt.grid(True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
