"""
Dataset utilities for YOLO project.
"""
from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple
import cv2
import numpy as np

def load_yolo_labels(label_path: Path) -> List[Dict]:
    """
    Parse YOLO format label file.
    
    Args:
        label_path: Path to the label file.
        
    Returns:
        List of dictionaries containing class_id, x_center, y_center, width, height.
    """
    labels = []
    if not label_path.exists():
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append({
                    'class_id': int(parts[0]),
                    'x_center': float(parts[1]),
                    'y_center': float(parts[2]),
                    'width': float(parts[3]),
                    'height': float(parts[4])
                })
    return labels

def validate_dataset(dataset_dir: Path) -> Dict:
    """
    Check dataset integrity.
    
    Args:
        dataset_dir: Path to the dataset root directory (containing images/ and labels/ subdirectories).
        
    Returns:
        Dictionary containing validation statistics.
    """
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    
    stats = {
        'total_images': 0,
        'total_labels': 0,
        'missing_labels': 0,
        'missing_images': 0,
        'class_distribution': {}
    }
    
    if not images_dir.exists() or not labels_dir.exists():
        return stats
        
    image_files = set(f.stem for f in images_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    label_files = set(f.stem for f in labels_dir.glob('*.txt'))
    
    stats['total_images'] = len(image_files)
    stats['total_labels'] = len(label_files)
    stats['missing_labels'] = len(image_files - label_files)
    stats['missing_images'] = len(label_files - image_files)
    
    for label_file in labels_dir.glob('*.txt'):
        labels = load_yolo_labels(label_file)
        for label in labels:
            cid = label['class_id']
            stats['class_distribution'][cid] = stats['class_distribution'].get(cid, 0) + 1
            
    return stats

def split_dataset(source_images: Path, source_labels: Path, output_dir: Path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test.
    """
    random.seed(seed)
    
    images = [f for f in source_images.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    
    for split, img_list in splits.items():
        split_img_dir = output_dir / 'images' / split
        split_lbl_dir = output_dir / 'labels' / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in img_list:
            lbl_path = source_labels / f"{img_path.stem}.txt"
            shutil.copy(img_path, split_img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, split_lbl_dir / lbl_path.name)

def yolo_to_xyxy(box: Dict, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized format to absolute xyxy coordinates."""
    x_center = box['x_center'] * img_w
    y_center = box['y_center'] * img_h
    width = box['width'] * img_w
    height = box['height'] * img_h
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def xyxy_to_yolo(box: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert absolute xyxy to YOLO normalized format."""
    x1, y1, x2, y2 = box
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    x_center = (x1 + (x2 - x1) / 2) / img_w
    y_center = (y1 + (y2 - y1) / 2) / img_h
    
    return x_center, y_center, width, height

def get_dataset_stats(dataset_dir: Path) -> Dict:
    """Return statistics (num images, class distribution, avg boxes per image)."""
    stats = validate_dataset(dataset_dir)
    total_boxes = sum(stats['class_distribution'].values())
    stats['avg_boxes_per_image'] = total_boxes / stats['total_images'] if stats['total_images'] > 0 else 0
    return stats

def generate_sample_dataset(output_dir: Path, num_images=50, img_size=(640, 640), classes=None):
    """
    Generate synthetic colored rectangle dataset for testing the full pipeline.
    """
    if classes is None:
        classes = [0, 1, 2, 3] # person, car, bicycle, motorcycle
        
    img_dir = output_dir / 'images'
    lbl_dir = output_dir / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    
    for i in range(num_images):
        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        img[:] = np.random.randint(50, 100, (3,))
        
        num_boxes = random.randint(1, 5)
        labels = []
        for _ in range(num_boxes):
            cid = random.choice(classes)
            w = random.randint(50, 200)
            h = random.randint(50, 200)
            x1 = random.randint(0, img_size[0] - w)
            y1 = random.randint(0, img_size[1] - h)
            x2 = x1 + w
            y2 = y1 + h
            
            color = colors[cid % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            x_center, y_center, yolo_w, yolo_h = xyxy_to_yolo((x1, y1, x2, y2), img_size[0], img_size[1])
            labels.append(f"{cid} {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
            
        cv2.imwrite(str(img_dir / f"sample_{i:03d}.jpg"), img)
        with open(lbl_dir / f"sample_{i:03d}.txt", 'w') as f:
            f.writelines(labels)

if __name__ == '__main__':
    base_dir = Path("sample_dataset")
    print(f"Generating sample dataset in {base_dir}")
    generate_sample_dataset(base_dir)
