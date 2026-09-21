"""
Metrics module for YOLO object detection.
"""
from typing import List, Dict, Union, Tuple
from pathlib import Path
import numpy as np

def compute_iou(box1: Union[List, Tuple], box2: Union[List, Tuple]) -> float:
    """
    Compute IoU between two boxes in xyxy format.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU computation between two sets of boxes.
    
    Args:
        boxes1: Array of shape (N, 4) in xyxy format
        boxes2: Array of shape (M, 4) in xyxy format
        
    Returns:
        Array of shape (N, M) containing IoU values
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
        
    x1 = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3])
    
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area1[:, np.newaxis] + area2 - inter
    return np.where(union > 0, inter / union, 0.0)

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using all-point interpolation.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        
    Returns:
        Average Precision value
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_precision_recall(predictions: List[Dict], ground_truths: List[Dict], iou_threshold=0.5) -> Dict:
    """
    Compute TP, FP, FN, precision, recall, F1.
    """
    preds_sorted = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
    
    tp = 0
    fp = 0
    matched_gt = set()
    
    for pred in preds_sorted:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truths):
            if i in matched_gt or gt['class_id'] != pred['class_id']:
                continue
            
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
            
    fn = len(ground_truths) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_map(predictions: List[Dict], ground_truths: List[Dict], iou_thresholds=None) -> Dict:
    """Compute mAP@50 and mAP@50:95."""
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        
    classes = set(gt['class_id'] for gt in ground_truths)
    
    aps_per_class = {c: [] for c in classes}
    
    for c in classes:
        cls_preds = [p for p in predictions if p['class_id'] == c]
        cls_gts = [g for g in ground_truths if g['class_id'] == c]
        
        for iou_thresh in iou_thresholds:
            metrics = compute_precision_recall(cls_preds, cls_gts, iou_thresh)
            aps_per_class[c].append(metrics['precision']) 
            
    map50 = np.mean([aps[0] for aps in aps_per_class.values()]) if aps_per_class else 0.0
    map50_95 = np.mean([np.mean(aps) for aps in aps_per_class.values()]) if aps_per_class else 0.0
    
    return {
        'mAP@50': map50,
        'mAP@50-95': map50_95
    }

def generate_evaluation_report(results: Dict, output_dir: Path):
    """Save a formatted text/CSV evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.txt"
    
    with open(report_path, "w") as f:
        f.write("YOLO Evaluation Report\n")
        f.write("="*22 + "\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
