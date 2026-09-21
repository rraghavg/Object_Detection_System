"""Unit tests for the metrics module (src/utils/metrics.py)."""
import numpy as np
import pytest
from src.utils.metrics import (
    compute_iou,
    compute_iou_matrix,
    compute_precision_recall,
    compute_ap,
    compute_map,
)


def test_iou_complete_overlap():
    """Two identical boxes -> IoU = 1.0"""
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert compute_iou(box1, box2) == 1.0


def test_iou_no_overlap():
    """Two non-overlapping boxes -> IoU = 0.0"""
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert compute_iou(box1, box2) == 0.0


def test_iou_partial_overlap():
    """Known partial overlap -> verify exact IoU value."""
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    # intersection area: 5*5 = 25. Union area: 100 + 100 - 25 = 175
    # IoU = 25 / 175 ≈ 0.1429
    iou = compute_iou(box1, box2)
    assert pytest.approx(iou, abs=0.01) == 25.0 / 175.0


def test_iou_matrix():
    """Test vectorized IoU with multiple box pairs."""
    boxes1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    boxes2 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    iou_mat = compute_iou_matrix(boxes1, boxes2)
    assert iou_mat.shape == (2, 2)
    assert iou_mat[0, 0] == 1.0
    assert iou_mat[1, 0] == 0.0
    assert pytest.approx(iou_mat[0, 1], abs=0.01) == 25.0 / 175.0


def test_precision_recall_perfect():
    """All predictions match ground truth -> precision=1, recall=1."""
    predictions = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0, 'confidence': 0.9},
        {'bbox': [20, 20, 30, 30], 'class_id': 1, 'confidence': 0.8},
    ]
    ground_truths = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0},
        {'bbox': [20, 20, 30, 30], 'class_id': 1},
    ]
    result = compute_precision_recall(predictions, ground_truths, iou_threshold=0.5)
    assert result['precision'] == 1.0
    assert result['recall'] == 1.0


def test_precision_recall_no_detections():
    """No predictions -> precision=0, recall=0."""
    predictions = []
    ground_truths = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0},
        {'bbox': [20, 20, 30, 30], 'class_id': 1},
    ]
    result = compute_precision_recall(predictions, ground_truths, iou_threshold=0.5)
    assert result['precision'] == 0.0
    assert result['recall'] == 0.0


def test_precision_recall_false_positives():
    """Extra predictions -> precision < 1, recall = 1."""
    predictions = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0, 'confidence': 0.9},
        {'bbox': [50, 50, 60, 60], 'class_id': 0, 'confidence': 0.7},  # false positive
    ]
    ground_truths = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0},
    ]
    result = compute_precision_recall(predictions, ground_truths, iou_threshold=0.5)
    assert result['precision'] == 0.5
    assert result['recall'] == 1.0


def test_precision_recall_missed_detections():
    """Missed objects -> precision = 1, recall < 1."""
    predictions = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0, 'confidence': 0.9},
    ]
    ground_truths = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0},
        {'bbox': [20, 20, 30, 30], 'class_id': 0},
    ]
    result = compute_precision_recall(predictions, ground_truths, iou_threshold=0.5)
    assert result['precision'] == 1.0
    assert result['recall'] == 0.5


def test_compute_ap():
    """Test AP calculation with known PR curve values."""
    recalls = np.array([0.0, 0.5, 0.5, 1.0])
    precisions = np.array([1.0, 1.0, 0.5, 0.0])
    ap = compute_ap(recalls, precisions)
    assert ap > 0.0


def test_compute_map():
    """Test mAP with simple multi-class scenario."""
    # Create predictions and ground truths that perfectly match
    predictions = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0, 'confidence': 0.9},
        {'bbox': [20, 20, 30, 30], 'class_id': 1, 'confidence': 0.8},
    ]
    ground_truths = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0},
        {'bbox': [20, 20, 30, 30], 'class_id': 1},
    ]
    result = compute_map(predictions, ground_truths)
    assert 'mAP@50' in result
    assert 'mAP@50-95' in result
    # Perfect matches should yield high mAP
    assert result['mAP@50'] > 0.0
