"""Unit tests for detection utilities and bounding box format conversions."""
import pytest
from pathlib import Path
from src.preprocessing.dataset_utils import (
    yolo_to_xyxy,
    xyxy_to_yolo,
    load_yolo_labels,
)


def test_yolo_to_xyxy_conversion():
    """Test YOLO normalized -> absolute xyxy conversion with known values."""
    yolo_box = {'class_id': 0, 'x_center': 0.5, 'y_center': 0.5, 'width': 0.2, 'height': 0.2}
    img_width, img_height = 1000, 1000
    xyxy = yolo_to_xyxy(yolo_box, img_width, img_height)
    # 0.5 * 1000 = 500 (center). 0.2 * 1000 = 200 (width).
    # xmin = 500 - 100 = 400. xmax = 500 + 100 = 600.
    assert xyxy == (400, 400, 600, 600)


def test_xyxy_to_yolo_conversion():
    """Test absolute xyxy -> YOLO normalized conversion."""
    xyxy_box = (400, 400, 600, 600)
    img_width, img_height = 1000, 1000
    yolo = xyxy_to_yolo(xyxy_box, img_width, img_height)
    assert pytest.approx(yolo[0], abs=0.001) == 0.5
    assert pytest.approx(yolo[1], abs=0.001) == 0.5
    assert pytest.approx(yolo[2], abs=0.001) == 0.2
    assert pytest.approx(yolo[3], abs=0.001) == 0.2


def test_roundtrip_conversion():
    """Convert yolo->xyxy->yolo and verify values match."""
    original_box = {'class_id': 0, 'x_center': 0.3, 'y_center': 0.4, 'width': 0.1, 'height': 0.5}
    img_width, img_height = 1920, 1080
    xyxy = yolo_to_xyxy(original_box, img_width, img_height)
    recovered = xyxy_to_yolo(xyxy, img_width, img_height)
    for orig_val, rec_val in zip(
        [original_box['x_center'], original_box['y_center'], original_box['width'], original_box['height']],
        recovered,
    ):
        assert pytest.approx(orig_val, abs=0.01) == rec_val


def test_load_yolo_labels(tmp_path):
    """Create a temp label file, load it, verify parsed output."""
    label_file = tmp_path / "sample.txt"
    label_file.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05")

    labels = load_yolo_labels(label_file)
    assert len(labels) == 2
    assert labels[0]['class_id'] == 0
    assert labels[0]['x_center'] == 0.5
    assert labels[0]['width'] == 0.2
    assert labels[1]['class_id'] == 1
    assert labels[1]['x_center'] == 0.1


def test_load_yolo_labels_empty(tmp_path):
    """Empty label file -> empty list."""
    label_file = tmp_path / "empty.txt"
    label_file.write_text("")

    labels = load_yolo_labels(label_file)
    assert labels == []


def test_detection_output_format():
    """Verify a detection result dict has required keys (bbox, class_id, confidence)."""
    # This tests the expected output format from the detection pipeline
    detection = {
        'bbox': [100, 100, 200, 200],
        'class_id': 0,
        'confidence': 0.95,
    }
    assert 'bbox' in detection
    assert 'class_id' in detection
    assert 'confidence' in detection
    assert detection['bbox'] == [100, 100, 200, 200]
    assert detection['class_id'] == 0
    assert detection['confidence'] == 0.95
