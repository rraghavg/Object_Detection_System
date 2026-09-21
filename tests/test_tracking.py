"""Unit tests for tracking utilities."""
import sys
import pytest
from unittest.mock import patch, MagicMock
from src.utils.visualization import generate_color_palette


@pytest.fixture(autouse=True)
def mock_ultralytics():
    """Mock ultralytics before importing tracker module."""
    mock_yolo_class = MagicMock()
    mock_module = MagicMock()
    mock_module.YOLO = mock_yolo_class
    sys.modules['ultralytics'] = mock_module
    
    # Force re-import of tracker module with mocked ultralytics
    if 'src.tracking.tracker' in sys.modules:
        del sys.modules['src.tracking.tracker']
    
    yield mock_yolo_class
    
    # Cleanup
    if 'ultralytics' in sys.modules:
        del sys.modules['ultralytics']
    if 'src.tracking.tracker' in sys.modules:
        del sys.modules['src.tracking.tracker']


def test_tracker_initialization(mock_ultralytics):
    """ObjectTracker initializes without error (mock the YOLO model)."""
    from src.tracking.tracker import ObjectTracker
    tracker = ObjectTracker(model_path="dummy.pt")

    assert tracker is not None
    assert tracker.model is mock_ultralytics.return_value


def test_track_summary_structure(mock_ultralytics):
    """Verify get_track_summary returns expected keys."""
    from src.tracking.tracker import ObjectTracker
    tracker = ObjectTracker(model_path="dummy.pt")
    # Simulate some tracked objects
    tracker.unique_ids = {1, 2, 3}
    tracker.track_durations = {1: 10, 2: 5, 3: 8}

    summary = tracker.get_track_summary()
    assert 'unique_objects_count' in summary
    assert 'track_durations_frames' in summary
    assert summary['unique_objects_count'] == 3


def test_tracker_reset(mock_ultralytics):
    """After reset, internal state is cleared."""
    from src.tracking.tracker import ObjectTracker
    tracker = ObjectTracker(model_path="dummy.pt")
    tracker.unique_ids = {1, 2}
    tracker.track_durations = {1: 10, 2: 5}

    tracker.reset()

    assert len(tracker.unique_ids) == 0
    assert len(tracker.track_durations) == 0


def test_color_palette_generation():
    """Test generate_color_palette returns correct number of distinct colors."""
    num_classes = 4
    palette = generate_color_palette(num_classes)

    assert len(palette) == num_classes
    # Ensure colors are 3-tuples (B, G, R) for OpenCV
    for color in palette:
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
