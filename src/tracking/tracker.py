"""
Tracker utilities wrapping ultralytics ByteTrack.
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: 'person', 1: 'car', 2: 'bicycle', 3: 'motorcycle'}

class ObjectTracker:
    """Wrapper around ultralytics ByteTrack tracking."""
    
    def __init__(self, model_path: str, tracker_type: str = 'bytetrack.yaml', conf: float = 0.25):
        self.model = YOLO(model_path)
        self.tracker_type = tracker_type
        self.conf = conf
        self.unique_ids = set()
        self.track_durations = {}
        
    def process_frame(self, frame) -> list[dict]:
        """
        Run detection+tracking on a single frame.
        
        Args:
            frame: OpenCV image array
            
        Returns:
            list[dict]: List of tracks in the frame containing track_id, bbox, class_id, class_name, confidence.
        """
        results = self.model.track(source=frame, conf=self.conf, persist=True, tracker=self.tracker_type, verbose=False)
        tracks = []
        
        for r in results:
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                
                for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                    cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
                    
                    tracks.append({
                        'track_id': track_id,
                        'bbox': box.tolist(),
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': float(conf)
                    })
                    
                    self.unique_ids.add(track_id)
                    self.track_durations[track_id] = self.track_durations.get(track_id, 0) + 1
                    
        return tracks

    def process_video(self, video_path: str, output_path: str = None, show: bool = False) -> dict:
        """
        Process entire video and return tracking summary.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return {}
            
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            tracks = self.process_frame(frame)
            
            # Simple drawing fallback if we need to output video
            if out or show:
                for t in tracks:
                    bbox = t['bbox']
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f"{t['class_name']} {t['track_id']}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if out:
                    out.write(frame)
                if show:
                    cv2.imshow('Tracking', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        if out:
            out.release()
        if show:
            cv2.destroyAllWindows()
            
        return self.get_track_summary()

    def get_track_summary(self) -> dict:
        """Return summary of all tracks."""
        return {
            'unique_objects_count': len(self.unique_ids),
            'track_durations_frames': self.track_durations
        }

    def reset(self):
        """Reset tracker state."""
        self.unique_ids.clear()
        self.track_durations.clear()
        # In ultralytics, persist state is tied to the YOLO model object state or kwargs.
        # We can reset the model object or instantiate a new one to cleanly reset ByteTrack internals.
        self.model = YOLO(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else self.model
