"""
Tracking script for object tracking in videos using YOLOv8 and ByteTrack.

Example usage:
    python src/track.py --source data/videos/test.mp4 --model yolov8n.pt --show
"""

import argparse
import logging
from pathlib import Path
import time
import cv2
from ultralytics import YOLO

try:
    from visualization import draw_tracks
except ImportError:
    # Dummy fallback if visualization module is not created yet
    def draw_tracks(image, tracks):
        return image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: 'person', 1: 'car', 2: 'bicycle', 3: 'motorcycle'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 object tracking.")
    parser.add_argument('--source', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model path')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='Tracker config file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='outputs/tracking', help='Output directory')
    parser.add_argument('--show', action='store_true', help='Display results flag')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting tracking with args: {args}")
    
    try:
        model = YOLO(args.model)
        source_path = Path(args.source)
        
        if not source_path.exists() or not source_path.is_file():
            logger.error(f"Source video not found: {source_path}")
            return
            
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / source_path.name
        
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {source_path}")
            return
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        
        unique_ids = set()
        track_durations = {}
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.track(source=frame, conf=args.conf, persist=True, tracker=args.tracker, verbose=False)
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
                        
                        unique_ids.add(track_id)
                        track_durations[track_id] = track_durations.get(track_id, 0) + 1
                        
            annotated_frame = draw_tracks(frame, tracks)
            out.write(annotated_frame)
            
            if args.show:
                cv2.imshow('Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_count += 1
            
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()
            
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"Tracking completed. Output saved to {out_path}")
        logger.info(f"Processed {frame_count} frames at {avg_fps:.2f} FPS")
        logger.info(f"Total unique objects tracked: {len(unique_ids)}")
        
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        raise

if __name__ == "__main__":
    main()
