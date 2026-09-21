"""
Detection script for running inference on images or videos using a trained YOLOv8 model.

Example usage:
    python src/detect.py --source data/images --model yolov8n.pt --conf 0.25 --save --show
"""

import argparse
import logging
from pathlib import Path
import cv2
from ultralytics import YOLO
import sys

# Assume visualization is available in the project
try:
    from visualization import draw_detections
except ImportError:
    # Dummy fallback if visualization module is not created yet
    def draw_detections(image, detections):
        return image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: 'person', 1: 'car', 2: 'bicycle', 3: 'motorcycle'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection.")
    parser.add_argument('--source', type=str, required=True, help='Path to image, video, or directory')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--output', type=str, default='outputs/detections', help='Output directory')
    parser.add_argument('--save', action='store_true', help='Save results flag')
    parser.add_argument('--show', action='store_true', help='Display results flag')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    return parser.parse_args()

def process_image(model, source_path: Path, output_dir: Path, args: argparse.Namespace):
    img = cv2.imread(str(source_path))
    if img is None:
        logger.error(f"Could not read image: {source_path}")
        return
        
    results = model.predict(source=img, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
    
    detections = []
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    total_conf = 0.0
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
            if cls_name in class_counts:
                class_counts[cls_name] += 1
                
            total_conf += conf
            detections.append({'bbox': xyxy, 'class_id': cls_id, 'class_name': cls_name, 'confidence': conf})
            
    annotated_img = draw_detections(img.copy(), detections)
    
    if args.save:
        out_path = output_dir / source_path.name
        cv2.imwrite(str(out_path), annotated_img)
        logger.info(f"Saved detection result to {out_path}")
        
    if args.show:
        cv2.imshow('Detection', annotated_img)
        cv2.waitKey(0)
        
    avg_conf = total_conf / max(1, len(detections))
    logger.info(f"Image {source_path.name}: Detected {len(detections)} objects. Class counts: {class_counts}, Avg Conf: {avg_conf:.2f}")

def process_video(model, source_path: Path, output_dir: Path, args: argparse.Namespace):
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {source_path}")
        return
        
    out = None
    if args.save:
        out_path = output_dir / source_path.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        
    frame_count = 0
    total_objects = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(source=frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
                
                detections.append({'bbox': xyxy, 'class_id': cls_id, 'class_name': cls_name, 'confidence': conf})
                total_objects += 1
                
        annotated_img = draw_detections(frame, detections)
        
        if args.save and out:
            out.write(annotated_img)
            
        if args.show:
            cv2.imshow('Detection', annotated_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        frame_count += 1
        
    cap.release()
    if out:
        out.release()
    if args.show:
        cv2.destroyAllWindows()
        
    logger.info(f"Video {source_path.name}: Processed {frame_count} frames, {total_objects} total detections.")

def main():
    args = parse_args()
    logger.info(f"Starting detection with args: {args}")
    
    output_dir = Path(args.output)
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        
    try:
        model = YOLO(args.model)
        source_path = Path(args.source)
        
        if source_path.is_file():
            if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                process_video(model, source_path, output_dir, args)
            else:
                process_image(model, source_path, output_dir, args)
        elif source_path.is_dir():
            for file_path in source_path.iterdir():
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    process_image(model, file_path, output_dir, args)
        else:
            logger.error(f"Source not found: {source_path}")
            
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise

if __name__ == "__main__":
    main()
