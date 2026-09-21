"""
Training script for YOLOv8 object detection model.

Example usage:
    python src/train.py --data configs/data.yaml --model yolov8n.pt --epochs 50 --batch 16
"""

import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 model.")
    parser.add_argument('--data', type=str, default='configs/data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pretrained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use')
    parser.add_argument('--project', type=str, default='runs/train', help='Output directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting training with arguments: {args}")
    
    try:
        # Load YOLO model
        model = YOLO(args.model)
        
        # Train model
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer=args.optimizer,
            project=args.project,
            name=args.name
        )
        logger.info("Training completed successfully.")
        
        # Determine best model path
        best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'
        if best_model_path.exists():
            logger.info(f"Best model saved at: {best_model_path}")
        else:
            logger.warning("Could not find best.pt in the expected location.")
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
