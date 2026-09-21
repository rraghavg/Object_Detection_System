"""
Unified inference wrapper for running either detection or tracking.

Example usage:
    python src/inference.py --source data/raw/videos/test.mp4 --model models/best.pt --output outputs/tracking/result.mp4 --mode track
"""

import argparse
import logging
import sys
from pathlib import Path

# Import functionality from detect and track modules
try:
    import detect
    import track
except ImportError:
    # Handle if run from outside src directory
    sys.path.append(str(Path(__file__).parent))
    import detect
    import track

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified YOLO inference script.")
    parser.add_argument('--source', type=str, required=True, help='Input path (image/video/dir)')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Model path')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--mode', type=str, choices=['detect', 'track'], default='detect', help='Inference mode: detect or track')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Display results flag')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting unified inference in '{args.mode}' mode with args: {args}")
    
    # We map the wrapper args to the specific module's args format
    # Creating a namespace mock for the underlying scripts
    if args.mode == 'detect':
        detect_args = argparse.Namespace(
            source=args.source,
            model=args.model,
            conf=args.conf,
            iou=0.45,
            output=args.output,
            save=True,
            show=args.show,
            imgsz=640
        )
        # Patch sys.argv or just pass args if we decoupled argparse in those modules.
        # For simplicity, we overwrite sys.argv
        sys.argv = ['detect.py', '--source', args.source, '--model', args.model, '--conf', str(args.conf), '--output', args.output, '--save']
        if args.show:
            sys.argv.append('--show')
        detect.main()
        
    elif args.mode == 'track':
        sys.argv = ['track.py', '--source', args.source, '--model', args.model, '--conf', str(args.conf), '--output', args.output]
        if args.show:
            sys.argv.append('--show')
        track.main()

if __name__ == "__main__":
    main()
