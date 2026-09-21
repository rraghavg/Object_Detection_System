"""
Evaluation script for YOLO model to compute metrics on a dataset split.

Example usage:
    python src/evaluate.py --data configs/data.yaml --model yolov8n.pt --split test
"""

import argparse
import logging
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO model.")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--split', type=str, default='test', help='Dataset split (test or val)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--output', type=str, default='outputs/metrics', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting evaluation with args: {args}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        model = YOLO(args.model)
        
        # In ultralytics, val() computes metrics. It can be run on val or test splits.
        split_arg = args.split if args.split in ['val', 'test'] else 'val'
        metrics = model.val(data=args.data, split=split_arg, conf=args.conf, iou=args.iou, project=str(output_dir), name='eval_run')
        
        logger.info("Evaluation completed. Processing metrics.")
        
        # Accessing metrics
        results_dict = metrics.results_dict
        names = model.names
        
        # Ultralytics metrics usually have class-wise values if requested, but results_dict gives overall metrics easily.
        # Constructing table
        table_data = []
        for i, class_id in enumerate(metrics.ap_class_index):
            class_name = names[class_id]
            # Ultralytics metrics.box contains P, R, mAP50, mAP50-95 per class in ap_class_index order
            precision = metrics.box.mp[i] if hasattr(metrics.box, 'mp') else metrics.box.p[i]
            recall = metrics.box.mr[i] if hasattr(metrics.box, 'mr') else metrics.box.r[i]
            map50 = metrics.box.map50[i]
            map_all = metrics.box.map[i]
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            table_data.append({
                'Class': class_name,
                'Precision': f"{precision:.2f}",
                'Recall': f"{recall:.2f}",
                'F1': f"{f1:.2f}",
                'mAP@50': f"{map50:.2f}",
                'mAP@50:95': f"{map_all:.2f}"
            })
            
        overall_p = metrics.box.mp
        overall_r = metrics.box.mr
        overall_map50 = metrics.box.map50.mean() if hasattr(metrics.box.map50, 'mean') else metrics.box.map50
        overall_map_all = metrics.box.map.mean() if hasattr(metrics.box.map, 'mean') else metrics.box.map
        overall_f1 = 2 * (overall_p * overall_r) / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
        
        table_data.append({
            'Class': 'Overall',
            'Precision': f"{overall_p:.2f}",
            'Recall': f"{overall_r:.2f}",
            'F1': f"{overall_f1:.2f}",
            'mAP@50': f"{overall_map50:.2f}",
            'mAP@50:95': f"{overall_map_all:.2f}"
        })
        
        df = pd.DataFrame(table_data)
        csv_path = output_dir / 'metrics_report.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Metrics saved to {csv_path}")
        print("\nEvaluation Metrics:")
        print(df.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
