# Model Checkpoints

This directory stores trained and pretrained YOLO model weights.

## Checkpoint Files

- `best.pt`: Model checkpoint with the highest validation performance during training.
- `last.pt`: Most recent checkpoint saved at the end of the last training epoch.

## Pretrained Models

To download pretrained YOLOv8 weights from Ultralytics, run:

```bash
yolo detect download model=yolov8n.pt
```

You can replace `yolov8n.pt` with other variants such as `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, or `yolov8x.pt` depending on your model size and latency requirements.

> **Note**: `.pt` files and other binary model checkpoint artifacts are ignored by git due to their large file size. If you wish to version control model weights with Git, consider using [Git LFS](https://git-lfs.com/) (`git lfs track "*.pt"`).
