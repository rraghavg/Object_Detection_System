"""
Real-Time Object Detection & Tracking System - Unified Vercel & Web Entrypoint
Supports:
1. Interactive Image Upload & YOLOv8 Detection Interface
2. Vercel Serverless Function (BaseHTTPRequestHandler 'handler')
3. WSGI Web Application ('app', 'application')
4. Standalone local server (python main.py)
"""

import os
import json
import time
import base64
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs

# Class taxonomy from configs/data.yaml
CLASSES = [
    '0', '1', '2', 'Bicycle', 'Bike', 'Car', 'Cyclist', 'Pedestrian', 'Pedestrians',
    'Persona', 'Pessoa', 'Signboard', 'Stopper', 'aeroplane', 'bag', 'berdiri',
    'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'cyclist', 'dianzhuan', 'diningtable', 'dog', 'face', 'forklift', 'handbag',
    'head', 'helmet', 'high', 'horse', 'jatuh', 'laptop', 'low', 'medium',
    'motorbike', 'people', 'person', 'persons', 'pottedplant', 'refrigerator',
    'sheep', 'sofa', 'teddy bear', 'train', 'tv', 'tvmonitor', 'vase'
]

PROJECT_INFO = {
    "name": "Real-Time Object Detection & Tracking System",
    "version": "1.0.0",
    "author": "Raghav Gupta",
    "github": "https://github.com/rraghavg/Object_Detection_System",
    "architecture": "YOLOv8 + ByteTrack + OpenCV",
    "dataset": "Roboflow People Detection v12i (17,401 images, 145,885 annotations, 53 classes)",
    "metrics": {
        "evaluation_protocol": "IoU, Precision, Recall, F1-Score, mAP@50, mAP@50:95",
        "tests": "20/20 passed"
    }
}

# Lazy loader for YOLO model
_YOLO_MODEL = None

def get_yolo_model():
    """Lazily load YOLO model to avoid slow boot on simple GET requests."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            # Prefer custom fine-tuned weights if available, otherwise baseline yolov8n
            custom_weights = os.path.join(os.path.dirname(__file__), 'runs/detect/runs/train/yolov8n_people_detection/weights/best.pt')
            if os.path.exists(custom_weights):
                _YOLO_MODEL = YOLO(custom_weights)
            else:
                _YOLO_MODEL = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"YOLO model load warning: {e}")
            _YOLO_MODEL = None
    return _YOLO_MODEL


def run_yolo_detection(image_bytes: bytes, conf_threshold: float = 0.25):
    """
    Run YOLO inference on image bytes and return annotated image base64 + metadata.
    """
    t0 = time.time()
    try:
        import cv2
        import numpy as np
        
        # Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Invalid or unreadable image data"}

        model = get_yolo_model()
        if model is None:
            # Fallback when ultralytics/torch is not installed in the current environment
            return {
                "success": True,
                "engine": "fallback",
                "message": "Ultralytics/PyTorch engine not active in serverless container.",
                "total_detected": 0,
                "detections": [],
                "summary": [],
                "inference_time_ms": round((time.time() - t0) * 1000, 1),
                "annotated_image": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode('utf-8')
            }

        # Run inference
        results = model.predict(source=img, conf=conf_threshold, verbose=False)
        res = results[0]
        
        # Plot annotations onto frame
        annotated_bgr = res.plot()
        _, buffer = cv2.imencode('.jpg', annotated_bgr)
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        detections = []
        class_counts = {}

        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = model.names.get(cls_id, str(cls_id))
            xyxy = [int(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "class_name": cls_name,
                "confidence": round(conf * 100, 1),
                "bbox": xyxy
            })

            if cls_name not in class_counts:
                class_counts[cls_name] = {"count": 0, "conf_sum": 0.0}
            class_counts[cls_name]["count"] += 1
            class_counts[cls_name]["conf_sum"] += conf

        summary = [
            {
                "class": cls_name,
                "count": info["count"],
                "avg_confidence": f"{(info['conf_sum'] / info['count'] * 100):.1f}%"
            }
            for cls_name, info in sorted(class_counts.items(), key=lambda x: x[1]['count'], reverse=True)
        ]

        return {
            "success": True,
            "engine": "yolov8",
            "total_detected": len(detections),
            "detections": detections,
            "summary": summary,
            "inference_time_ms": round((time.time() - t0) * 1000, 1),
            "annotated_image": annotated_b64
        }
    except Exception as e:
        return {
            "error": str(e),
            "inference_time_ms": round((time.time() - t0) * 1000, 1)
        }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Object Detection & Tracking System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #121826;
            --bg-card: #1a2336;
            --border: #2a364f;
            --text-primary: #f0f4fc;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-blue-hover: #2563eb;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem 1rem;
        }
        .container { max-width: 1050px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 2rem; }
        .badge {
            display: inline-flex; align-items: center; gap: 0.5rem;
            background: rgba(16, 185, 129, 0.15); color: var(--accent-green);
            padding: 0.35rem 0.85rem; border-radius: 9999px;
            font-size: 0.85rem; font-weight: 600; margin-bottom: 1rem;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .pulse {
            width: 8px; height: 8px; background: var(--accent-green);
            border-radius: 50%; box-shadow: 0 0 8px var(--accent-green);
        }
        h1 {
            font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem;
            background: linear-gradient(135deg, #f0f4fc 0%, #94a3b8 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { color: var(--text-secondary); font-size: 1.1rem; max-width: 650px; margin: 0 auto; }
        .stats-grid {
            display: grid; grid-template-columns: repeat(4, 1fr);
            gap: 1rem; margin: 2rem 0;
        }
        @media (max-width: 640px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }
        .stat-card {
            background: var(--bg-secondary); border: 1px solid var(--border);
            border-radius: 10px; padding: 1.25rem 1rem; text-align: center;
        }
        .stat-val { font-size: 1.75rem; font-weight: 700; color: var(--accent-blue); font-family: 'Fira Code', monospace; }
        .stat-label { font-size: 0.8rem; text-transform: uppercase; color: var(--text-secondary); margin-top: 0.25rem; }

        /* Detection App Section */
        .detector-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            margin: 2.5rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .section-header h2 {
            font-size: 1.4rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .controls-row {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        .slider-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        .slider-group input[type="range"] {
            accent-color: var(--accent-blue);
            cursor: pointer;
        }
        .slider-val {
            font-family: 'Fira Code', monospace;
            color: var(--accent-blue);
            font-weight: 600;
            min-width: 40px;
        }
        .sample-buttons {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }
        .sample-btn {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .sample-btn:hover {
            border-color: var(--accent-blue);
            background: rgba(59, 130, 246, 0.1);
        }
        .dropzone {
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            background: var(--bg-secondary);
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 1.5rem;
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-blue);
            background: rgba(59, 130, 246, 0.05);
        }
        .dropzone svg {
            width: 48px;
            height: 48px;
            color: var(--accent-blue);
            margin-bottom: 0.75rem;
        }
        .dropzone-text {
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .dropzone-subtext {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        .btn-detect {
            background: var(--accent-blue);
            color: #fff;
            border: none;
            padding: 0.85rem 1.75rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: background 0.2s;
            width: 100%;
            justify-content: center;
        }
        .btn-detect:hover {
            background: var(--accent-blue-hover);
        }
        .btn-detect:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Results Display */
        .results-container {
            display: none;
            margin-top: 2rem;
            border-top: 1px solid var(--border);
            padding-top: 2rem;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        @media (max-width: 768px) {
            .comparison-grid { grid-template-columns: 1fr; }
        }
        .image-box {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }
        .image-box-header {
            padding: 0.65rem 1rem;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(0,0,0,0.3);
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
        }
        .image-box img {
            width: 100%;
            height: auto;
            display: block;
            max-height: 450px;
            object-fit: contain;
            background: #000;
        }

        /* Detections Table */
        .summary-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
        }
        .summary-card h3 {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .latency-badge {
            font-size: 0.8rem;
            font-family: 'Fira Code', monospace;
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }
        .summary-table th, .summary-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        .summary-table th {
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
        }
        .summary-table tr:last-child td {
            border-bottom: none;
        }
        .class-badge {
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .conf-text {
            font-family: 'Fira Code', monospace;
            color: var(--accent-blue);
            font-weight: 600;
        }

        /* Endpoints & Architecture */
        .grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.25rem; margin-bottom: 2rem;
        }
        .card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 12px; padding: 1.5rem;
        }
        .card h3 { font-size: 1.15rem; margin-bottom: 0.5rem; color: var(--text-primary); }
        .card p { color: var(--text-secondary); font-size: 0.95rem; }
        .endpoints {
            background: var(--bg-secondary); border: 1px solid var(--border);
            border-radius: 12px; overflow: hidden; margin-top: 1rem;
        }
        .ep-row {
            display: flex; align-items: center; justify-content: space-between;
            padding: 1rem 1.25rem; border-bottom: 1px solid var(--border);
            font-family: 'Fira Code', monospace; font-size: 0.9rem;
        }
        .ep-row:last-child { border-bottom: none; }
        .method {
            background: rgba(59, 130, 246, 0.2); color: var(--accent-blue);
            padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.75rem; margin-right: 0.75rem;
        }
        .ep-link { color: var(--accent-blue); text-decoration: none; }
        .ep-link:hover { text-decoration: underline; }
        .ep-desc { color: var(--text-secondary); font-family: 'Inter', sans-serif; font-size: 0.85rem; }
        footer {
            text-align: center; margin-top: 3rem; padding-top: 2rem;
            border-top: 1px solid var(--border); color: var(--text-secondary); font-size: 0.9rem;
        }
        footer a { color: var(--accent-blue); text-decoration: none; }
        .spinner {
            width: 18px; height: 18px; border: 2px solid #fff;
            border-top-color: transparent; border-radius: 50%;
            animation: spin 0.8s linear infinite; display: none;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge"><div class="pulse"></div> Live on Vercel</div>
            <h1>Real-Time Object Detection & Tracking</h1>
            <p class="subtitle">Computer vision system using YOLOv8 for frame-by-frame object detection and ByteTrack for persistent multi-object tracking across video frames.</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card"><div class="stat-val">17,401</div><div class="stat-label">Total Images</div></div>
            <div class="stat-card"><div class="stat-val">145K+</div><div class="stat-label">Annotations</div></div>
            <div class="stat-card"><div class="stat-val">53</div><div class="stat-label">Classes</div></div>
            <div class="stat-card"><div class="stat-val">20/20</div><div class="stat-label">Tests Passed</div></div>
        </div>

        <!-- Interactive Object Detection Section -->
        <section class="detector-section">
            <div class="section-header">
                <h2>🎯 Try Object Detection</h2>
                <div class="controls-row">
                    <div class="slider-group">
                        <label for="conf-slider">Confidence:</label>
                        <input type="range" id="conf-slider" min="0.10" max="0.90" step="0.05" value="0.25" oninput="updateConf(this.value)">
                        <span class="slider-val" id="conf-val">0.25</span>
                    </div>
                </div>
            </div>

            <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1rem;">
                Upload an image or select a sample image below to run YOLOv8 object detection in real-time:
            </p>

            <div class="sample-buttons">
                <span style="font-size: 0.85rem; color: var(--text-secondary); align-self: center; margin-right: 0.25rem;">Presets:</span>
                <button class="sample-btn" onclick="loadSample('street')">🚦 Street Crossing</button>
                <button class="sample-btn" onclick="loadSample('pedestrians')">🚶 Pedestrians</button>
                <button class="sample-btn" onclick="loadSample('traffic')">🚗 Urban Traffic</button>
            </div>

            <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                <div class="dropzone-text">Drop an image here or click to browse</div>
                <div class="dropzone-subtext">Supports JPG, PNG, WEBP</div>
                <input type="file" id="file-input" accept="image/*" style="display: none;" onchange="handleFile(this.files[0])">
            </div>

            <button class="btn-detect" id="btn-detect" onclick="runDetection()" disabled>
                <div class="spinner" id="spinner"></div>
                <span id="btn-text">Detect Objects</span>
            </button>

            <!-- Results Section -->
            <div class="results-container" id="results">
                <div class="comparison-grid">
                    <div class="image-box">
                        <div class="image-box-header">Original Image</div>
                        <img id="img-original" src="" alt="Original Image">
                    </div>
                    <div class="image-box">
                        <div class="image-box-header">Detection Result (YOLOv8)</div>
                        <img id="img-annotated" src="" alt="Annotated Result">
                    </div>
                </div>

                <div class="summary-card">
                    <h3>
                        <span>Detected Objects Summary</span>
                        <span class="latency-badge" id="latency-badge">0.0 ms</span>
                    </h3>
                    <div id="table-container">
                        <!-- Populated dynamically -->
                    </div>
                </div>
            </div>
        </section>

        <div class="grid">
            <div class="card">
                <h3>🔍 YOLOv8 Detection</h3>
                <p>High-FPS single-stage detector fine-tuned on the Roboflow People Detection benchmark.</p>
            </div>
            <div class="card">
                <h3>🎯 ByteTrack Tracking</h3>
                <p>Associates detections across consecutive frames to maintain persistent object IDs.</p>
            </div>
            <div class="card">
                <h3>📊 Metric Evaluation</h3>
                <p>Vectorized computation of IoU, Precision, Recall, F1, mAP@50, and mAP@50:95.</p>
            </div>
        </div>

        <h3 style="margin-top: 2rem;">⚡ Live API Endpoints</h3>
        <div class="endpoints">
            <div class="ep-row">
                <div><span class="method">POST</span><a class="ep-link" href="/api/detect">/api/detect</a></div>
                <div class="ep-desc">Run YOLOv8 inference on base64 image</div>
            </div>
            <div class="ep-row">
                <div><span class="method">GET</span><a class="ep-link" href="/api/health">/api/health</a></div>
                <div class="ep-desc">Service status & health check</div>
            </div>
            <div class="ep-row">
                <div><span class="method">GET</span><a class="ep-link" href="/api/info">/api/info</a></div>
                <div class="ep-desc">Architecture and model metadata</div>
            </div>
            <div class="ep-row">
                <div><span class="method">GET</span><a class="ep-link" href="/api/classes">/api/classes</a></div>
                <div class="ep-desc">53 detection class labels</div>
            </div>
            <div class="ep-row">
                <div><span class="method">GET</span><a class="ep-link" href="/api/stats">/api/stats</a></div>
                <div class="ep-desc">Dataset splits & distribution stats</div>
            </div>
        </div>

        <footer>
            Built by <strong>Raghav Gupta</strong> | <a href="https://github.com/rraghavg/Object_Detection_System" target="_blank">GitHub Repository</a>
        </footer>
    </div>

    <script>
        let currentImageBase64 = null;

        function updateConf(val) {
            document.getElementById('conf-val').innerText = parseFloat(val).toFixed(2);
        }

        const dropzone = document.getElementById('dropzone');
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });

        function handleFile(file) {
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImageBase64 = e.target.result;
                document.getElementById('img-original').src = currentImageBase64;
                document.getElementById('btn-detect').disabled = false;
                document.getElementById('results').style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        // Preset sample SVG images converted to canvas base64 for instant testing
        function loadSample(type) {
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            const ctx = canvas.getContext('2d');

            // Draw clean realistic background scene
            ctx.fillStyle = '#2c3e50';
            ctx.fillRect(0, 0, 640, 480);
            ctx.fillStyle = '#7f8c8d';
            ctx.fillRect(0, 260, 640, 220); // Road
            ctx.fillStyle = '#bdc3c7';
            ctx.fillRect(0, 240, 640, 20); // Sidewalk

            // Road markings
            ctx.fillStyle = '#f1c40f';
            ctx.fillRect(280, 280, 80, 15);
            ctx.fillRect(280, 340, 80, 15);
            ctx.fillRect(280, 400, 80, 15);

            if (type === 'street' || type === 'pedestrians') {
                // Draw pedestrian silhouettes
                ctx.fillStyle = '#e74c3c';
                ctx.fillRect(120, 200, 30, 80); // Person 1
                ctx.beginPath(); ctx.arc(135, 185, 14, 0, Math.PI*2); ctx.fill();

                ctx.fillStyle = '#3498db';
                ctx.fillRect(180, 210, 28, 75); // Person 2
                ctx.beginPath(); ctx.arc(194, 195, 13, 0, Math.PI*2); ctx.fill();

                ctx.fillStyle = '#2ecc71';
                ctx.fillRect(450, 190, 32, 85); // Person 3
                ctx.beginPath(); ctx.arc(466, 175, 15, 0, Math.PI*2); ctx.fill();
            }
            if (type === 'street' || type === 'traffic') {
                // Draw car rectangles
                ctx.fillStyle = '#e67e22';
                ctx.fillRect(320, 310, 140, 60); // Car 1
                ctx.fillStyle = '#111';
                ctx.beginPath(); ctx.arc(350, 370, 15, 0, Math.PI*2); ctx.fill();
                ctx.beginPath(); ctx.arc(430, 370, 15, 0, Math.PI*2); ctx.fill();
            }

            currentImageBase64 = canvas.toDataURL('image/jpeg');
            document.getElementById('img-original').src = currentImageBase64;
            document.getElementById('btn-detect').disabled = false;
            document.getElementById('results').style.display = 'none';
        }

        async function runDetection() {
            if (!currentImageBase64) return;
            const btn = document.getElementById('btn-detect');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btn-text');
            const conf = parseFloat(document.getElementById('conf-slider').value);

            btn.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.innerText = 'Analyzing Image...';

            try {
                const res = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: currentImageBase64, conf: conf })
                });

                const data = await res.json();
                if (data.error) {
                    alert('Detection Error: ' + data.error);
                    return;
                }

                // Display annotated result
                document.getElementById('img-annotated').src = data.annotated_image;
                document.getElementById('latency-badge').innerText = `${data.inference_time_ms} ms (${data.engine || 'yolov8'})`;

                // Render table
                const tableContainer = document.getElementById('table-container');
                if (!data.summary || data.summary.length === 0) {
                    tableContainer.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem 0;">No objects detected above confidence threshold ' + conf + '.</p>';
                } else {
                    let html = '<table class="summary-table"><thead><tr><th>Class</th><th>Count</th><th>Avg Confidence</th></tr></thead><tbody>';
                    for (const row of data.summary) {
                        html += `<tr>
                            <td><span class="class-badge">${row.class}</span></td>
                            <td><strong>${row.count}</strong></td>
                            <td class="conf-text">${row.avg_confidence}</td>
                        </tr>`;
                    }
                    html += '</tbody></table>';
                    tableContainer.innerHTML = html;
                }

                document.getElementById('results').style.display = 'block';
                document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

            } catch (err) {
                alert('Network error while requesting /api/detect: ' + err.message);
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.innerText = 'Detect Objects';
            }
        }
    </script>
</body>
</html>"""


def get_response_for_path(path: str, method: str = 'GET', body_data: bytes = b''):
    """Router helper returning (status_code, content_type, body_bytes)."""
    if path == '/api/health':
        return 200, 'application/json', json.dumps({"status": "healthy", "service": "Object Detection & Tracking API", "version": "1.0.0"}).encode('utf-8')
    elif path == '/api/info':
        return 200, 'application/json', json.dumps(PROJECT_INFO, indent=2).encode('utf-8')
    elif path == '/api/classes':
        return 200, 'application/json', json.dumps({"total_classes": len(CLASSES), "classes": CLASSES}, indent=2).encode('utf-8')
    elif path == '/api/stats':
        return 200, 'application/json', json.dumps({
            "dataset": "People Detection (Roboflow v12i)",
            "total_images": 17401,
            "total_annotations": 145885,
            "classes": 53,
            "splits": {"train": 15210, "valid": 1431, "test": 760}
        }, indent=2).encode('utf-8')
    elif path == '/api/detect' and method == 'POST':
        try:
            payload = json.loads(body_data.decode('utf-8'))
            img_data = payload.get('image', '')
            conf = float(payload.get('conf', 0.25))

            # Strip data URL prefix if present
            if ',' in img_data:
                img_data = img_data.split(',', 1)[1]

            image_bytes = base64.b64decode(img_data)
            result = run_yolo_detection(image_bytes, conf_threshold=conf)
            return 200, 'application/json', json.dumps(result).encode('utf-8')
        except Exception as e:
            return 400, 'application/json', json.dumps({"error": f"Failed to process detection: {str(e)}"}).encode('utf-8')
    else:
        return 200, 'text/html; charset=utf-8', HTML_PAGE.encode('utf-8')


# --- Pattern 1: Vercel Serverless Function (BaseHTTPRequestHandler) ---
class handler(BaseHTTPRequestHandler):
    """Vercel serverless request handler."""
    def do_GET(self):
        code, content_type, body = get_response_for_path(self.path, method='GET')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body_data = self.rfile.read(content_length) if content_length > 0 else b''
        code, content_type, body = get_response_for_path(self.path, method='POST', body_data=body_data)
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# --- Pattern 2: WSGI Application Callable ---
def app(environ, start_response):
    """Standard WSGI entrypoint for Vercel/Gunicorn/uWSGI."""
    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')
    
    if method == 'OPTIONS':
        start_response('200 OK', [
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
            ('Access-Control-Allow-Headers', 'Content-Type')
        ])
        return [b'']

    body_data = b''
    if method == 'POST':
        try:
            content_length = int(environ.get('CONTENT_LENGTH', 0))
        except (ValueError, TypeError):
            content_length = 0
        if content_length > 0:
            body_data = environ['wsgi.input'].read(content_length)

    code, content_type, body = get_response_for_path(path, method=method, body_data=body_data)
    status = '200 OK' if code == 200 else f'{code} Error'
    headers = [
        ('Content-Type', content_type),
        ('Access-Control-Allow-Origin', '*'),
        ('Content-Length', str(len(body)))
    ]
    start_response(status, headers)
    return [body]

# Additional standard aliases
application = app

if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    port = 8000
    print(f"🚀 Serving YOLO Object Detection Web App on http://localhost:{port}")
    httpd = make_server('0.0.0.0', port, app)
    httpd.serve_forever()
