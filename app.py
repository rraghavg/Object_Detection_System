"""
Real-Time Object Detection & Tracking System - Web Application & API Entrypoint
Compatible with Vercel Serverless Python runtime (WSGI/ASGI), local execution, and cloud deployment.
"""

import json
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
        "device": "Apple Silicon MPS / CUDA / CPU"
    }
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Object Detection & Tracking System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
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
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
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
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 2.5rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .pulse {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        h1 {
            font-size: 2.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 0.75rem;
            background: linear-gradient(135deg, #f0f4fc 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            max-width: 650px;
            margin: 0 auto;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: border-color 0.2s;
        }
        .card:hover {
            border-color: var(--accent-blue);
        }
        .card h3 {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .card p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }
        @media (max-width: 640px) {
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1rem;
            text-align: center;
        }
        .stat-val {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--accent-blue);
            font-family: 'Fira Code', monospace;
        }
        .stat-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 600;
            margin: 2rem 0 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .endpoints-list {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }
        .endpoint-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border);
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
        }
        .endpoint-item:last-child { border-bottom: none; }
        .method {
            background: rgba(59, 130, 246, 0.2);
            color: var(--accent-blue);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.75rem;
            margin-right: 0.75rem;
        }
        .endpoint-link {
            color: var(--accent-blue);
            text-decoration: none;
        }
        .endpoint-link:hover { text-decoration: underline; }
        .endpoint-desc {
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
        }
        pre {
            background: #0d1117;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid var(--border);
            font-family: 'Fira Code', monospace;
            font-size: 0.85rem;
            color: #58a6ff;
            margin-top: 1rem;
        }
        footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        footer a { color: var(--accent-blue); text-decoration: none; }
        footer a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge"><div class="pulse"></div> System Live on Vercel</div>
            <h1>Real-Time Object Detection & Tracking</h1>
            <p class="subtitle">Interview-ready computer vision pipeline combining YOLOv8 for frame-by-frame multi-object detection and ByteTrack for persistent ID tracking.</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val">17,401</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">145K+</div>
                <div class="stat-label">Annotations</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">53</div>
                <div class="stat-label">Classes</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">20/20</div>
                <div class="stat-label">Tests Passed</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>🔍 YOLOv8 Detection</h3>
                <p>Single-stage detection head delivering high FPS inference across varying scales, lighting conditions, and camera angles.</p>
            </div>
            <div class="card">
                <h3>🎯 ByteTrack Tracking</h3>
                <p>Associates detections across consecutive frames using Kalman filtering and bipartite matching to maintain persistent object IDs.</p>
            </div>
            <div class="card">
                <h3>📊 Quantitative Evaluation</h3>
                <p>Vectorized performance analysis reporting IoU, Precision, Recall, F1-Score, mAP@50, and mAP@50:95.</p>
            </div>
        </div>

        <div class="section-title">⚡ Available API Endpoints</div>
        <div class="endpoints-list">
            <div class="endpoint-item">
                <div><span class="method">GET</span><a class="endpoint-link" href="/api/health">/api/health</a></div>
                <div class="endpoint-desc">Service health status & timestamp</div>
            </div>
            <div class="endpoint-item">
                <div><span class="method">GET</span><a class="endpoint-link" href="/api/info">/api/info</a></div>
                <div class="endpoint-desc">Architecture & project metadata</div>
            </div>
            <div class="endpoint-item">
                <div><span class="method">GET</span><a class="endpoint-link" href="/api/classes">/api/classes</a></div>
                <div class="endpoint-desc">Taxonomy of all 53 detected classes</div>
            </div>
            <div class="endpoint-item">
                <div><span class="method">GET</span><a class="endpoint-link" href="/api/stats">/api/stats</a></div>
                <div class="endpoint-desc">Dataset splits & annotation distribution</div>
            </div>
        </div>

        <div class="section-title">💻 Quick CLI Usage</div>
        <pre><code># Track objects in video
python src/track.py --source data/test.mp4 --model models/best.pt --output outputs/tracking/result.mp4

# Run quantitative evaluation
python src/evaluate.py --data configs/data.yaml --model models/best.pt --split test</code></pre>

        <footer>
            Built by <strong>Raghav Gupta</strong> | <a href="https://github.com/rraghavg/Object_Detection_System" target="_blank">View on GitHub</a>
        </footer>
    </div>
</body>
</html>
"""

def app(environ, start_response):
    """
    Standard WSGI Application handler.
    Called by Vercel serverless python runner or any WSGI web server (Gunicorn, uWSGI, wsgiref).
    """
    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')

    headers = [('Access-Control-Allow-Origin', '*'), ('Access-Control-Allow-Methods', 'GET, OPTIONS')]

    if method == 'OPTIONS':
        start_response('200 OK', headers)
        return [b'']

    # Routing
    if path == '/api/health':
        status = '200 OK'
        headers.append(('Content-Type', 'application/json'))
        body = json.dumps({"status": "healthy", "service": "Object Detection & Tracking API", "version": "1.0.0"}).encode('utf-8')

    elif path == '/api/info':
        status = '200 OK'
        headers.append(('Content-Type', 'application/json'))
        body = json.dumps(PROJECT_INFO, indent=2).encode('utf-8')

    elif path == '/api/classes':
        status = '200 OK'
        headers.append(('Content-Type', 'application/json'))
        body = json.dumps({
            "total_classes": len(CLASSES),
            "classes": CLASSES
        }, indent=2).encode('utf-8')

    elif path == '/api/stats':
        status = '200 OK'
        headers.append(('Content-Type', 'application/json'))
        body = json.dumps({
            "dataset": "People Detection (Roboflow v12i)",
            "splits": {
                "train": {"images": 15210, "percentage": "87.4%"},
                "valid": {"images": 1431, "percentage": "8.2%"},
                "test": {"images": 760, "percentage": "4.4%"}
            },
            "total_images": 17401,
            "total_annotations": 145885,
            "avg_objects_per_image": 8.4,
            "top_classes": [
                {"name": "person", "count": 56084, "percentage": "38.4%"},
                {"name": "Pedestrian", "count": 31527, "percentage": "21.6%"},
                {"name": "face", "count": 8181, "percentage": "5.6%"},
                {"name": "chair", "count": 3740, "percentage": "2.6%"},
                {"name": "car", "count": 3549, "percentage": "2.4%"}
            ]
        }, indent=2).encode('utf-8')

    else:
        # Default: Serve Interactive Dashboard
        status = '200 OK'
        headers.append(('Content-Type', 'text/html; charset=utf-8'))
        body = HTML_TEMPLATE.encode('utf-8')

    start_response(status, headers)
    return [body]


# Vercel entrypoint aliases
handler = app
application = app

if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    port = 8000
    print(f"🚀 Starting development server at http://localhost:{port}")
    httpd = make_server('0.0.0.0', port, app)
    httpd.serve_forever()
