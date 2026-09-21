"""
Real-Time Object Detection & Tracking System - Unified Vercel & Web Entrypoint
Supports:
1. Vercel Serverless Function (BaseHTTPRequestHandler 'handler')
2. WSGI Web Application ('app', 'application')
3. Standalone local server (python main.py)
"""

import json
from http.server import BaseHTTPRequestHandler

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
            --accent-green: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2.5rem 1rem;
        }
        .container { max-width: 950px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 2.5rem; }
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
</body>
</html>"""


def get_response_for_path(path: str):
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
    else:
        return 200, 'text/html; charset=utf-8', HTML_PAGE.encode('utf-8')


# --- Pattern 1: Vercel Serverless Function (BaseHTTPRequestHandler) ---
class handler(BaseHTTPRequestHandler):
    """Vercel serverless request handler."""
    def do_GET(self):
        code, content_type, body = get_response_for_path(self.path)
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.end_headers()


# --- Pattern 2: WSGI Application Callable ---
def app(environ, start_response):
    """Standard WSGI entrypoint for Vercel/Gunicorn/uWSGI."""
    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')
    
    if method == 'OPTIONS':
        start_response('200 OK', [('Access-Control-Allow-Origin', '*'), ('Access-Control-Allow-Methods', 'GET, OPTIONS')])
        return [b'']
    
    code, content_type, body = get_response_for_path(path)
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
    print(f"Serving on http://localhost:{port}")
    httpd = make_server('0.0.0.0', port, app)
    httpd.serve_forever()
