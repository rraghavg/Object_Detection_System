"""
Vercel Serverless Function - api/index.py
Provides both BaseHTTPRequestHandler 'handler' and WSGI 'app' / 'application'.
"""

import json
from http.server import BaseHTTPRequestHandler

# Import from main if available, otherwise define standalone
try:
    from main import handler, app, application
except ImportError:
    CLASSES = [
        '0', '1', '2', 'Bicycle', 'Bike', 'Car', 'Cyclist', 'Pedestrian', 'Pedestrians',
        'Persona', 'Pessoa', 'Signboard', 'Stopper', 'aeroplane', 'bag', 'berdiri',
        'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'cyclist', 'dianzhuan', 'diningtable', 'dog', 'face', 'forklift', 'handbag',
        'head', 'helmet', 'high', 'horse', 'jatuh', 'laptop', 'low', 'medium',
        'motorbike', 'people', 'person', 'persons', 'pottedplant', 'refrigerator',
        'sheep', 'sofa', 'teddy bear', 'train', 'tv', 'tvmonitor', 'vase'
    ]

    class handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data = {
                "status": "online",
                "service": "Object Detection & Tracking API",
                "version": "1.0.0",
                "classes": len(CLASSES)
            }
            self.wfile.write(json.dumps(data).encode('utf-8'))

    def app(environ, start_response):
        start_response('200 OK', [('Content-Type', 'application/json')])
        return [b'{"status": "online"}']

    application = app

__all__ = ['handler', 'app', 'application']
