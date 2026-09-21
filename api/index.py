"""
Vercel Serverless Function - api/index.py
Provides both BaseHTTPRequestHandler 'handler' and WSGI 'app' / 'application'.
"""

import sys
import os

# Ensure project root is in sys.path when bundled in api/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from main import handler, app, application
except ImportError:
    # Fallback definition if main cannot be resolved
    import json
    from http.server import BaseHTTPRequestHandler

    class handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "online", "service": "Object Detection API"}).encode('utf-8'))

    def app(environ, start_response):
        start_response('200 OK', [('Content-Type', 'application/json')])
        return [b'{"status": "online"}']

    application = app

__all__ = ['handler', 'app', 'application']
