"""
Vercel entrypoint app.py
Imports and exposes handler, app, application from main.py.
"""
from main import handler, app, application, HTML_PAGE, PROJECT_INFO, CLASSES, run_yolo_detection

__all__ = ['handler', 'app', 'application']

if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    server = make_server('0.0.0.0', 8000, app)
    print("Serving on http://localhost:8000")
    server.serve_forever()
