"""Unit tests for web app and API endpoints."""
import json
import base64
import io
from main import app, get_response_for_path


def test_api_health():
    code, ctype, body = get_response_for_path('/api/health')
    assert code == 200
    assert ctype == 'application/json'
    data = json.loads(body.decode('utf-8'))
    assert data['status'] == 'healthy'


def test_api_info():
    code, ctype, body = get_response_for_path('/api/info')
    assert code == 200
    data = json.loads(body.decode('utf-8'))
    assert 'architecture' in data
    assert 'YOLOv8' in data['architecture']


def test_api_classes():
    code, ctype, body = get_response_for_path('/api/classes')
    assert code == 200
    data = json.loads(body.decode('utf-8'))
    assert data['total_classes'] == 53
    assert 'person' in data['classes']


def test_api_stats():
    code, ctype, body = get_response_for_path('/api/stats')
    assert code == 200
    data = json.loads(body.decode('utf-8'))
    assert data['total_images'] == 17401


def test_html_dashboard():
    code, ctype, body = get_response_for_path('/')
    assert code == 200
    assert 'text/html' in ctype
    html = body.decode('utf-8')
    assert 'Try Object Detection' in html
    assert 'Detected Objects Summary' in html


def test_api_detect_invalid_payload():
    code, ctype, body = get_response_for_path('/api/detect', method='POST', body_data=b'invalid json')
    assert code == 400
    data = json.loads(body.decode('utf-8'))
    assert 'error' in data
