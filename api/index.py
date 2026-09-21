"""
Vercel Serverless Function entrypoint.
Imports and delegates requests to the main application in app.py.
"""

from app import app, handler, application

__all__ = ['app', 'handler', 'application']
