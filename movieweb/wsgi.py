"""
WSGI config for movieweb project.

It exposes the WSGI callable as a module-level variable named ``application``
and ``app`` for Vercel Serverless Functions.
"""

import os
import sys
import traceback
from pathlib import Path

# Ensure project root is in sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movieweb.settings')

try:
    from django.core.wsgi import get_wsgi_application
    _django_app = get_wsgi_application()

    def application(environ, start_response):
        try:
            return _django_app(environ, start_response)
        except Exception:
            err_msg = traceback.format_exc()
            start_response("500 Internal Server Error", [("Content-Type", "text/plain; charset=utf-8")])
            return [f"DJANGO RUNTIME ERROR ON VERCEL:\n\n{err_msg}".encode("utf-8")]

except Exception:
    startup_err = traceback.format_exc()
    def application(environ, start_response):
        start_response("500 Internal Server Error", [("Content-Type", "text/plain; charset=utf-8")])
        return [f"DJANGO STARTUP ERROR ON VERCEL:\n\n{startup_err}".encode("utf-8")]

# Alias for Vercel
app = application
