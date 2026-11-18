"""
WSGI config for movieweb project.

It exposes the WSGI callable as a module-level variable named ``application``
and ``app`` for Vercel Serverless Functions.
"""

import os
import sys
from pathlib import Path
from django.core.wsgi import get_wsgi_application

# Ensure project root is in sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movieweb.settings')

application = get_wsgi_application()

# Alias for Vercel Python runtime
app = application
