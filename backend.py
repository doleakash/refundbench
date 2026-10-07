"""Compatibility import for the FastAPI app; wiring lives in app.bootstrap."""

from app.bootstrap import create_app

app = create_app()
