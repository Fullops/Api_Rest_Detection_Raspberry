# app/middlewares.py
import time
import logging
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Middleware personalizado para logging
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", str(time.time()))
        logging.info(f"Request [{request_id}]: {request.method} {request.url}")

        response = await call_next(request)

        process_time = time.time() - start_time
        logging.info(f"Response [{request_id}]: {response.status_code} ({process_time:.2f}s)")

        return response

# Función centralizada para registrar middlewares
def setup_middlewares(app):
    """Agrega todos los middlewares necesarios a la aplicación FastAPI."""

    # Middleware de CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción, especifica los dominios permitidos
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware de logging personalizado
    app.add_middleware(LoggingMiddleware)
