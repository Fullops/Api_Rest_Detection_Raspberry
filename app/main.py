# app/main.py
from fastapi import FastAPI
from app.detection.routes import router as detection_router
from app.middleware import setup_middlewares
from app.detection.mqtt import router as mqtt_router

app = FastAPI(title="Backend for QR, Barcode and Object Detection")

# Configurar todos los middlewares
setup_middlewares(app)

# Incluir rutas
app.include_router(detection_router, prefix="/api", tags=["Detección"])
app.include_router(mqtt_router, prefix="/api", tags=["MQTT"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "API funcionando correctamente"}