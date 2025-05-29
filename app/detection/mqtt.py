from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.detection.services import detect_persons_haar_from_base64  # o mobilenet si prefieres
import logging

router = APIRouter()

@router.post("/mqtt/message")
async def receive_mqtt_message(request: Request):
    data = await request.json()
    logging.info(f"[MQTT] Mensaje recibido desde EMQX: {data}")

    payload = data.get("payload")

    if not payload or "image_base64" not in payload:
        return JSONResponse(status_code=400, content={"error": "Falta 'image_base64'"})

    try:
        result = detect_persons_haar_from_base64(payload["image_base64"])
        return JSONResponse(
            status_code=200,
            content={
                "message": "Detección ejecutada automáticamente",
                "result": result
            }
        )
    except Exception as e:
        logging.error(f"[MQTT] Error al procesar la imagen: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error al ejecutar detección: {str(e)}"}
        )
