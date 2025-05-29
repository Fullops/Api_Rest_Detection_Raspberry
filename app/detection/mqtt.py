from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging

router = APIRouter()

@router.post("/mqtt/message")
async def receive_mqtt_message(request: Request):
    body = await request.json()
    logging.info(f"[MQTT] Mensaje recibido desde EMQX: {body}")

    # Aquí puedes hacer algo con el mensaje, por ejemplo:
    topic = body.get("topic")
    payload = body.get("payload")

    return JSONResponse(status_code=200, content={"status": "ok", "topic": topic, "payload": payload})
