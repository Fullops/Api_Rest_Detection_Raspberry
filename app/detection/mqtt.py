from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.detection.services import detect_persons_mobilenet_from_base64  # o mobilenet si prefieres
from paho.mqtt import client as mqtt_client
import logging
from paho.mqtt import client as mqtt_client
import ssl
import json

router = APIRouter()

broker = "d755c7bc.ala.us-east-1.emqxsl.com"
port = 8883
username = "user_test"
password = "1234"
topic = "deteccion/respuesta"
ca_cert = "/etc/mosquitto/ca_certificates/emqxsl-ca.crt"

def publish_response(message: dict):
    client = mqtt_client.Client()
    client.username_pw_set(username, password)
    client.tls_set(ca_certs=ca_cert, tls_version=ssl.PROTOCOL_TLSv1_2)
    client.connect(broker, port)
    client.loop_start()

    result = client.publish(topic, json.dumps(message))

    # Espera explícitamente hasta que se envíe
    result.wait_for_publish()

    client.loop_stop()
    client.disconnect()


@router.post("/mqtt/message")
async def receive_mqtt_message(request: Request):
    data = await request.json()
    logging.info(f"[MQTT] Recibido: {data}")

    payload = data.get("payload")

    if not payload or "image_base64" not in payload:
        error = {"status": "error", "message": "Falta 'image_base64'"}
        publish_response(error)
        logging.warning("[MQTT] Enviado error por MQTT")
        return JSONResponse(status_code=400, content=error)

    try:
        result = detect_persons_mobilenet_from_base64(payload["image_base64"])
        response = {
            "status": "ok",
            "message": "Detección ejecutada",
            "result": result
        }
        publish_response(response)
        logging.info("[MQTT] Enviado resultado por MQTT")
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        error = {"status": "error", "message": str(e)}
        publish_response(error)
        logging.error(f"[MQTT] Error y enviado por MQTT: {e}")
        return JSONResponse(status_code=500, content=error)
