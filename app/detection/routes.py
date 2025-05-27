from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from app.detection.schemas import ImageBase64Payload
from app.detection.services import detect_persons_haar_from_base64
from app.detection.services import detect_faces_haar_from_base64
from app.detection.services import detect_persons_mobilenet_from_base64
from app.detection.schemas import ImageBase64Payload

router = APIRouter()

@router.post("/detect-haar-base64/")
async def detect_haar_from_base64(payload: ImageBase64Payload):
    try:
        result = detect_persons_haar_from_base64(payload.image_base64)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Personas detectadas con Haarcascade." if result else "No se detectaron personas.",
                "result": result or []
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en Haarcascade: {str(e)}"
        )

@router.post("/detect-face-haar/")
async def detect_faces_haar(payload: ImageBase64Payload):
    try:
        result = detect_faces_haar_from_base64(payload.image_base64)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Rostros detectados." if result else "No se detectaron rostros.",
                "result": result or []
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en detección Haarcascade: {str(e)}"
        )

@router.post("/detect-mobilenet/")
async def detect_persons_mobilenet(payload: ImageBase64Payload):
    try:
        result = detect_persons_mobilenet_from_base64(payload.image_base64)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Personas detectadas con MobileNet SSD." if result else "No se detectaron personas.",
                "result": result or []
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en MobileNet SSD: {str(e)}"
        )