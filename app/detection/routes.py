from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from app.detection.schemas import ImageBase64Payload
from app.detection.services import read_code_from_image
from app.detection.services import detect_code_boxes_from_base64
from app.detection.services import read_code_from_base64
from app.detection.services import detect_persons_haar_from_base64
from app.detection.services import detect_faces_haar_from_base64
from app.detection.services import detect_persons_mobilenet_from_base64
from app.detection.services import detect_persons_from_base64

router = APIRouter()

@router.post("/read-code/")
async def read_code(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = read_code_from_image(image_bytes)

        if isinstance(result, str):  # mensaje de error o no detección
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": status.HTTP_204_NO_CONTENT,
                    "message": result,
                    "result": None
                }
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": status.HTTP_200_OK,
                "message": "Códigos detectados exitosamente.",
                "result": result
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la imagen: {str(e)}"
        )

@router.post("/detect-code-boxes/")
async def detect_code_boxes(payload: ImageBase64Payload):
    try:
        if not hasattr(payload, "image_base64") or not payload.image_base64:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Falta el campo 'image_base64' o está vacío en el cuerpo de la solicitud."
            )

        result = detect_code_boxes_from_base64(payload.image_base64)

        if not result:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": status.HTTP_204_NO_CONTENT,
                    "message": "No se detectó ningún código.",
                    "result": None
                }
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": status.HTTP_200_OK,
                "message": "Códigos detectados.",
                "result": result
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al detectar códigos: {str(e)}"
        )

@router.post("/read-code-base64/")
async def read_code_from_base64_route(payload: ImageBase64Payload):
    try:
        if not hasattr(payload, "image_base64") or not payload.image_base64:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Falta el campo 'image_base64' o está vacío en el cuerpo de la solicitud."
            )

        result = read_code_from_base64(payload.image_base64)

        if not result:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": status.HTTP_204_NO_CONTENT,
                    "message": "No se detectó ningún código en la imagen.",
                    "result": None
                }
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": status.HTTP_200_OK,
                "message": "Código leído exitosamente.",
                "result": result
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la imagen: {str(e)}"
        )

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
    
@router.post("/detect-yolo/")
async def detect_from_base64(payload: ImageBase64Payload):
    try:
        result = detect_persons_from_base64(payload.image_base64)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Personas detectadas exitosamente." if result else "No se detectaron personas.",
                "result": result or []
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar imagen base64: {str(e)}"
        )