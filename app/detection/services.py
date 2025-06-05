# app/detection/services.py
from io import BytesIO
import cv2
import numpy as np
from io import BytesIO
import base64
import os
from datetime import datetime
from pyzbar.pyzbar import decode
from PIL import Image

net = cv2.dnn.readNetFromCaffe(
    "mobilenet_ssd/deploy.prototxt",
    "mobilenet_ssd/mobilenet_iter_73000.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

def read_code_from_image(image_bytes: bytes) -> list:
    try:
        img = Image.open(BytesIO(image_bytes))
        results = decode(img)

        if not results:
            return "No se detectó ningún código"

        codes = []
        for code in results:
            bbox = code.rect  # Bounding box con x, y, width, height

            codes.append({
                "type": code.type,
                "data": code.data.decode("utf-8"),
                "bounding_box": {
                    "x": bbox.left,
                    "y": bbox.top,
                    "width": bbox.width,
                    "height": bbox.height
                }
            })

        return codes

    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"

import cv2
import numpy as np
import base64
import os
from pyzbar.pyzbar import decode

def detect_code_boxes_from_base64(image_base64: str, save_path: str = "deteccion_resultado.jpg") -> list:
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",")[1]

    image_data = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No se pudo decodificar la imagen")

    results = decode(image)
    boxes = []

    for obj in results:
        (x, y, w, h) = obj.rect
        tipo = obj.type

        boxes.append({
            "type": tipo,
            "bounding_box": {
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h)
            }
        })

        # Dibujar rectángulo y texto
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, tipo, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Guardar imagen con bounding boxes
    if boxes:
        cv2.imwrite(save_path, image)

    return boxes

def preprocess_image_for_qr(image: np.ndarray) -> np.ndarray:
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mejorar el contraste
    gray = cv2.equalizeHist(gray)

    # Aplicar umbral adaptativo para binarización
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return thresh

def try_multiple_rotations(image: np.ndarray) -> list:
    results = decode(image)
    if results:
        return results

    rotation_flags = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }

    for angle in [90, 180, 270]:
        rotated = cv2.rotate(image, rotation_flags[angle])
        results = decode(rotated)
        if results:
            return results
        
    return []

def read_qr_with_opencv(image: np.ndarray) -> list:
    detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
    result = []
    if retval:
        for info, pts in zip(decoded_info, points):
            if info:
                result.append({
                    "type": "QR_CODE",
                    "data": info,
                    "bounding_box": {
                        "x1": int(pts[0][0]),
                        "y1": int(pts[0][1]),
                        "x2": int(pts[2][0]),
                        "y2": int(pts[2][1])
                    }
                })
    return result

def read_code_from_base64(image_base64: str) -> list:
    try:
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]

        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("No se pudo decodificar la imagen")

        # Intentar con pyzbar (sin preprocesamiento)
        results = try_multiple_rotations(image)
        output = []

        for obj in results:
            (x, y, w, h) = obj.rect
            output.append({
                "type": obj.type,
                "data": obj.data.decode("utf-8"),
                "bounding_box": {
                    "x1": int(x),
                    "y1": int(y),
                    "x2": int(x + w),
                    "y2": int(y + h)
                }
            })

        # Si no detectó con pyzbar, intentar con OpenCV QR detector
        if not output:
            output = read_qr_with_opencv(image)

        # Como último recurso, usar imagen preprocesada
        if not output:
            processed = preprocess_image_for_qr(image)
            results = try_multiple_rotations(processed)
            for obj in results:
                (x, y, w, h) = obj.rect
                output.append({
                    "type": obj.type,
                    "data": obj.data.decode("utf-8"),
                    "bounding_box": {
                        "x1": int(x),
                        "y1": int(y),
                        "x2": int(x + w),
                        "y2": int(y + h)
                    }
                })

        return output

    except Exception as e:
        raise RuntimeError(f"Error al procesar imagen base64: {str(e)}")

    
def detect_persons_haar_from_base64(image_base64: str) -> list:
    try:
        # Remover encabezado si viene con "data:image..."
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]

        # Decodificar imagen base64
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("No se pudo decodificar la imagen")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Cargar clasificador Haar
        cascade = cv2.CascadeClassifier("haar_models/haarcascade_fullbody.xml")

        # Detectar personas
        detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        persons = []
        for (x, y, w, h) in detections:
            # Dibujar bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            persons.append({
                "bounding_box": {
                    "x1": int(x),
                    "y1": int(y),
                    "x2": int(x + w),
                    "y2": int(y + h),
                }
            })

        # Si se detectaron personas, guardar imagen
        # if persons:
        #     output_dir = "haar_detected_images"
        #     os.makedirs(output_dir, exist_ok=True)

        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     output_path = os.path.join(output_dir, f"haar_detected_{timestamp}.jpg")
        #     cv2.imwrite(output_path, image)

        return persons

    except Exception as e:
        raise RuntimeError(f"Error al procesar imagen Haarcascade: {str(e)}")
    
def detect_faces_haar_from_base64(image_base64: str) -> list:
    try:
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]

        # Decodificar base64 a imagen OpenCV
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("No se pudo decodificar la imagen")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Cargar Haarcascade
        face_cascade = cv2.CascadeClassifier("haar_models/haarcascade_frontalface_default.xml")

        # Detección
        detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        faces = []
        image_with_boxes = image.copy()  # Trabajamos sobre una copia

        for (x, y, w, h) in detections:
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 2)
            faces.append({
                "bounding_box": {
                    "x1": int(x),
                    "y1": int(y),
                    "x2": int(x + w),
                    "y2": int(y + h),
                }
            })

        # Guardar imagen si hay rostros
        # if faces:
        #     output_dir = "haar_detected_faces"
        #     os.makedirs(output_dir, exist_ok=True)
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     output_path = os.path.join(output_dir, f"face_detected_{timestamp}.jpg")

        #     # Guardar la imagen modificada con los cuadros
        #     cv2.imwrite(output_path, image_with_boxes)

        return faces

    except Exception as e:
        raise RuntimeError(f"Error en detección de rostros Haarcascade: {str(e)}")

def detect_persons_mobilenet_from_base64(image_base64: str) -> list:
    try:
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]

        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("No se pudo decodificar la imagen")

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        persons = []
        image_with_boxes = image.copy()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")

                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_with_boxes, f"Persona {confidence:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    persons.append({
                        "confidence": round(float(confidence), 3),
                        "bounding_box": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        }
                    })

        # if persons:
        #     output_dir = "mobilenet_detected_persons"
        #     os.makedirs(output_dir, exist_ok=True)
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     output_path = os.path.join(output_dir, f"person_detected_{timestamp}.jpg")
        #     cv2.imwrite(output_path, image_with_boxes)

        return persons

    except Exception as e:
        raise RuntimeError(f"Error en detección con MobileNet SSD: {str(e)}")