# app/detection/services.py
from io import BytesIO
import cv2
import numpy as np
from io import BytesIO
import base64
import os
from datetime import datetime

net = cv2.dnn.readNetFromCaffe(
    "mobilenet_ssd/deploy.prototxt",
    "mobilenet_ssd/mobilenet_iter_73000.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
    
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