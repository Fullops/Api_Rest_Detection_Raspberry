from pydantic import BaseModel

class ImageBase64Payload(BaseModel):
    image_base64: str
