import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

# Model configuration
MODEL_PATH = "hair_classifier_empty.onnx"
TARGET_SIZE = (200, 200)

# ImageNet normalization parameters
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Load model once (outside handler for Lambda optimization)
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess(img):
    x = np.array(img, dtype=np.float32)
    x = x / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return x


def predict(url):
    img = download_image(url)
    img = prepare_image(img, TARGET_SIZE)
    x = preprocess(img)
    result = session.run([output_name], {input_name: x})
    return float(result[0][0][0])


def lambda_handler(event, context):
    url = event.get("url")
    if not url:
        return {"statusCode": 400, "body": "Missing url parameter"}
    
    prediction = predict(url)
    return {"statusCode": 200, "prediction": prediction}


if __name__ == "__main__":
    # Test locally
    url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    result = predict(url)
    print(f"Prediction: {result}")