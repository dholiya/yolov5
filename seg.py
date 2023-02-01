from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from segment import predict
import sys
import numpy as np
import torch
from PIL import Image
import argparse
from ultralytics import YOLO
import torchvision.transforms as T
from utils.general import cv2
import logging
import base64
import io

app = FastAPI()

@app.get('/')
def hello():
    return "0hello"

@app.post("/api")
async def image_segmentation(file: UploadFile = File(...)):
    logging.info(file.filename)
    ndarray = read_imagefile(await file.read(), file.filename)
    img = cv2.imencode('.jpg', ndarray)[1]
    return base64.b64encode(img).decode('utf-8')

def read_imagefile(file, filename) -> str:
    # image = Image.open(file).convert('RGB')
    image = Image.open(BytesIO(file))
    with open(filename, 'wb') as f:
        f.write(file)
    save_dir = predict.run(weights='yolov5m-seg.pt', source=filename, retina_masks=True)
    # predict.run(weights='yolov5m-seg.pt', source='data/images/sample.jpg')
    return save_dir

if __name__ == "__main__":
    uvicorn.run(app)