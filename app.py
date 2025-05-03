# Modified version of E.py for PythonAnywhere hosting
# Original: ESCSSolver.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import base64
import threading
import signal
import requests
import sys
import json
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageSequence
import asyncio
from concurrent.futures import ThreadPoolExecutor

VERSION = '3.0'
solveCounter = 1

def readApiKey():
    try:
        with open('key.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        # Handle missing key file more gracefully in web environment
        return None

def Authkey():
    return True

def resource_path(relative_path):
    # Adjust resource path for PythonAnywhere environment
    base_path = os.path.abspath('.')
    path = os.path.join(base_path, relative_path)
    if not os.path.exists(path):
        return os.path.join('attachments', relative_path)
    return path

class CaptchaEngine:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        configPath = resource_path('data.cfg')
        weightPath = resource_path('data.weights')
        labelPath = resource_path('data.nms')
        self.LABELS_PLATE = open(labelPath).read().strip().split('\n')
        self.net = cv2.dnn_DetectionModel(configPath, weightPath)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net.setInputSize(self.width, self.height)
        self.net.setInputScale(0.00392156862745098)
        self.net.setInputSwapRB(True)
        blank_image = np.zeros((self.width, self.height, 3), np.uint8)
        self.net.detect(blank_image, confThreshold=0.4, nmsThreshold=0.4)
        self.lock = threading.Lock()

    def solve_cv2(self, image):
        def get_key_x(item):
            return item[1]
        H, W, _ = image.shape
        classes, confidences, boxes = self.net.detect(image, confThreshold=0.5, nmsThreshold=0.3)
        info = []
        if not len(classes) == 0:
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                x, y, w, h = box
                info.append((self.LABELS_PLATE[classId], x, y, w, h, confidence))
        info = sorted(info, key=get_key_x)
        text = ''
        for d in info:
            lbl, x, y, w, h, _ = d
            if w > 10 and h > 10 and (w < W) and (h < H):
                text = text + lbl.replace(' ', '')
        return text

    def solve(self, sbase64):
        cvimg = self.base64_to_cv2(sbase64)
        return self.solve_cv2(cvimg)

    def base64_to_cv2(self, sbase64):
        base64_str = sbase64
        arr = sbase64.split(',')
        if len(arr) > 1:
            base64_str = arr[1]
        im_bytes = base64.b64decode(base64_str)
        nparr = np.fromstring(im_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image

app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=['*'], 
    allow_headers=['*']
)

width = 256
height = 256
Solver = CaptchaEngine(width, height)
executor = ThreadPoolExecutor(max_workers=4)

def merge_gif_preserve_color(base64_gif_str):
    gif_data = base64.b64decode(base64_gif_str)
    gif_io = BytesIO(gif_data)
    gif = Image.open(gif_io)
    frames = []
    for frame in ImageSequence.Iterator(gif):
        rgb_frame = frame.convert('RGB')
        np_frame = np.array(rgb_frame)
        frames.append(np_frame)
    merged = np.minimum.reduce(frames)
    return merged

@app.post('/api_base64')
async def api_base64(request: Request):
    global solveCounter
    req = await request.json()
    base64_str = req['data']
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',', 1)[1]
    cv_img = merge_gif_preserve_color(base64_str)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(executor, Solver.solve_cv2, cv_img)
    solveCounter = solveCounter + 1
    data = {'result': text, 'status': '1'}
    response = JSONResponse(content=data)
    return response

@app.get('/')
async def root():
    return {"message": "ESCS Solver API", "version": VERSION, "solves": solveCounter}

# For local testing only - not used on PythonAnywhere
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=1234)
