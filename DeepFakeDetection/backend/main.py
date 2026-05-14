import sys
import os
import io
import base64

import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Allow imports from the parent DeepFakeDetection directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.efficientnet_model import model, device

# Paths relative to this file
_BASE = os.path.dirname(__file__)
CHECKPOINT = os.path.join(_BASE, "..", "checkpoints", "best_model.pth")
HTML_PATH  = os.path.join(_BASE, "..", "web", "webapp")

# Load trained weights once at startup
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print("\n✅ Checkpoint loaded — backend ready.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

CLASS_NAMES = ["FAKE", "REAL"]

app = FastAPI(title="VERIFY_AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100

    # Grad-CAM
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    rgb_img = np.array(img_resized).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    _, buf = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    heatmap_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(confidence, 2),
        "heatmap": heatmap_b64,
    }
