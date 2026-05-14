import streamlit as st
import torch
import numpy as np
import cv2

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from models.efficientnet_model import (
    model,
    device
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="DeepFake Detection AI",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("🧠 DeepFake Detection System")

st.write(
    "Upload an image to detect "
    "whether it is REAL or FAKE."
)

# =========================
# LOAD MODEL
# =========================
model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROCESS IMAGE
# =========================
if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # =========================
    # PREPROCESS
    # =========================
    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # =========================
    # PREDICTION
    # =========================
    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    prediction = prediction.item()

    confidence = confidence.item() * 100

    class_names = ["Fake", "Real"]

    predicted_class = class_names[prediction]

    # =========================
    # SHOW RESULTS
    # =========================
    st.subheader("Prediction")

    st.write(
        f"### {predicted_class}"
    )

    st.write(
        f"Confidence: "
        f"{confidence:.2f}%"
    )

    # =========================
    # GRAD-CAM
    # =========================
    target_layers = [model.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    image_resized = image.resize((224, 224))

    rgb_image = np.array(
        image_resized
    ).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # =========================
    # SHOW HEATMAP
    # =========================
    st.subheader("Grad-CAM Heatmap")

    st.image(
        visualization,
        caption="AI Attention Map",
        use_container_width=True
    )

    #pip install streamlit pytorch-grad-cam opencv-python