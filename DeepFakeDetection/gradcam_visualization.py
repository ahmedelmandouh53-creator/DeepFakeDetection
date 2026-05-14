import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

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
# LOAD TRAINED MODEL
# =========================
model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()

print("\n✅ Best Model Loaded!")

# =========================
# IMAGE PATH
# =========================
IMAGE_PATH = "sample.jpg"

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
# LOAD IMAGE
# =========================
image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_resized = image.resize((224, 224))

input_tensor = transform(image).unsqueeze(0)

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

print(
    f"\nPrediction: {predicted_class}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

# =========================
# TARGET LAYER
# =========================
target_layers = [model.layer4[-1]]

# =========================
# GRAD-CAM
# =========================
cam = GradCAM(
    model=model,
    target_layers=target_layers
)

grayscale_cam = cam(
    input_tensor=input_tensor
)[0]

# =========================
# ORIGINAL IMAGE
# =========================
rgb_image = np.array(
    image_resized
).astype(np.float32) / 255.0

# =========================
# OVERLAY HEATMAP
# =========================
visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)

# =========================
# SAVE RESULTS
# =========================
os.makedirs("outputs", exist_ok=True)

output_path = (
    "outputs/gradcam_result.jpg"
)

cv2.imwrite(
    output_path,
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)

# =========================
# SHOW IMAGE
# =========================
plt.figure(figsize=(10, 5))

plt.imshow(visualization)

plt.title(
    f"{predicted_class} "
    f"({confidence:.2f}%)"
)

plt.axis("off")

plt.show()

print(
    "\n✅ Grad-CAM saved!"
)

#pip install grad-cam opencv-python