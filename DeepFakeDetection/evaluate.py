import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tqdm import tqdm

from utils.data_loader import test_loader

from models.efficientnet_model import model, device

# =========================
# LOAD BEST MODEL
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
# STORE RESULTS
# =========================
all_labels = []
all_predictions = []

# =========================
# TEST LOOP
# =========================
with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )

# =========================
# METRICS
# =========================
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

# =========================
# PRINT RESULTS
# =========================
print("\n========== FINAL RESULTS ==========")

print(f"\nAccuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1-Score  : {f1*100:.2f}%")

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(
    all_labels,
    all_predictions,
    target_names=["Fake", "Real"]
)

print("\n========== CLASSIFICATION REPORT ==========\n")

print(report)

# =========================
# SAVE REPORT
# =========================
os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/classification_report.txt",
    "w"
) as f:

    f.write(report)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(
    all_labels,
    all_predictions
)

# =========================
# PLOT CONFUSION MATRIX
# =========================
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

print(
    "\n✅ Confusion Matrix Saved!"
)

print(
    "✅ Classification Report Saved!"
)
