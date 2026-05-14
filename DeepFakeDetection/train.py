import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.data_loader import (
    train_loader,
    validation_loader
)

from models.efficientnet_model import (
    model,
    criterion,
    optimizer,
    scheduler,
    device
)

# =========================
# SETTINGS
# =========================
MAX_EPOCHS = 50
PATIENCE = 5

# =========================
# CREATE FOLDERS
# =========================
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================
# TRACKING LISTS
# =========================
train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

# =========================
# BEST MODEL TRACKING
# =========================
best_val_accuracy = 0.0
best_epoch = 0

epochs_without_improvement = 0

# =========================
# TRAINING LOOP
# =========================
for epoch in range(MAX_EPOCHS):

    print(f"\n========== EPOCH {epoch+1}/{MAX_EPOCHS} ==========")

    # =========================
    # TRAIN MODE
    # =========================
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # =========================
    # TRAIN LOOP
    # =========================
    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

    # =========================
    # TRAIN RESULTS
    # =========================
    avg_train_loss = (
        train_loss / len(train_loader)
    )

    train_accuracy = (
        100 * train_correct / train_total
    )

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # =========================
    # VALIDATION MODE
    # =========================
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in tqdm(validation_loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    # =========================
    # VALIDATION RESULTS
    # =========================
    avg_val_loss = (
        val_loss / len(validation_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    # =========================
    # LR SCHEDULER
    # =========================
    scheduler.step(val_accuracy)

    # =========================
    # PRINT RESULTS
    # =========================
    print(f"\nTrain Loss: {avg_train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    print(f"\nValidation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # =========================
    # OVERFITTING CHECK
    # =========================
    gap = train_accuracy - val_accuracy

    print(f"\nGeneralization Gap: {gap:.2f}%")

    if gap > 10:

        print(
            "\n⚠️ Possible Overfitting Detected!"
        )

    # =========================
    # SAVE BEST MODEL
    # =========================
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )

        print("\n✅ NEW BEST MODEL SAVED!")

        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    # =========================
    # EARLY STOPPING
    # =========================
    if epochs_without_improvement >= PATIENCE:

        print("\n🛑 MODEL CONVERGED")
        print("Early stopping activated!")

        break

# =========================
# FINAL RESULTS
# =========================
print("\n🏆 TRAINING FINISHED")

print(
    f"\n🔥 Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"🔥 Best Epoch: {best_epoch}"
)

# =========================
# LOSS CURVE
# =========================
plt.figure(figsize=(12, 6))

plt.plot(
    train_losses,
    marker='o',
    linewidth=2,
    label="Train Loss"
)

plt.plot(
    val_losses,
    marker='o',
    linewidth=2,
    label="Validation Loss"
)

plt.axvline(
    x=best_epoch - 1,
    linestyle='--',
    linewidth=2,
    label=f"Best Epoch = {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Convergence Curve - Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/loss_curve.png",
    dpi=300,
    bbox_inches='tight'
)

# =========================
# ACCURACY CURVE
# =========================
plt.figure(figsize=(12, 6))

plt.plot(
    train_accuracies,
    marker='o',
    linewidth=2,
    label="Train Accuracy"
)

plt.plot(
    val_accuracies,
    marker='o',
    linewidth=2,
    label="Validation Accuracy"
)

plt.axvline(
    x=best_epoch - 1,
    linestyle='--',
    linewidth=2,
    label=f"Best Epoch = {best_epoch}"
)

plt.scatter(
    best_epoch - 1,
    best_val_accuracy,
    s=200,
    label=(
        f"Best Accuracy = "
        f"{best_val_accuracy:.2f}%"
    )
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title("Convergence Curve - Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/accuracy_curve.png",
    dpi=300,
    bbox_inches='tight'
)

print(
    "\n📊 Convergence curves saved "
    "in outputs/"
)