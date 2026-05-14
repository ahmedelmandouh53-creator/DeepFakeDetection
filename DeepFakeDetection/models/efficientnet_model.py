import torch
import torch.nn as nn
from torchvision import models

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# LOAD MODEL
model = models.resnet50(weights="DEFAULT")

# FREEZE
for param in model.parameters():
    param.requires_grad = False

# UNFREEZE LAST BLOCK
for param in model.layer4.parameters():
    param.requires_grad = True

# CLASSIFIER
model.fc = nn.Sequential(

    nn.Linear(
        model.fc.in_features,
        512
    ),

    nn.BatchNorm1d(512),

    nn.ReLU(),

    nn.Dropout(0.5),

    nn.Linear(512, 2)
)

# MOVE TO DEVICE
model = model.to(device)

# LOSS
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

# OPTIMIZER
optimizer = torch.optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=0.00005,
    weight_decay=1e-4
)

# SCHEDULER
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode='max',

    factor=0.5,

    patience=2
)

print("\n✅ Anti-Overfitting ResNet50 Ready!")