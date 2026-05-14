from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VALIDATION_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

# =========================
# TRAIN TRANSFORMS
# =========================
train_transforms = transforms.Compose([

    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# =========================
# VALIDATION / TEST
# =========================
test_transforms = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD DATASETS
# =========================
train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transforms
)

validation_dataset = datasets.ImageFolder(
    root=VALIDATION_DIR,
    transform=test_transforms
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transforms
)

# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# =========================
# PRINT DATASET INFO
# =========================
print("\n========== DATASET INFO ==========")
print("Classes:", train_dataset.classes)

print(f"\nTrain Images: {len(train_dataset)}")
print(f"Validation Images: {len(validation_dataset)}")
print(f"Test Images: {len(test_dataset)}")

print("==================================")