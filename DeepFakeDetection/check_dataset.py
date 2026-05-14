import os

base_path = "DeepFakeDetection/dataset"

for split in ["train", "validation", "test"]:

    print(f"\n===== {split.upper()} =====")

    split_path = os.path.join(base_path, split)

    for cls in os.listdir(split_path):

        class_path = os.path.join(split_path, cls)

        if os.path.isdir(class_path):

            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            print(f"{cls}: {len(images)} images")