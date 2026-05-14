from utils.data_loader import train_loader

if __name__ == "__main__":

    # Get one batch
    images, labels = next(iter(train_loader))

    print("\n===== BATCH INFO =====")
    print("Image batch shape:", images.shape)
    print("Labels shape:", labels.shape)

    print("\nPipeline working successfully!")