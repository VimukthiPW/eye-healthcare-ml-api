import os
import shutil
import random
from pathlib import Path

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "eye_validation_dataset"
OUTPUT_DIR = BASE_DIR / "eye_validation_split"

# -------------------------------------------------
# Split ratios
# -------------------------------------------------

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

random.seed(42)

# -------------------------------------------------
# Create output folders
# -------------------------------------------------

for split in ["train", "val", "test"]:
    for category in ["eye", "non_eye"]:
        folder = OUTPUT_DIR / split / category
        folder.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Split images
# -------------------------------------------------

for category in ["eye", "non_eye"]:

    source_folder = SOURCE_DIR / category

    # Get all image files including subfolders
    images = []

    for file in source_folder.rglob("*"):
        if file.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ]:
            images.append(file)

    print(f"\n{category}: {len(images)} images found")

    # Shuffle
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Copy images
    for image in train_images:
        destination = OUTPUT_DIR / "train" / category / image.name
        shutil.copy2(image, destination)

    for image in val_images:
        destination = OUTPUT_DIR / "val" / category / image.name
        shutil.copy2(image, destination)

    for image in test_images:
        destination = OUTPUT_DIR / "test" / category / image.name
        shutil.copy2(image, destination)

    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test       : {len(test_images)}")

print("\nDataset splitting completed successfully! ✅")