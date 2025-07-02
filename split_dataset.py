import os
import shutil
import random
from pathlib import Path

# Set the path to the original dataset
original_dataset = Path("PlantVillage")  # change if your folder name is different
train_dir = Path("data/train")
test_dir = Path("data/test")

# Create folders if they don't exist
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# Set split ratio
split_ratio = 0.8

# Loop through all class folders
for class_folder in original_dataset.iterdir():
    if class_folder.is_dir():
        images = list(class_folder.glob("*.jpg"))  # or .png depending on your files
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]

        # Create class subfolders
        (train_dir / class_folder.name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_folder.name).mkdir(parents=True, exist_ok=True)

        # Copy images to train
        for img in train_images:
            shutil.copy(img, train_dir / class_folder.name / img.name)

        # Copy images to test
        for img in test_images:
            shutil.copy(img, test_dir / class_folder.name / img.name)

print("✅ Dataset split and saved into 'data/train' and 'data/test'")
