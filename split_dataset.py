
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("split_dataset")

CLASSES = ["glass", "metal", "paper", "plastic"]
RANDOM_SEED = 42

for class_name in CLASSES:
    source_dir = DATASET_DIR / class_name

    images = sorted([
        file for file in source_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    train_files, temp_files = train_test_split(
        images,
        test_size=0.30,
        random_state=RANDOM_SEED
    )

    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=RANDOM_SEED
    )

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, files in splits.items():
        target_dir = OUTPUT_DIR / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            shutil.copy2(file, target_dir / file.name)

        print(f"{split_name}/{class_name}: {len(files)} images")

print("\n데이터셋 분할 완료!")
