import os
import random
import shutil
import csv
from pathlib import Path
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'Dataset'
TRAIN_DIR = DATASET_DIR / 'train'
VAL_DIR = DATASET_DIR / 'val'
SET_SEED = 42
SPLIT_RATIO = 0.8
    
def run_dataset_creation():


    print(BASE_DIR)
    SOURCE_DIR = "C:\\Users\\LENOVO\\Desktop\\archive"
    print(os.path.exists(SOURCE_DIR))

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    df = []

    random.seed(SET_SEED)

    files = [f for f in os.listdir(SOURCE_DIR)]
    random.shuffle(files)

    split_index = int(len(files) * SPLIT_RATIO)
    train_files = files[:split_index]
    val_files = files[split_index:]

    for file in tqdm(train_files):
        file_path = os.path.join(SOURCE_DIR, file)
        # Assuming label is the prefix before the first dot
        label = file.split('.')[0]
        df.append({'filename': file, 'label': label})
        shutil.copy(os.path.join(SOURCE_DIR, file), TRAIN_DIR / file)
    for file in tqdm(val_files):
        file_path = os.path.join(SOURCE_DIR, file)
        # Assuming label is the prefix before the first dot
        label = file.split('.')[0]
        df.append({'filename': file, 'label': label})
        shutil.copy(os.path.join(SOURCE_DIR, file), VAL_DIR / file)

    df = pd.DataFrame(df)
    df.to_csv('file_labels.csv', index=False)
    print("Dataset creation completed.")

if __name__ == "__main__":
    run_dataset_creation()