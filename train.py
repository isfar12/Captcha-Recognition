import os
import glob
import torch
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm

import dataset_creation
import dataset
from model import CaptchaModel
import engine

def run_traininig():
    # Load image paths
    train_images = glob.glob(os.path.join(dataset_creation.TRAIN_DIR, '*.jpg'))
    train_targets_orig = [x.split('\\')[-1].split('.')[0] for x in train_images]

    test_images = glob.glob(os.path.join(dataset_creation.VAL_DIR, '*.jpg'))
    test_targets_orig = [x.split('\\')[-1].split('.')[0] for x in test_images]

    # Split each captcha string into individual characters
    # e.g., "8AE5T" becomes ['8', 'A', 'E', '5', 'T']
    train_targets = [[c for c in x] for x in train_targets_orig]
    test_targets = [[c for c in x] for x in test_targets_orig]
    
    # Flatten all characters from all captchas into a single list
    # This ensures LabelEncoder sees all unique characters
    targets_flat = [c for clist in train_targets for c in clist]
    targets_flat += [c for clist in test_targets for c in clist]
    
    # Fit LabelEncoder on individual characters (not whole strings)
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
        # Save label encoder for later use in prediction
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(lbl_enc, f)
    print(f"Label encoder saved to 'label_encoder.pkl'")
        # print(f"Number of unique characters: {len(lbl_enc.classes_)}")
    # print(f"Characters: {lbl_enc.classes_}")
    
    # Encode each character in each captcha
    # e.g., ['8', 'A', 'E', '5', 'T'] becomes [10, 2, 5, 9, 15]
    train_targets_enc = [lbl_enc.transform(x) for x in train_targets]
    train_targets_enc = np.array(train_targets_enc)
    train_targets_enc = train_targets_enc + 1  # Add 1 to reserve 0 for blank/padding
    
    test_targets_enc = [lbl_enc.transform(x) for x in test_targets]
    test_targets_enc = np.array(test_targets_enc)
    test_targets_enc = test_targets_enc + 1  # Add 1 to reserve 0 for blank/padding

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images,
        labels=train_targets_enc
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_images,
        labels=test_targets_enc
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    model =CaptchaModel(num_characters=len(lbl_enc.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
    )
    EPOCHS = 10
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss = engine.train_fn(
            model,
            train_loader,
            optimizer,
            device
        )
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss}")
        predictions, true_labels, val_loss = engine.eval_fn(
            model,
            test_loader,
            device
        )
        print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss}")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_captcha_model.pth')
            print(f"   ✓ Best model saved (Val Loss: {val_loss:.4f})")
            
if __name__ == "__main__":
    run_traininig()
