"""
CAPTCHA Recognition - Prediction/Inference Script
Simple prediction using existing model and dataset files
"""

import os
import glob
import torch
import numpy as np
import pickle
from sklearn import preprocessing

# Import from existing files
from model import CaptchaModel
from dataset import ClassificationDataset

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================


def remove_duplicates(x):
    """Remove consecutive duplicate characters"""
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    
    preds = preds.permute(1, 0, 2)  # batch, sequence_length, num_classes
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()

    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")  # Blank token placeholder
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))

    return cap_preds


def load_model(model_path, lbl_enc, device='cpu'):

    model = CaptchaModel(num_characters=len(lbl_enc.classes_))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_single_image(image_path, model, lbl_enc, device='cpu'):

    # Use ClassificationDataset for preprocessing (same as training)
    dataset = ClassificationDataset(
        image_paths=[image_path],
        labels=[[0, 0, 0, 0, 0]]  # Dummy labels, not used for prediction
    )

    # Get preprocessed image
    image_dict = dataset[0]
    image_tensor = image_dict['images'].unsqueeze(
        0).to(device)  # Add batch dimension

    # Predict
    with torch.no_grad():
        preds, _ = model(image_tensor)

    # Decode
    text = decode_predictions(preds, lbl_enc)[0]

    return text


def predict_batch(image_paths, model, lbl_enc, device='cpu', batch_size=32):

    all_predictions = []

    # Create dummy labels for the dataset
    dummy_labels = [[0, 0, 0, 0, 0] for _ in range(len(image_paths))]

    # Use ClassificationDataset for preprocessing
    dataset = ClassificationDataset(
        image_paths=image_paths,
        labels=dummy_labels
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Process batches
    for data in dataloader:
        images = data['images'].to(device)

        # Predict
        with torch.no_grad():
            preds, _ = model(images)

        # Decode
        texts = decode_predictions(preds, lbl_enc)
        all_predictions.extend(texts)

    return all_predictions


def save_label_encoder(lbl_enc, save_path='label_encoder.pkl'):
    """Save LabelEncoder for later use"""
    with open(save_path, 'wb') as f:
        pickle.dump(lbl_enc, f)
    print(f"Label encoder saved to {save_path}")


def load_label_encoder(load_path='label_encoder.pkl'):
    """Load saved LabelEncoder"""
    with open(load_path, 'rb') as f:
        lbl_enc = pickle.load(f)
    print(f"Label encoder loaded from {load_path}")
    return lbl_enc


if __name__ == "__main__":
    """
    Example usage for prediction
    """

    # Configuration
    MODEL_PATH = 'best_captcha_model.pth'
    LABEL_ENCODER_PATH = 'label_encoder.pkl'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("CAPTCHA PREDICTION")
    print("="*70)

    # ========== Load label encoder ==========
    try:
        lbl_enc = load_label_encoder(LABEL_ENCODER_PATH)
    except:
        print("Label encoder file not found. Creating from dataset...")

    # ========== Load model ==========
    print(f"\nLoading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, lbl_enc, DEVICE)
    print(f"Model loaded successfully on {DEVICE}")
    print(f"Number of character classes: {len(lbl_enc.classes_)}")

    # ========== EXAMPLE 1: Predict single image ==========
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*70)

    test_image = "Dataset/val/8AE5T.jpg"  # Change to your image path

    if os.path.exists(test_image):
        predicted_text = predict_single_image(
            test_image, model, lbl_enc, DEVICE)
        actual_text = test_image.split(os.sep)[-1].split('.')[0]

        print(f"\nImage: {test_image}")
        print(f"Predicted: {predicted_text}")
        print(f"Actual: {actual_text}")
        print(f"Match: {'✓' if predicted_text == actual_text else '✗'}")
    else:
        print(f"Test image not found: {test_image}")

    # ========== EXAMPLE 2: Predict multiple images ==========
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Prediction")
    print("="*70)

    test_dir = "Dataset/val"
    test_images = glob.glob(os.path.join(
        test_dir, '*.jpg'))[:10]  # First 10 images

    if len(test_images) > 0:
        print(f"\nPredicting {len(test_images)} images...")
        predictions = predict_batch(
            test_images, model, lbl_enc, DEVICE, batch_size=8)

        # Calculate accuracy
        correct = 0
        print("\nResults:")
        print("-" * 70)
        for img_path, pred_text in zip(test_images, predictions):
            actual_text = img_path.split(os.sep)[-1].split('.')[0]
            match = pred_text == actual_text
            if match:
                correct += 1

            print(
                f"Image: {os.path.basename(img_path):20s} | Pred: {pred_text:10s} | Actual: {actual_text:10s} | {'✓' if match else '✗'}")

        accuracy = (correct / len(test_images)) * 100
        print("-" * 70)
        print(f"Accuracy: {correct}/{len(test_images)} = {accuracy:.2f}%")
    else:
        print(f"No test images found in {test_dir}")

    # ========== EXAMPLE 3: Custom prediction ==========
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Image Prediction")
    print("="*70)
    print("\nTo predict your own image:")
    print("  prediction = predict_single_image('path/to/image.jpg', model, lbl_enc, DEVICE)")
    print("  print(f'Predicted: {prediction}')")
'''
Label encoder loaded from label_encoder.pkl

Loading model from best_captcha_model.pth...
Model loaded successfully on cpu
Number of character classes: 60

======================================================================
EXAMPLE 1: Single Image Prediction
======================================================================
Test image not found: Dataset/val/8AE5T.jpg

======================================================================
EXAMPLE 2: Batch Prediction
======================================================================

Predicting 10 images...

Results:
----------------------------------------------------------------------
Image: 8a3T3.jpg            | Pred: 8a3T3      | Actual: 8a3T3      | ✓
Image: 8a8kr.jpg            | Pred: 8d8k       | Actual: 8a8kr      | ✗
Image: 8aGBh.jpg            | Pred: 8aGBh      | Actual: 8aGBh      | ✓
Image: 8agUl.jpg            | Pred: 8ACUL      | Actual: 8agUl      | ✗
Image: 8aMUi.jpg            | Pred: 8AMUl      | Actual: 8aMUi      | ✗
Image: 8ar4T.jpg            | Pred: 8ar1T      | Actual: 8ar4T      | ✗
Image: 8arDi.jpg            | Pred: 8arDi      | Actual: 8arDi      | ✓
Image: 8aUUx.jpg            | Pred: 8aUX       | Actual: 8aUUx      | ✗
Image: 8aZhr.jpg            | Pred: 8aZhr      | Actual: 8aZhr      | ✓
Image: 8bAr6.jpg            | Pred: 8bAr6      | Actual: 8bAr6      | ✓
----------------------------------------------------------------------
Accuracy: 5/10 = 50.00%

======================================================================
EXAMPLE 3: Custom Image Prediction
======================================================================

To predict your own image:
  prediction = predict_single_image('path/to/image.jpg', model, lbl_enc, DEVICE)
  print(f'Predicted: {prediction}')

'''