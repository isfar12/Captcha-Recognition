# CAPTCHA Recognition System

A deep learning-based CAPTCHA recognition system using PyTorch that achieves automated text extraction from CAPTCHA images. This project implements a Convolutional Neural Network (CNN) combined with Gated Recurrent Units (GRU) and Connectionist Temporal Classification (CTC) loss for sequence recognition.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Web UI](#-web-ui)
- [Demo](#-demo)
- [Results](#-results)
- [Technical Details](#-technical-details)
- [Requirements](#-requirements)
- [License](#-license)

## 🎯 Overview

This project tackles the challenge of automatically recognizing text in CAPTCHA images. The system uses a custom deep learning architecture that combines:

- **CNNs** for feature extraction from images
- **Bidirectional GRU** for sequence modeling
- **CTC Loss** for alignment-free sequence prediction

The model can recognize alphanumeric characters (both uppercase/lowercase letters and digits) from CAPTCHA images.

## ✨ Features

- **End-to-end Pipeline**: Complete workflow from dataset creation to model training and inference
- **Flexible Architecture**: Custom CNN-RNN hybrid model with configurable parameters
- **Batch Prediction**: Efficient batch processing for multiple images
- **Data Augmentation**: Built-in image preprocessing and normalization
- **Model Persistence**: Save and load trained models with label encoders
- **Kaggle Compatible**: Ready-to-run version for Kaggle notebooks
- **Easy Inference**: Simple API for predicting single or multiple images

## 🏗️ Model Architecture

### Network Components

1. **Convolutional Layers**

   - Conv2D (3 → 128 channels, kernel=3×3, padding=1)
   - MaxPool2D (2×2)
   - Conv2D (128 → 64 channels, kernel=3×3, padding=1)
   - MaxPool2D (2×2)

2. **Feature Processing**

   - Linear transformation (640 → 64)
   - Dropout (0.3)

3. **Sequential Modeling**

   - Bidirectional GRU (2 layers, hidden_size=32, dropout=0.3)
   - Output: 64 features (32×2 directions)

4. **Classification**
   - Linear layer to num_characters + 1 (blank token)
   - CTC Loss for training

### Architecture Flow

```
Input Image (3×40×150)
    ↓
CNN Feature Extraction
    ↓
Permute & Reshape (sequence formation)
    ↓
Linear + Dropout
    ↓
Bidirectional GRU (2 layers)
    ↓
Classifier
    ↓
CTC Loss / Prediction
```

## 📊 Dataset

- **Total Images**: 98,000 CAPTCHA images
- **Split Ratio**: 80% training, 20% validation
- **Image Format**: JPG
- **Image Size**: Resized to 40×150 pixels
- **Character Set**: Alphanumeric (digits 0-9, uppercase and lowercase letters A-Z, a-z)
- **Filename Convention**: Image filename is the CAPTCHA label (e.g., `8AE5T.jpg`)

### Dataset Structure

```
Dataset/
├── train/          # Training images (~78,400 images)
├── val/            # Validation images (~19,600 images)
└── file_labels.csv # Metadata file
```

## 🚀 Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd "Captcha Recognition"
```

2. Install required packages:

```bash
pip install torch torchvision
pip install albumentations
pip install scikit-learn
pip install Pillow
pip install numpy pandas
pip install tqdm
```

## 📁 Project Structure

```
Captcha Recognition/
├── dataset_creation.py      # Script to split and organize dataset
├── dataset.py              # PyTorch Dataset class with transformations
├── model.py                # CNN-GRU model architecture
├── engine.py               # Training and evaluation functions
├── train.py                # Main training script
├── predict.py              # Inference script with examples
├── best_captcha_model.pth  # Saved trained model weights
├── label_encoder.pkl       # Saved label encoder for character mapping
├── Dataset/                # Dataset directory
│   ├── train/             # Training images
│   ├── val/               # Validation images
│   └── file_labels.csv    # Labels metadata
├── view_data.ipynb         # Notebook for data exploration
├── Kaggle_version_Result.ipynb  # Kaggle-compatible version
└── CAPTCHA_Note_GPT_Made.ipynb  # Documentation notebook
```

## 💻 Usage

### 1. Dataset Preparation

```python
from dataset_creation import run_dataset_creation

# Prepare the dataset from source directory
run_dataset_creation()
```

Modify the `SOURCE_DIR` in `dataset_creation.py` to point to your CAPTCHA images.

### 2. Training

```bash
python train.py
```

Training configuration:

- Batch size: 32
- Epochs: 10 (configurable)
- Learning rate: 0.001
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

### 3. Prediction

#### Single Image Prediction

```python
import torch
from predict import load_model, load_label_encoder, predict_single_image

# Load model and encoder
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lbl_enc = load_label_encoder('label_encoder.pkl')
model = load_model('best_captcha_model.pth', lbl_enc, DEVICE)

# Predict
prediction = predict_single_image('path/to/image.jpg', model, lbl_enc, DEVICE)
print(f'Predicted: {prediction}')
```

#### Batch Prediction

```python
from predict import predict_batch

# Predict multiple images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
predictions = predict_batch(image_paths, model, lbl_enc, DEVICE, batch_size=8)
```

#### Run Prediction Examples

```bash
python predict.py
```

## 🌐 Web UI

A user-friendly web interface is provided using [Streamlit](https://streamlit.io/). This allows you to upload a CAPTCHA image and get instant predictions from the trained model.

### How to Run the Web UI

1. Make sure you have installed all requirements (see Installation section).
2. From the project root, run:

```bash
streamlit run WebUI/streamlit.py
```

3. The app will open in your browser. Upload a CAPTCHA image to see the prediction.

**Note:** If you run from the root, use the path above. If you run from inside the `WebUI` folder, use:

```bash
streamlit run streamlit.py
```

## 🖼️ Demo

Below are example screenshots of the Web UI in action:

|      Prediction1 CAPTCHA       |       Prediction2 Result       |
| :----------------------------: | :----------------------------: |
| ![Upload Demo](Demo/demo1.png) | ![Result Demo](Demo/demo2.png) |

## 📈 Results

### Model Performance

The trained model demonstrates the following performance on sample validation images:

```
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
```

### Performance Notes

- The model achieves 50% exact match accuracy on this sample
- Common challenges include:
  - Similar-looking characters (e.g., 'a' vs 'A', '1' vs 'l', '0' vs 'O')
  - Repeated characters (e.g., 'UU' predicted as 'U')
  - Case sensitivity confusion
- Model can be improved with:
  - More training epochs
  - Data augmentation techniques
  - Larger training dataset
  - Fine-tuning hyperparameters

## 🔧 Technical Details

### Image Preprocessing

1. **Resize**: Images resized to 40×150 pixels
2. **Normalization**:
   - Mean: (0.485, 0.456, 0.406)
   - Std: (0.229, 0.224, 0.225)
3. **Format**: Converted to tensor with shape (3, 40, 150)

### Label Encoding

- Characters encoded using `sklearn.preprocessing.LabelEncoder`
- Encoded values shifted by +1 (reserve 0 for CTC blank token)
- Each CAPTCHA label split into individual characters
- Example: "8AE5T" → ['8', 'A', 'E', '5', 'T'] → [encoded_values]

### CTC Decoding

The prediction pipeline includes:

1. Softmax activation on model outputs
2. Argmax to get predicted character indices
3. CTC blank token removal
4. Duplicate character removal (consecutive duplicates)

### Training Strategy

- **Loss Function**: CTC Loss (Connectionist Temporal Classification)
- **Optimizer**: Adam with learning rate 0.001
- **Scheduler**: ReduceLROnPlateau for adaptive learning rate
- **Best Model**: Saved based on validation loss

## 📦 Requirements

### Core Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
albumentations>=1.0.0
scikit-learn>=0.24.0
Pillow>=8.0.0
numpy>=1.19.0
pandas>=1.2.0
tqdm>=4.60.0
```

### Hardware Requirements

- **Training**: GPU recommended (CUDA-capable)
- **Inference**: Can run on CPU
- **Memory**: Minimum 8GB RAM

## 🎓 Key Concepts

### Why CTC Loss?

CTC (Connectionist Temporal Classification) is ideal for sequence recognition tasks where:

- Input and output lengths differ
- Alignment between input and output is unknown
- No need for character-level annotations

### Model Design Choices

1. **CNN**: Extracts spatial features from CAPTCHA images
2. **GRU**: Models sequential dependencies between characters
3. **Bidirectional**: Captures context from both left and right
4. **Dropout**: Prevents overfitting

## 🚧 Future Improvements

- [ ] Implement additional data augmentation (rotation, noise, distortion)
- [ ] Experiment with attention mechanisms
- [ ] Try different architectures (ResNet, EfficientNet backbones)
- [ ] Implement ensemble methods
- [ ] Add character-level confidence scores
- [ ] Support for variable-length CAPTCHAs
- [ ] Real-time inference optimization
- [ ] Web API deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- Dataset: CAPTCHA Dataset (98K images)
- Framework: PyTorch
- Inspiration: CTC-based OCR systems

---

**Note**: This project is for educational purposes. Using automated CAPTCHA solvers to bypass security measures may violate terms of service and is discouraged.

## 📞 Contact

For questions or feedback, please open an issue in the repository.

---

_Last Updated: January 7, 2026_
