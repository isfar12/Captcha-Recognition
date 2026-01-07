from model import CaptchaModel
from dataset import ClassificationDataset
from predict import load_label_encoder, decode_predictions
from dataset_creation import BASE_DIR
import sys
import os
import streamlit as st
import torch
from PIL import Image
import numpy as np
import albumentations as A

# Ensure parent directory is on sys.path before local imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


# Page configuration
st.set_page_config(
    page_title="CAPTCHA Solver",
    layout="centered"
)

# Custom CSS for light background and styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        background-color: #f5f7fa;
    }
    .upload-text {
        text-align: center;
        color: #4a5568;
        font-size: 18px;
        margin: 20px 0;
    }
    .result-box {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 20px 0;
    }
    .prediction-text {
        font-size: 36px;
        font-weight: bold;
        color: #2d3748;
        letter-spacing: 8px;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        MODEL_PATH = os.path.join(BASE_DIR, 'best_captcha_model.pth')
        ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lbl_enc = load_label_encoder(ENCODER_PATH)

        model = CaptchaModel(num_characters=len(lbl_enc.classes_))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        return model, lbl_enc, device

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def predict_captcha(image, model, lbl_enc, device):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        dataset = ClassificationDataset(
            image_paths=[],
            labels=[[0, 0, 0, 0, 0]]
        )

        if dataset.transform is None:
            dataset.transform = A.Compose([
                A.Resize(40, 150),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ])

        image_np = np.array(image)
        transformed = dataset.transform(image=image_np)
        image_tensor = transformed["image"]
        image_tensor = np.transpose(image_tensor, (2, 0, 1)).astype(np.float32)
        image_tensor = torch.tensor(
            image_tensor, dtype=torch.float).unsqueeze(0).to(device)

        with torch.no_grad():
            preds, _ = model(image_tensor)

        text = decode_predictions(preds, lbl_enc)[0]
        return text

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


def main():

    st.markdown("<h1 style='text-align: center; color: #2d3748;'>CAPTCHA Solver</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #718096; font-size: 16px;'>Upload a CAPTCHA image to decode the text</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner('Loading model...'):
        model, lbl_enc, device = load_model()

    if model is None:
        st.error(
            "Failed to load model. Please ensure model files exist in the correct location.")
        return

    uploaded_file = st.file_uploader(
        "Choose a CAPTCHA image",
        type=["png", "jpg", "jpeg"],
        help="Upload a CAPTCHA image (PNG, JPG, or JPEG format)"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Uploaded CAPTCHA',
                     use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Predict button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔍 Decode CAPTCHA", use_container_width=True):
                with st.spinner('Analyzing...'):
                    prediction = predict_captcha(image, model, lbl_enc, device)

                if prediction:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class='result-box'>
                            <p style='color: #718096; font-size: 14px; margin-bottom: 10px;'>PREDICTED TEXT</p>
                            <p class='prediction-text'>{prediction}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("CAPTCHA decoded successfully!")
    else:
        # Instructions
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("Upload a CAPTCHA image to get started")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #a0aec0; font-size: 12px;'>Built with PyTorch & Streamlit | CNN-GRU Model</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
