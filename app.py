# Indian Food Classifier
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os

# ─── Page Config ───
st.set_page_config(
    page_title="Indian Food Classifier",
    page_icon="🍛",
    layout="centered"
)

# ─── Global Styles ───
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* ── Animated gradient background ── */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e) !important;
        background-size: 400% 400% !important;
        animation: gradientShift 15s ease infinite !important;
        font-family: 'Outfit', sans-serif;
        min-height: 100vh;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Floating particles effect via pseudo-element ── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image:
            radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.15), transparent),
            radial-gradient(2px 2px at 40% 70%, rgba(255,255,255,0.1), transparent),
            radial-gradient(1px 1px at 60% 20%, rgba(255,255,255,0.12), transparent),
            radial-gradient(2px 2px at 80% 50%, rgba(255,255,255,0.08), transparent),
            radial-gradient(1px 1px at 10% 80%, rgba(255,255,255,0.1), transparent),
            radial-gradient(1px 1px at 90% 10%, rgba(255,255,255,0.12), transparent);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Header ── */
    .page-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f5af19, #f12711, #f5af19);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%   { background-position: 0% center; }
        50%  { background-position: 100% center; }
        100% { background-position: 0% center; }
    }
    .page-subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 300;
        color: rgba(255,255,255,0.55);
        text-align: center;
        margin-bottom: 40px;
        letter-spacing: 0.5px;
    }

    /* ── Glassmorphism Cards ── */
    .card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 32px 28px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .card:hover {
        background: rgba(255,255,255,0.09);
        border-color: rgba(255,255,255,0.18);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.4);
        margin-bottom: 16px;
    }

    /* ── Classify button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f12711, #f5af19) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 14px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 2rem !important;
        letter-spacing: 0.8px !important;
        transition: all 0.3s ease !important;
        margin-top: 12px !important;
        box-shadow: 0 4px 20px rgba(241,39,17,0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(241,39,17,0.5) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
    }

    /* ── Result card ── */
    .result-card {
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
        animation: fadeUp 0.5s ease-out;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.35);
        margin-bottom: 10px;
    }
    .result-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f5af19, #f12711);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 6px;
    }
    .result-confidence {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        font-weight: 400;
        color: rgba(255,255,255,0.5);
    }

    /* ── Confidence bar ── */
    .conf-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        height: 10px;
        margin-top: 20px;
        overflow: hidden;
    }
    .conf-bar-fill {
        background: linear-gradient(90deg, #f12711, #f5af19);
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-shadow: 0 0 12px rgba(245,175,25,0.4);
    }

    /* ── Scores Row ── */
    .scores-row {
        display: flex;
        justify-content: center;
        gap: 56px;
        margin-top: 22px;
    }
    .score-item {
        text-align: center;
    }
    .score-name {
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 500;
        color: rgba(255,255,255,0.35);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .score-pct {
        font-family: 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: rgba(255,255,255,0.8);
        margin-top: 4px;
    }

    /* ── Steps (How it works) ── */
    .steps-row {
        display: flex;
        justify-content: center;
        gap: 28px;
        margin-top: 16px;
    }
    .step {
        text-align: center;
        flex: 1;
        max-width: 160px;
        padding: 16px 8px;
        border-radius: 16px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        transition: all 0.3s ease;
    }
    .step:hover {
        background: rgba(255,255,255,0.08);
        transform: translateY(-3px);
    }
    .step-num {
        font-family: 'Outfit', sans-serif;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #f12711, #f5af19);
        color: #fff;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(241,39,17,0.3);
    }
    .step-text {
        font-family: 'Outfit', sans-serif;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.5);
        line-height: 1.5;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.2);
        margin-top: 50px;
        padding-bottom: 24px;
        letter-spacing: 1px;
    }
    .footer-accent {
        color: rgba(245,175,25,0.5);
        font-weight: 600;
    }

    /* ── Hide Streamlit default elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Style the file uploader ── */
    .stFileUploader {
        font-family: 'Outfit', sans-serif !important;
    }
    .stFileUploader > div {
        border: 2px dashed rgba(255,255,255,0.15) !important;
        border-radius: 16px !important;
        padding: 28px !important;
        background: rgba(255,255,255,0.04) !important;
        transition: all 0.3s ease !important;
    }
    .stFileUploader > div:hover {
        border-color: rgba(245,175,25,0.5) !important;
        background: rgba(255,255,255,0.06) !important;
    }
    .stFileUploader label {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 400 !important;
        color: rgba(255,255,255,0.5) !important;
    }
    .stFileUploader small {
        color: rgba(255,255,255,0.3) !important;
    }

    /* ── Info / Warning boxes ── */
    .stAlert {
        border-radius: 14px !important;
        font-family: 'Outfit', sans-serif !important;
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }

    /* ── Image container ── */
    .stImage {
        border-radius: 16px !important;
        overflow: hidden !important;
    }
    .stImage img {
        border-radius: 16px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        color: rgba(255,255,255,0.5) !important;
    }

    /* ── Branded divider ── */
    .brand-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(241,39,17,0.4), rgba(245,175,25,0.4), transparent);
        border-radius: 2px;
        margin: 8px auto 32px;
        width: 120px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper ───
def is_likely_food_image(image_array):
    img_array = np.array(image_array)
    return len(img_array.shape) == 3 and np.std(img_array) > 20


# ─── Header ───
st.markdown('<div class="page-title">Indian Food Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Identify North Indian & South Indian cuisine with deep learning</div>', unsafe_allow_html=True)

# ─── Upload Section ───
uploaded_file = st.file_uploader(
    "Browse or drag & drop an image (JPG, JPEG, PNG)",
    type=['jpg', 'jpeg', 'png'],
)

# ─── Uploaded Preview + Classify ───
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if not is_likely_food_image(image):
        st.warning("This image may not contain food.")

    # Classify button
    classify_clicked = st.button("✦  Classify Cuisine", type="primary", use_container_width=True)

    if classify_clicked:
        try:
            model = keras.models.load_model("model6_densenet.h5")

            img = Image.open(uploaded_file).resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, 0)

            with st.spinner("Analyzing your dish…"):
                pred = model.predict(arr, verbose=0)[0][0]

            north = (1 - pred) * 100
            south = pred * 100
            conf = max(north, south)

            # Confidence threshold — if uncertain, it's likely not Indian food
            CONFIDENCE_THRESHOLD = 70.0

            if conf < CONFIDENCE_THRESHOLD:
                # Not Indian Cuisine
                st.markdown("""
                <div class="result-card" style="border-color: rgba(255,80,80,0.25);">
                    <div class="result-label">Classification Result</div>
                    <div class="result-value" style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Not Indian Cuisine</div>
                    <div class="result-confidence">The uploaded image does not appear to be Indian food</div>
                    <div style="margin-top:16px; font-size:0.85rem; color:rgba(255,255,255,0.35);">
                        Model confidence was only {conf:.1f}% — too low to classify
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                if north > south:
                    label = "North Indian"
                else:
                    label = "South Indian"

                # Result card
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Classification Result</div>
                    <div class="result-value">{label}</div>
                    <div class="result-confidence">{conf:.1f}% confidence</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{conf:.1f}%"></div>
                    </div>
                    <div class="scores-row">
                        <div class="score-item">
                            <div class="score-name">North Indian</div>
                            <div class="score-pct">{north:.1f}%</div>
                        </div>
                        <div class="score-item">
                            <div class="score-name">South Indian</div>
                            <div class="score-pct">{south:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Model error — {e}")

else:
    # How it works (shown when no image uploaded)
    st.markdown("""
    <div class="card">
        <div class="card-title">How it works</div>
        <div class="steps-row">
            <div class="step">
                <div class="step-num">1</div>
                <div class="step-text">Upload an image of Indian food</div>
            </div>
            <div class="step">
                <div class="step-num">2</div>
                <div class="step-text">DenseNet-121 analyzes the image</div>
            </div>
            <div class="step">
                <div class="step-num">3</div>
                <div class="step-text">Get instant classification & confidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───
st.markdown('<div class="footer">Powered by <span class="footer-accent">DenseNet-121</span> · Indian Food Classification</div>', unsafe_allow_html=True)
