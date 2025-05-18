import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import cv2
from tensorflow.keras.models import load_model
from models.autoencoder import load_autoencoder_model, detect_anomalies
from models.cnn_classifier import classify_images
from utils.parquet_loader import load_and_preprocess_parquet
from utils.image_converter import convert_to_images

def display_image(image_path):
    img = cv2.imread(image_path)
    st.image(img, caption="Generated Image", use_column_width=True)

st.set_page_config(page_title="Network Anomaly Classifier", layout="wide")
st.title("Network Packet Anomaly Detection & Classification")

uploaded_file = st.file_uploader("Upload a Parquet file", type=["parquet"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
        temp_file.write(uploaded_file.read())
        parquet_path = temp_file.name
    
    st.subheader("Loading and preprocessing data...")

    features_scaled, labels, full_df, scaler = load_and_preprocess_parquet(
        parquet_paths=[parquet_path],
        train_on_benign_only=False
    )

    st.write(features_scaled)
    
    input_dim = features_scaled.shape[1]
    st.write(f"Loaded {len(features_scaled)} samples with {input_dim} features each.")

    st.subheader("Running anomaly detection using autoencoder...")

    autoencoder_model = load_autoencoder_model("models_saved/simple_autoencoder.pth", input_dim)
    errors, anomaly_flags = detect_anomalies(autoencoder_model, features_scaled)

    total_anomalies = np.sum(anomaly_flags)
    st.write(f"Detected {total_anomalies} anomalies (out of {len(anomaly_flags)} samples)")

    if total_anomalies == 0:
        st.warning("No anomalies detected. Nothing to classify.")
        st.stop()

    st.subheader("Converting anomalous packets to RGB images...")

    anomalous_samples = features_scaled[anomaly_flags == 1]
    st.write(anomalous_samples)
    df_anomalies = pd.DataFrame(anomalous_samples)

    st.write(df_anomalies.info())

    label_name = "Anomaly"
    feature_count = 77 

    if label == 'Benign':
        convert_to_images(df_anomalies, 'Benign', feature_count)
    else:
        convert_to_images(df_anomalies, label, feature_count)
    
    feature_count = features.shape[1]
    
    if st.button("Generate Image from Anomalies"):
        for path in image_paths:
            display_image(path)
            
    st.success("Images created in 'data/converted_images/Anomaly/'")

    st.subheader("Classifying anomaly images using CNN...")

    cnn_model = load_model("models_saved/cnn_model.h5")
    results = classify_images(cnn_model, image_dir="data/converted_images/Anomaly")

    st.subheader("Classification Results")

    if len(results) == 0:
        st.write("NO!")
    else:
        for image_name, predicted_class in results.items():
            st.write(f"`{image_name}` → Prediction: **{predicted_class}**")
