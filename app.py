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
from collections import Counter

class_labels = {'DrDoS_DNS': 0, 'DrDoS_LDAP': 1, 'DrDoS_MSSQL': 2, 'DrDoS_NTP': 3, 'DrDoS_NetBIOS': 4, 'DrDoS_SNMP': 5,
           'DrDoS_UDP': 6, 'LDAP': 7, 'MSSQL': 8, 'NetBIOS': 9, 'Portmap': 10,
           'Syn': 11, 'TFTP': 12, 'UDP': 13, 'UDP-lag': 14, 'WebDDoS': 15}

st.set_page_config(page_title="Network Anomaly Classifier", layout="wide")
st.title("Network Packet Anomaly Detection & Classification")

uploaded_file = st.file_uploader("Upload a Parquet file", type=["parquet"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
        temp_file.write(uploaded_file.read())
        parquet_path = temp_file.name
    
    st.subheader("Loading and preprocessing data...")

    features_scaled, features, labels, full_df, scaler = load_and_preprocess_parquet(
        parquet_paths=[parquet_path],
        train_on_benign_only=True
    )
    
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

    anomalous_samples = features[anomaly_flags == 1]
    df_anomalies = pd.DataFrame(anomalous_samples)

    label_name = "Anomaly"
    feature_count = 77
    
    temp_paths = convert_to_images(df_anomalies, label_name, feature_count)
            
    st.success(f"{len(temp_paths)} Images created in memory")

    st.subheader("Classifying anomaly images using CNN...")

    cnn_model = load_model("models_saved/cnn_model.h5")
    results = classify_images(cnn_model, image_dir=temp_paths)

    st.subheader("Classification Results")

    if len(results) == 0:
        st.write("NO!")
    else:
           classes = []
           st.subheader("Image Grid with Predictions")
           columns_per_row = 3
           cols = st.columns(columns_per_row)
           for idx, (image_path, predicted_class) in enumerate(results.items()):
                      predicted_label = [key for key, value in class_labels.items() if value == predicted_class][0]
                          
                      # Read the image
                      img = cv2.imread(image_path)
                      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                          
                          # Display in the appropriate column
                      col = cols[idx % columns_per_row]
                      col.image(img_rgb, caption=f"{predicted_label}", use_column_width=True)
                      classes.append(predicted_class)
        counts = Counter(classes)
        max_count = max(counts.values())
        mode = [key for key, value in counts.items() if value == max_count][0]
        mode_label = [key for key, value in class_labels.items() if value == mode][0]
        st.subheader(f"Overall Prediction: **{mode_label}**")
        
        
