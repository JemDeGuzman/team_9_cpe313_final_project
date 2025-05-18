import time
import torch
import numpy as np
import pandas as pd
import cv2
import os
from collections import deque
import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib
import torch.nn as nn

# Define Autoencoder Model
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load trained model
def load_model(model_path):
    model = SimpleAutoencoder(77)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Create RGB image from packet data chunk
def create_rgb_image(chunk, feature_count):
    chunk = chunk.flatten()
    if len(chunk) != feature_count * 3:
        raise ValueError("Invalid chunk size for image generation.")

    img = np.zeros((feature_count, feature_count, 3), dtype=np.uint8)

    for i in range(3):
        channel_data = chunk[i*feature_count:(i+1)*feature_count]
        channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=255.0, neginf=0.0)
        # Normalize each channel to 0–255 dynamically for better contrast
        channel_data = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data) + 1e-8)
        img[:, :, i] = (channel_data * 255).astype(np.uint8)

    return img

# Simulate real-time packet streaming, anomaly detection and image generation
def simulate_real_time(dataframe, model, threshold, scaler, chunk_size=231, feature_count=77):
    buffer = deque()
    image_count = 0

    # Ensure output directory exists
    os.makedirs("stream_output", exist_ok=True)

    for _, row in dataframe.iterrows():
        # Preprocess
        raw = row.values.astype(np.float32)
        scaled = scaler.transform([raw])[0]
        tensor = torch.tensor(scaled, dtype=torch.float32)

        # Detect anomaly
        with torch.no_grad():
            output = model(tensor)
            error = torch.nn.functional.mse_loss(output, tensor).item()

        # If anomaly, add to buffer
        if error > threshold:
            buffer.append(raw)

        # When buffer has enough data for an image
        if len(buffer) >= chunk_size:
            chunk = np.array([buffer.popleft() for _ in range(chunk_size)])
            img = create_rgb_image(chunk, feature_count)
            image_count += 1

            # Save and stream the image
            filename = f"stream_output/image_{image_count}.png"
            cv2.imwrite(filename, img)
            st.image(img, caption=f"Anomaly Image #{image_count}")

        # Simulate streaming speed
        time.sleep(0.2)

# Streamlit UI setup
st.title("Real-Time Packet Anomaly Detection")

model = load_model("simple_autoencoder.pth")
scaler = joblib.load("trained_scaler.pkl") 
threshold = 0.0201

uploaded_file = st.file_uploader("Upload Test Data (CSV/Parquet)", type=["csv", "parquet"])

if uploaded_file:
    df = pd.read_parquet(uploaded_file) if uploaded_file.name.endswith(".parquet") else pd.read_csv(uploaded_file)

    df = pd.read_parquet(uploaded_file) if uploaded_file.name.endswith(".parquet") else pd.read_csv(uploaded_file)

    # Remove non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        st.warning(f"Non-numeric columns were removed: {list(non_numeric_cols)}")
    df = df.select_dtypes(include=[np.number])

    simulate_real_time(df, model, threshold, scaler)
