import time
import torch
import numpy as np
import pandas as pd
import cv2
from collections import deque
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn as nn

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

def load_model(model_path):
    model = SimpleAutoencoder(77)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model

def create_rgb_image(chunk, feature_count):
    chunk = chunk.flatten()
    if len(chunk) != feature_count * 3:
        raise ValueError("Invalid chunk size for image generation.")

    img = np.zeros((feature_count, feature_count, 3), dtype=np.uint8)

    for i in range(3):
        channel_data = chunk[i*feature_count:(i+1)*feature_count]
        channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=255.0, neginf=0.0)
        img[:, :, i] = np.clip(channel_data, 0, 255).astype(np.uint8)

    return img

def simulate_real_time(dataframe, model, threshold, scaler, chunk_size=231, feature_count=77):
    buffer = deque()
    image_count = 0

    for _, row in dataframe.iterrows():
        # Preprocess
        raw = row.values.astype(np.float32)
        scaled = scaler.transform([raw])[0]
        tensor = torch.tensor(scaled, dtype=torch.float32).to(next(model.parameters()).device)

        # Detect anomaly
        with torch.no_grad():
            output = model(tensor)
            error = torch.nn.functional.mse_loss(output, tensor).item()

        if error > threshold:
            buffer.append(raw)

        # When we have enough for an image
        if len(buffer) >= chunk_size:
            chunk = np.array([buffer.popleft() for _ in range(chunk_size)])
            img = create_rgb_image(chunk, feature_count)
            image_count += 1

            # Save or stream the image
            cv2.imwrite(f"stream_output/image_{image_count}.png", img)
            st.image(img, caption=f"Anomaly Image #{image_count}")
        
        # Simulate real-time delay
        time.sleep(0.2)  # adjust this for streaming speed

st.title("Real-Time Packet Anomaly Detection")

model = load_model("simple_autoencoder.pth")  # Load your trained AE
scaler = ...  # Load your trained scaler
threshold = 0.0201

uploaded_file = st.file_uploader("Upload Test Data (CSV/Parquet)", type=["csv", "parquet"])
if uploaded_file:
    df = pd.read_parquet(uploaded_file) if uploaded_file.name.endswith(".parquet") else pd.read_csv(uploaded_file)
    simulate_real_time(df, model, threshold, scaler)
