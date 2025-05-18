import numpy as np
import pandas as pd
import cv2
import os
import tempfile

benign_count = 0
anom_count = 0

def convert_to_images(df, label_name, feature_count):
    global benign_count, anom_count

    counter = 0
    chunk_size = feature_count * 3
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    output_dir = os.path.join("data/converted_images", str(label_name))
    os.makedirs(output_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        if len(chunk) != chunk_size:
            continue

        img = np.zeros((feature_count, feature_count, 3), dtype=np.uint8)

        # Process each channel (R, G, B)
        for channel in range(3):
            channel_data = chunk.iloc[channel*feature_count : (channel+1)*feature_count]
            channel_processed = np.nan_to_num(channel_data, nan=0.0, posinf=255.0, neginf=0.0)
            channel_processed = np.clip(channel_processed, 0, 255).astype(np.uint8)
            img[:, :, channel] = channel_processed

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True, delete_on_close=False) as tmp_file:
            temp_img_path = tmp_file.name
            cv2.imwrite(temp_img_path, img)
        
        if label_name == 'Benign':
            benign_count += 1
        else:
            anom_count += 1
        counter += 1

