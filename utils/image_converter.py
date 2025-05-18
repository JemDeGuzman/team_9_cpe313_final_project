import numpy as np
import os
import cv2

benign_count = 0
anom_count = 0

def min_max_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros(data.shape)  # Avoid division by zero
    return (data - min_val) / (max_val - min_val) * 255

def convert_to_images(df, label_name, feature_count):
    global benign_count, anom_count

    counter = 0
    chunk_size = feature_count * 3
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    output_dir = os.path.join("converted_images", str(label_name))
    os.makedirs(output_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        if len(chunk) != chunk_size:
            continue

        img = np.zeros((feature_count, feature_count, 3), dtype=np.uint8)

        for channel in range(3):
            channel_data = chunk.iloc[channel * feature_count : (channel + 1) * feature_count].to_numpy()
            
            # Handle NaNs and infinities first
            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=255.0, neginf=0.0)

            # ✅ SCALE THE CHANNEL TO [0, 255] RIGHT HERE
            channel_scaled = min_max_scale(channel_data)

            # Final cleanup
            channel_scaled = np.clip(channel_scaled, 0, 255).astype(np.uint8)

            # Place into the image
            img[:, :, channel] = channel_scaled.reshape((feature_count,))

        # Save the image
        filename = os.path.join(output_dir, f"{label_name}_{counter}.png")
        cv2.imwrite(filename, img)
        
        if label_name == 'Benign':
            benign_count += 1
        else:
            anom_count += 1
        counter += 1

