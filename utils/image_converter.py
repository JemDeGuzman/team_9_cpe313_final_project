import numpy as np
import os
import cv2

def convert_to_images(data, label_name, feature_count=77, output_dir="data/converted_images"):
    os.makedirs(os.path.join(output_dir, str(label_name)), exist_ok=True)
    saved_images = 0
    image_paths = []

    chunk_size = feature_count * 3
    flat_data = data.flatten()
    chunks = [flat_data[i:i+chunk_size] for i in range(0, len(flat_data), chunk_size)]

    for idx, chunk in enumerate(chunks):
        if len(chunk) != chunk_size:
            continue  # Incomplete chunk, skip

        img = np.zeros((feature_count, feature_count, 3), dtype=np.uint8)

        for channel in range(3):
            start = channel * feature_count
            end = (channel + 1) * feature_count
            channel_data = chunk[start:end]
            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=255.0, neginf=0.0)
            channel_data = np.clip(channel_data, 0, 255).astype(np.uint8)
            img[:, :, channel] = channel_data.reshape((feature_count,))

        filename = os.path.join(output_dir, str(label_name), f"{label_name}_{saved_images}.png")
        cv2.imwrite(filename, img)
        image_paths.append(filename)
        saved_images += 1

    return image_paths
