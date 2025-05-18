import numpy as np
import os
import cv2
import pandas as pd

def convert_to_images(df, label_name, feature_count=77, output_dir="data/converted_images/Anomaly"):
    os.makedirs(os.path.join(output_dir, str(label_name)), exist_ok=True)
    image_paths = []

    rows_per_image = feature_count * 3  # 231 rows needed per image
    total_rows = len(df)

    num_images = total_rows // rows_per_image
    if total_rows % rows_per_image != 0:
        num_images += 1  # Add one more image if leftover rows exist

    for i in range(num_images):
        start_idx = i * rows_per_image
        end_idx = start_idx + rows_per_image

        chunk = df.iloc[start_idx:end_idx].copy()

        # Pad if we have fewer than 231 rows
        if len(chunk) < rows_per_image:
            padding_rows = rows_per_image - len(chunk)
            padding = pd.DataFrame(np.zeros((padding_rows, feature_count)), columns=df.columns)
            chunk = pd.concat([chunk, padding], ignore_index=True)

        img = np.zeros((feature_count, feature_count, 3), dtype=np.uint8)

        for channel in range(3):
            channel_data = chunk.iloc[channel * feature_count : (channel + 1) * feature_count].to_numpy()

            # Replace NaNs and infinities, normalize and clip
            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=255.0, neginf=0.0)
            channel_data = np.clip(channel_data, 0, 255)

            # Flatten and reshape to 77x77
            flat = channel_data.flatten()
            if flat.size != feature_count * feature_count:
                continue  # Skip incomplete image
            img[:, :, channel] = flat.reshape((feature_count, feature_count)).astype(np.uint8)

        filename = os.path.join(output_dir, str(label_name), f"{label_name}_{i}.png")
        cv2.imwrite(filename, img)
        image_paths.append(filename)

    return image_paths
