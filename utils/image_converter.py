import numpy as np
import os
import cv2

benign_count = 0
anom_count = 0

def convert_to_images(df, label_name, feature_count=77, output_dir="converted_images"):
    global benign_count, anom_count

    os.makedirs(os.path.join(output_dir, str(label_name)), exist_ok=True)
    counter = 0

    for i in range(0, len(df) - 230, 231):
        chunk = df.iloc[i:i+231]

        if len(chunk) != 231:
            continue

        img = np.zeros((feature_count, feature_count, 3), dtype=np.uint8)

        for channel in range(3):
            channel_data = chunk.iloc[channel*feature_count : (channel+1)*feature_count].to_numpy()
            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=255.0, neginf=0.0)
            channel_data = np.clip(channel_data, 0, 255).astype(np.uint8)
            img[:, :, channel] = channel_data

        filename = os.path.join(output_dir, str(label_name), f"{label_name}_{counter}.png")
        cv2.imwrite(filename, img)

        if label_name == 'Benign':
            benign_count += 1
        else:
            anom_count += 1
        counter += 1
