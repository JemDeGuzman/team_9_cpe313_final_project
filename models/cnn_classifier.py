import tensorflow as tf
import numpy as np
from PIL import Image
import os

def load_cnn_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path, target_size=(77, 77)):
    """
    Loads and preprocesses an image from the given path.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def classify_images(model, image_dir, target_size=(77, 77)):
    """
    Runs inference on all PNG images in the given directory.
    Returns a dict of {filename: prediction}.
    """
    results = {}
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(".png"):
            path = os.path.join(image_dir, filename)
            img = preprocess_image(path, target_size)
            prediction = model.predict(img)
            predicted_label = np.argmax(prediction, axis=1)[0]
            results[filename] = predicted_label
    return results
