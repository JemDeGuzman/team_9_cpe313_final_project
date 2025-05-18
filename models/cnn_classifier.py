import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing import image

def load_cnn_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads and preprocesses an image from the given path.
    """
    img = image.load_img(img_path, target_size=img_size)  # Now uses (224, 224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
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
