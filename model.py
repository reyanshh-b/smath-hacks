import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

DEFAULT_CLASS_NAMES = ['healthy', 'unhealthy']

def load_class_names(path='models/class_names.txt'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return DEFAULT_CLASS_NAMES

def preprocess_image(path, target_size=(224, 224)):
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x