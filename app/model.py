from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple
import numpy as np
import cv2
import tensorflow as tf


class CIFAR10Model:
    """CIFAR-10 CNN model for image classification."""

    def __init__(self, model_path: str) -> None:
        # CIFAR-10 class names
        self.class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        print(f"[*] Model '{model_path}' berhasil dimuat.")

    def predict(self, image_batch: np.ndarray) -> Tuple[str, float]:
        """Predict the class of a preprocessed image batch."""
        # Get predictions
        predictions = self.model.predict(image_batch, verbose=0)

        # Get the class with highest probability
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        return self.class_names[predicted_class_idx], confidence


@lru_cache(maxsize=1)
def load_model_once() -> CIFAR10Model:
    """Load and cache the CIFAR-10 model once."""
    # Get the path to the model file
    current_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(current_dir, "models", "model_cifar10.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load and cache the model once
    return CIFAR10Model(model_path)


def preprocess_image(file_stream) -> np.ndarray:
    """
    Process image from file stream using OpenCV to match CIFAR-10 model requirements.

    Args:
        file_stream: File stream from uploaded image

    Returns:
        Preprocessed image batch ready for model prediction
    """
    try:
        # Read image from stream
        filestr = file_stream.read()
        npimg = np.frombuffer(filestr, np.uint8)

        # Decode as color image (BGR format from OpenCV)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Convert BGR to RGB (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input size (32x32)
        img_resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] range (same as training)
        img_normalized = img_resized / 255.0

        # Reshape for Keras input: (1, height, width, channels)
        # Result: (1, 32, 32, 3)
        img_batch = np.reshape(img_normalized, (1, 32, 32, 3))

        return img_batch

    except Exception as e:
        print(f"[!] Error saat memproses gambar: {e}")
        return None


def predict_image(model: CIFAR10Model, image_path: str) -> Tuple[str, float]:
    """Predict the class of an image file using the CIFAR-10 model."""
    try:
        # Read image file
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input size (32x32)
        img_resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] range
        img_normalized = img_resized / 255.0

        # Reshape for Keras input: (1, height, width, channels)
        img_batch = np.reshape(img_normalized, (1, 32, 32, 3))

        # Get prediction
        label, confidence = model.predict(img_batch)
        return label, confidence

    except Exception as e:
        print(f"[!] Error saat prediksi: {e}")
        raise
