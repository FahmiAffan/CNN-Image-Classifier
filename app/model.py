from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple
from PIL import Image
import numpy as np
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

    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        # Reshape for model input (batch_size, height, width, channels)
        image_batch = np.expand_dims(image_array, axis=0)

        # Get predictions
        predictions = self.model.predict(image_batch, verbose=0)

        # Get the class with highest probability
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        return self.class_names[predicted_class_idx], confidence


@lru_cache(maxsize=1)
def load_model_once() -> CIFAR10Model:
    # Get the path to the model file
    current_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(current_dir, "models", "model_cifar10.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load and cache the model once
    return CIFAR10Model(model_path)


def _prepare_image(
    image_path: str, target_size: tuple[int, int] = (32, 32)
) -> np.ndarray:
    """Prepare image for CIFAR-10 model prediction."""
    image = Image.open(image_path).convert("RGB").resize(target_size)
    array = np.asarray(image, dtype=np.float32)
    # Normalize to [0, 1] range as expected by CIFAR-10 models
    array = array / 255.0
    return array


def predict_image(model: CIFAR10Model, image_path: str) -> Tuple[str, float]:
    """Predict the class of an image using the CIFAR-10 model."""
    array = _prepare_image(image_path)
    label, confidence = model.predict(array)
    return label, confidence
