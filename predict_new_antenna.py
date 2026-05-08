#Author: Caolán Corrigan
#Student Number: 18478834

import os
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = (128, 128)  # height, width
MODEL_FILE = "best_model.keras"
Y_MEAN_FILE = "y_mean.npy"
Y_STD_FILE = "y_std.npy"

def load_image_grayscale(path: str, img_size=(128, 128)) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(img_size[::-1], Image.BILINEAR)  # PIL expects (width, height)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array


def predict_new_image(image_path: str) -> np.ndarray:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Could not find {MODEL_FILE}")
    if not os.path.exists(Y_MEAN_FILE):
        raise FileNotFoundError(f"Could not find {Y_MEAN_FILE}")
    if not os.path.exists(Y_STD_FILE):
        raise FileNotFoundError(f"Could not find {Y_STD_FILE}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find image: {image_path}")

    model = load_model(MODEL_FILE)
    model.summary()
    y_mean = np.load(Y_MEAN_FILE)
    y_std_safe = np.load(Y_STD_FILE)

    img = load_image_grayscale(image_path, IMG_SIZE)
    img = np.expand_dims(img, axis=0)  # (1, 128, 128, 1)

    pred_std = model.predict(img, verbose=0)
    pred = pred_std * y_std_safe + y_mean

    gmax_dbi, s11_db, input_resistance_ohm, input_reactance_ohm = pred[0]

    print(f"\nPrediction for: {image_path}")
    print(f"Predicted Maximum Gain (dBi): {gmax_dbi:.3f}")
    print(f"Predicted S11 / Return Loss (dB): {s11_db:.3f}")
    print(f"Predicted Input Resistance (ohm): {input_resistance_ohm:.3f}")
    print(f"Predicted Input Reactance (ohm): {input_reactance_ohm:.3f}")

    return pred[0]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_new_antenna.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_new_image(image_path)