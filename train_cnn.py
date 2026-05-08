# Author: Caolán Corrigan
# Student Number: 18478834

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers


# Configuration
IMG_DIR = "images_128x128"
LABEL_CSV = "labels.csv"
IMG_SIZE = (128, 128)  # height, width

# For reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Loading labels from CSV
print("Loading labels from CSV...")
df = pd.read_csv(LABEL_CSV)

# Optional filtering to remove extreme impedance outliers
print("Rows before filtering:", len(df))

df = df[
    (df["InputResistance_ohm"] >= 0) &
    (df["InputResistance_ohm"] <= 100) &
    (df["InputReactance_ohm"] >= -40) &
    (df["InputReactance_ohm"] <= 100)
].reset_index(drop=True)

print("Rows after filtering:", len(df))

# Targets
target_cols = ["Gmax_dBi",
               "S11_dB",
               "InputResistance_ohm",
               "InputReactance_ohm"
]

# Check that columns exist
print("Columns found in CSV:", df.columns.tolist())
missing_cols = [col for col in ["filename"] + target_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# Extract filenames and targets
filenames = df["filename"].values
y = df[target_cols].values.astype(np.float32)  # shape: (N, 4)

print("Number of samples:", len(filenames))


# Loading images into a NumPy array
def load_image_grayscale(path, img_size=(128, 128)):
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize(img_size[::-1], Image.BILINEAR)  # PIL uses (width, height)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
    return img_array


print("Loading images...")
X_list = []
for fname in filenames:
    img_path = os.path.join(IMG_DIR, fname)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_array = load_image_grayscale(img_path, IMG_SIZE)
    X_list.append(img_array)

X = np.stack(X_list, axis=0)  # shape: (N, 128, 128, 1)

print("Image array shape:", X.shape)
print("Target array shape:", y.shape)


# Train / Validation / Test Split (80 / 10 / 10)

# Split off test set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=SEED, shuffle=True
)

# After removing 10% test, validation fraction inside temp set is 10/90
val_fraction_of_temp = 0.10 / 0.90

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_fraction_of_temp,
    random_state=SEED,
    shuffle=True
)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:  ", X_val.shape, y_val.shape)
print("Test shape: ", X_test.shape, y_test.shape)


# Standardize targets for more stable training

# Compute mean and std on the training targets only
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)

print("Target means:", y_mean)
print("Target stds: ", y_std)

# Avoid divide-by-zero
y_std_safe = np.where(y_std == 0, 1.0, y_std)

# Standardize
y_train_std = (y_train - y_mean) / y_std_safe
y_val_std   = (y_val   - y_mean) / y_std_safe


# Define CNN model
def build_cnn_model(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    outputs = layers.Dense(4, activation="linear")(x)

    return models.Model(inputs, outputs)


model = build_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
model.summary()


# Compile model
learning_rate = 5e-4
optimizer = optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)


# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True
)

lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


# Training
batch_size = 32
epochs = 100

history = model.fit(
    X_train, y_train_std,
    validation_data=(X_val, y_val_std),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[checkpoint, early_stop, lr_scheduler],
    verbose=1
)

# Save normalization values for prediction
np.save("y_mean.npy", y_mean)
np.save("y_std.npy", y_std_safe)


# Plot training curves
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("MSE (standardized targets)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")
plt.close()


# Evaluate on test set

# Predict standardized outputs
y_test_pred_std = model.predict(X_test, verbose=0)

# De-standardize
y_test_pred = y_test_pred_std * y_std_safe + y_mean

# Compute MAE per output in real units
mae_per_output = np.mean(np.abs(y_test_pred - y_test), axis=0)

for name, mae_val in zip(target_cols, mae_per_output):
    print(f"Test MAE for {name}: {mae_val:.3f}")

# Save predictions for checking individual test examples
results_df = pd.DataFrame()
for i, name in enumerate(target_cols):
    results_df[f"true_{name}"] = y_test[:, i]
    results_df[f"pred_{name}"] = y_test_pred[:, i]
    results_df[f"abs_error_{name}"] = np.abs(y_test_pred[:, i] - y_test[:, i])

results_df.to_csv("test_predictions.csv", index=False)

pretty_names = {
    "Gmax_dBi": "Maximum Gain (dBi)",
    "S11_dB": "S11 (dB)",
    "InputResistance_ohm": "Input Resistance (ohm)",
    "InputReactance_ohm": "Input Reactance (ohm)"
}

# Scatter plot predicted vs true for each target
for i, name in enumerate(target_cols):
    plt.figure()

    plt.scatter(y_test[:, i], y_test_pred[:, i], alpha=0.7)

    label_name = pretty_names.get(name, name)
    plt.xlabel(f"True {label_name}")
    plt.ylabel(f"Predicted {label_name}")
    plt.title(f"Predicted vs True: {label_name}")

    min_val = min(y_test[:, i].min(), y_test_pred[:, i].min())
    max_val = max(y_test[:, i].max(), y_test_pred[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scatter_{name}.png")
    plt.close()

print("Done. Check 'training_loss.png', 'scatter_*.png', and 'test_predictions.csv' for results.")
