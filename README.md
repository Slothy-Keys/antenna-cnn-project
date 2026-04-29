# Antenna CNN Project

This project trains a convolutional neural network (CNN) to predict patch antenna parameters from generated images.
To use the predict_new_antenna file ensure that the patch antenna is 128x128, grayscale and is in coax cable format.

## Outputs
- Maximum Gain (dBi)
- S11 (dB)
- Main Beam Angle (degrees)

## Files

- train_cnn_main.py → main training script
- predict_new_antenna.py → predict new antenna image
- best_model.keras → trained model
- y_mean.npy / y_std.npy → normalization values
- labels.csv → dataset labels
- create_antenna.m → MATLAB code to create one antenna
- generate_patch_dataset.m → MATLAB code to create antenna dataset of any size.

## How to run

Install Anaconda
Open anaconda prompt
Create your environment
>conda create -n antenna_cnn python=3.10
Enter environment
>conda activate antenna_cnn
Install packages
>pip install tensorflow numpy matplotlib pandas scikit-learn opencv-python pillow jupyter

Train:
python train_cnn_main.py

Predict:
python predict_new_antenna.py <image_path>
