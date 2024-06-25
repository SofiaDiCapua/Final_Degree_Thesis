import os
import h5py
import warnings
import numpy as np
import keras
import keras.backend as K
from openbabel import pybel
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Convolution3D, Conv3D, UpSampling3D, Cropping3D, ZeroPadding3D, BatchNormalization, Activation, Add, Concatenate
import matplotlib.pyplot as plt
import pandas as pd
from keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall

def plot_training_history(log_file_path, plot_file_path):
    # Read the CSV log file
    df = pd.read_csv(log_file_path)
    df['epoch'] = range(1, len(df) + 1)
    # Plot training & validation loss values
    plt.figure()
    plt.plot(df['epoch'], df['loss'], label='Training Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(plot_file_path)
    plt.close()
    print(f"Training history plot saved to {plot_file_path}")



final_weights_file = "../data"
model_name = "NoDeepSurf"
    
# Plot the training history
log_file_path = "/home/ubuntu/tfg/data/training_historyNoDeepSurf.csv" 
plot_file_path = os.path.join(final_weights_file, f"training_history_plot_{model_name}.png")
plot_training_history(log_file_path, plot_file_path)

