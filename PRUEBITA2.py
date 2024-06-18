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


   # Train the modified model
    # history = modified_model.fit(X, y, 
    #         batch_size=batch_size, epochs=epochs, 
    #         validation_split=0.1, shuffle=True,
    #         callbacks=[checkpoint_cb, early_stopping_cb, csv_logger_cb])
final_weights_file = "../data"
model_name = "Pruebita"
history = {
    'epoch': [0, 1, 2, 3, 4],
    'loss': [0.693, 0.687, 0.682, 0.678, 0.674],  # Pérdida en el conjunto de entrenamiento por cada época
    'accuracy': [0.52, 0.55, 0.58, 0.60, 0.62],  # Precisión en el conjunto de entrenamiento por cada época
    'val_loss': [0.692, 0.686, 0.681, 0.677, 0.673],  # Pérdida en el conjunto de validación por cada época
    'val_accuracy': [0.51, 0.54, 0.57, 0.59, 0.61]  # Precisión en el conjunto de validación por cada época
}

    # Optionally, save the training history to a file
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(final_weights_file, f"training_history{model_name}.csv"), index=False) 
    
    # Plot the training history
log_file_path = os.path.join(final_weights_file, f"training_history{model_name}.csv")
plot_file_path = os.path.join(final_weights_file, f"training_history_plot_{model_name}.png")
plot_training_history(log_file_path, plot_file_path)

