#!conda install -y -c conda-forge openbabel
# Import files
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
from keras.layers import Input, Conv3D, UpSampling3D, Cropping3D, ZeroPadding3D, BatchNormalization, Activation, Add, Concatenate
import matplotlib.pyplot as plt
import pandas as pd

from PUResNet_files.data import Featurizer, make_grid
from PUResNet_files.PUResNet import PUResNet
from train_functions import get_grids, get_training_data, DiceLoss

def modify_PUResNet(new_input_shape, new_output_shape, weights_path):
    # Initialize the original model
    original_model = PUResNet()

    # Create new input layer with the desired shape
    new_input = Input(shape=new_input_shape)

    # Add new layers to adjust the new input to the original input shape
    x = UpSampling3D(size=(2, 2, 2))(new_input)  # First upsample to (32, 32, 32)
    x = ZeroPadding3D(padding=(2, 2, 2))(x)  # Pad dimensions to (36, 36, 36)
    x = Conv3D(18, (3, 3, 3), activation='relu', padding='same')(x)

    # Manually recreate the architecture of the original model
    for layer in original_model.layers[1:]:
        if isinstance(layer, Conv3D):
            x = Conv3D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=layer.activation,
                kernel_regularizer=layer.kernel_regularizer
            )(x)
        elif isinstance(layer, UpSampling3D):
            x = UpSampling3D(size=layer.size)(x)
        elif isinstance(layer, ZeroPadding3D):
            x = ZeroPadding3D(padding=layer.padding)(x)
        elif isinstance(layer, Cropping3D):
            x = Cropping3D(cropping=layer.cropping)(x)
        elif isinstance(layer, BatchNormalization):
            x = BatchNormalization(axis=layer.axis)(x)
        elif isinstance(layer, Activation):
            x = Activation(layer.activation)(x)
        elif isinstance(layer, Add):
            x = Add()([x, x])  # Note: Add layer's input handling needs attention
        elif isinstance(layer, Concatenate):
            x = Concatenate(axis=layer.axis)([x, x])  # Note: Concatenate layer's input handling needs attention
        else:
            raise ValueError(f"Layer type {type(layer)} is not handled in the reconstruction")

    # Adjust the original output to the desired output shape
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = Cropping3D(cropping=((10, 10), (10, 10), (10, 10)))(x)
    new_output = Conv3D(filters=new_output_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)

    # Build the new model
    modified_model = Model(inputs=new_input, outputs=new_output)

    # Load the weights from the original model, skipping the input and output layers that do not match
    modified_model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return modified_model


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

def train_function(data_folder_path, X, y, batch_size, epochs, loss_function, final_weights_file):
    # Define a single checkpoint file
    checkpoint_file = os.path.join(data_folder_path, "best_weights.h5")

    ## DEFINE CALLBACKS ##
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,  # Single checkpoint file
        monitor="val_loss",
        save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", 
                                                      patience=15)
    csv_logger_cb = keras.callbacks.CSVLogger(os.path.join(data_folder_path, 'training_log.csv'))

    ## TRAIN THE MODEL ##
    # Create the original PUResNet model with the initial input shape
    original_model = PUResNet()

    # Save weights of the original model for later reuse
    original_weights_path = '/home/lmc/Documents/Sofia_TFG/whole_trained_model1.hdf'
    original_model.save_weights(original_weights_path)

    new_input_shape = (16, 16, 16, 18)
    new_output_shape = (16, 16, 16, 1)

    # Modify the model
    modified_model = modify_PUResNet(new_input_shape, new_output_shape, original_weights_path)
    print("The modified model is created")
    # Print the summary of the modified model
    print(modified_model.summary())

def plot_training_history(log_file_path, plot_file_path):
    # Read the CSV log file
    df = pd.read_csv(log_file_path)
    
    # Train the modified model
    history = modified_model.fit(X, y, 
            batch_size=batch_size, epochs=epochs, 
            validation_split=0.1, shuffle=True,
            callbacks=[checkpoint_cb, early_stopping_cb, csv_logger_cb])

    # Save the trained weights
    modified_model.save_weights(final_weights_file)
    print(f"Final trained weights saved to {final_weights_file}")

    # Optionally, save the training history to a file
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(data_folder_path, 'training_history.csv'), index=False)

    # Evaluate the model
    y_pred = (modified_model.predict(X) > 0.5).astype("int32")

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y.flatten(), y_pred.flatten()).ravel()
    f1 = f1_score(y.flatten(), y_pred.flatten())

    results = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "F1 score": f1
    }

    # Print results
    results_df = pd.DataFrame([results])
    print(results_df)

    # Optionally, save the results to a CSV file
    results_df.to_csv(os.path.join(data_folder_path, 'training_results.csv'), index=False)

    # Plot the training history
    log_file_path = os.path.join(data_folder_path, 'training_log.csv')
    plot_file_path = os.path.join(data_folder_path, 'training_history_plot.png')
    plot_training_history(log_file_path, plot_file_path)


if __name__ == "__main__":
    
    data_folder_path = "../data/train/final_data" 
    final_weights_file = "../data"
    ## DEFINE VARIABLES ##
    batch_size = 2
    epochs = 300
    loss_function = DiceLoss
    k = 4  # Number of folds for K-fold cross-validation

    # Prepare the data
    # To not see any warnings: 
    pybel.ob.obErrorLog.StopLogging()
    # To see warnings: pybel.ob.obErrorLog.StartLogging()
    proteins, binding_sites, _ = get_training_data(data_folder_path) 

    # Check that the two sets have the same number of training parameters
    print(proteins.shape) # It should give (2368, 16, 16, 16, 18)
    print(binding_sites.shape) # It should give (2368, 16, 16, 16, 1)

    # Call train function
    train_function(data_folder_path, proteins, binding_sites, batch_size, epochs, loss_function, final_weights_file)
