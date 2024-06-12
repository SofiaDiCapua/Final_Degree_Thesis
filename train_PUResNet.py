#!conda install -y -c conda-forge openbabel
# Import files
import os
import warnings
import numpy as np
import keras
import keras.backend as K
from openbabel import pybel
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D
import matplotlib.pyplot as plt
import pandas as pd

from PUResNet_files.data import Featurizer, make_grid
from PUResNet_files.PUResNet import PUResNet
from train_functions import get_grids, get_training_data, DiceLoss

def modify_PUResNet(new_input_shape, new_output_shape, weights_path):
    # Recreate the original model with the new input shape
    modified_model = PUResNet(input_shape=new_input_shape)
    
    # Load the weights from the original model, skipping the input layer
    modified_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    
    # Ensure the output layer matches the desired output shape
    new_output = Conv3D(filters=new_output_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(modified_model.layers[-2].output)
    modified_model = Model(inputs=modified_model.input, outputs=new_output)
    
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

def train_function(data_folder_path, X, y):
    # I have based myself on this tutorial:
    # https://keras.io/examples/vision/3D_image_classification/

    ## DEFINE VARIABLES ##
    batch_size = 5
    epochs = 300
    loss_function = DiceLoss
    k = 4  # Number of folds for K-fold cross-validation

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Store results for each fold
    results = []

    # Define a single checkpoint file
    checkpoint_file = os.path.join(data_folder_path, "best_weights.h5")

    fold = 1
    for train_index, test_index in kf.split(X):
        print(f"Training fold {fold}...")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

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
        original_model = PUResNet(input_shape=(16, 16, 16, 18))

        # Save weights of the original model for later reuse
        original_weights_path = '/home/lmc/Documents/Sofia_TFG/whole_trained_model1.hdf'
        original_model.save_weights(original_weights_path)

        new_input_shape = (16, 16, 16, 18)
        new_output_shape = (16, 16, 16, 1)

        # Modify the model
        modified_model = modify_PUResNet(new_input_shape, new_output_shape, original_weights_path)

        # Print the summary of the modified model
        print(modified_model.summary())

        modified_model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
        
        # Train the modified model
        modified_model.fit(X_train, y_train, 
                batch_size=batch_size, epochs=epochs, 
                validation_split=0.1, shuffle=True,
                callbacks=[checkpoint_cb, early_stopping_cb, csv_logger_cb])

        # Evaluate the model
        y_pred = (modified_model.predict(X_test) > 0.5).astype("int32")

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test.flatten(), y_pred.flatten()).ravel()
        f1 = f1_score(y_test.flatten(), y_pred.flatten())

        results.append({
            "Fold": fold,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "F1 score": f1
        })

        fold += 1

    # Print results as a table
    results_df = pd.DataFrame(results)
    results_df.loc['Average'] = results_df.mean()
    print(results_df)

    # Optionally, save the results to a CSV file
    results_df.to_csv(os.path.join(data_folder_path, 'k_fold_results.csv'), index=False)

    # Plot the training history for the last fold
    log_file_path = os.path.join(data_folder_path, 'training_log.csv')
    plot_file_path = os.path.join(data_folder_path, 'training_history_plot.png')
    plot_training_history(log_file_path, plot_file_path)


if __name__ == "__main__":
    
    data_folder_path = "../data/train/final_data" # Poner nombre del zip
    # Prepare the data
    # To not see any warnings: 
    pybel.ob.obErrorLog.StopLogging()
    # To see warnings: pybel.ob.obErrorLog.StartLogging()
    proteins, binding_sites, _ = get_training_data(data_folder_path) 

    # Remove unnecessary dimensions
    proteins = np.squeeze(proteins, axis=1)
    binding_sites = np.squeeze(binding_sites, axis=1)

    # Check that the two sets have the same number of training parameters
    print(proteins.shape) # It should give (2368, 16, 16, 16, 18)
    print(binding_sites.shape) # It should give (2368, 16, 16, 16, 1)

    # Call train function
    train_function(data_folder_path, proteins, binding_sites)
