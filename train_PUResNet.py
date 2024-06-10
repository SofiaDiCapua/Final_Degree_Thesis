#!conda install -y -c conda-forge openbabel
#Import files
import os
import warnings
import numpy as np
import keras
import keras.backend as K
from openbabel import pybel
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D

from PUResNet_files.data import Featurizer, make_grid
from PUResNet_files.PUResNet import PUResNet
from train_functions import get_grids, get_training_data, DiceLoss

def modify_PUResNet(original_model, new_input_shape, new_output_shape):
    # Create new input layer with the desired shape
    new_input = Input(shape=new_input_shape)
    
    # Get the intermediate layers from the original model
    # We will apply the original model (minus the last layer) to the new input
    intermediate_output = original_model(new_input)
    
    # Create a new output layer with the desired shape
    new_output = Conv3D(filters=new_output_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(intermediate_output)
    
    # Build the new model
    modified_model = Model(inputs=new_input, outputs=new_output)
    
    return modified_model

def train_function(data_folder_path, X_train, y_train):
    # I have based myself on this tutorial:
    # https://keras.io/examples/vision/3D_image_classification/

    ## DEFINE VARIABLES ##
    # In the paper, a batch size of 5 was used.
    # They also found DiceLoss to be the best loss function to train the model
    batch_size = 5
    epochs = 300
    loss_function = DiceLoss


    ## DEFINE CALLBACKS ##
    # A Callback is an object that can perform actions at various stages of training
    # ModelCheckpoint will save the best weights of the training
    # EarlyStopping stops the training when val_loss stops improving
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=data_folder_path+"_best_weights.h5",
        monitor = "val_loss",
        save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", 
                                                    patience=15)


    ## TRAIN THE MODEL ##
    # HERE IS WHERE THE FINE TUNNING IS MADE
    original_model = PUResNet()
    new_input_shape = (16, 16, 16, 18)
    new_output_shape = (16, 16, 16, 1)

    # Modify the model
    modified_model = modify_PUResNet(original_model, new_input_shape, new_output_shape)

    # Print the summary of the modified model
    print(modified_model.summary())

    modified_model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
    
    # Train the modified model
    modified_model.fit(X_train, y_train, 
            batch_size=batch_size, epochs=epochs, 
            validation_split=0.1, shuffle=True,
            callbacks=[checkpoint_cb, early_stopping_cb])

    # Save the final model weights
    weights_path = '/home/lmc/Documents/Sofia_TFG/Final_Degree_Thesis/PUResNet_files'
    final_weights_path = os.path.join(weights_path, "final_weights.h5")
    modified_model.save_weights(final_weights_path)
    print(f"Final model weights saved to {final_weights_path}")

    # Optionally, save the entire model (architecture + weights)
    final_model_path = os.path.join(weights_path, "final_model.h5")
    modified_model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")



if __name__ == "__main__":
    
    data_folder_path = "../data/train/final_data" # Poner nombre del zip
    #Prepare the data
    # To not see any warnings: 
    pybel.ob.obErrorLog.StopLogging()
    # To see warnings: pybel.ob.obErrorLog.StartLogging()
    proteins, binding_sites, _ = get_training_data(data_folder_path) 

    # Upload training data
    proteins = np.load(data_folder_path+'_proteins.npy')
    binding_sites = np.load(data_folder_path+'_binding_sites.npy')

    # Check that the two sets have the same number of training parameters
    print(proteins.shape)
    print(binding_sites.shape)

    # Separate between train and test sets
    X_train, X_test, y_train, y_test = train_test_split(proteins, 
                                                        binding_sites, 
                                                        test_size=0.2, 
                                                        random_state=42)

    # Call train function
    train_function(data_folder_path, X_train, y_train)