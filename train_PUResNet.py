#!conda install -y -c conda-forge openbabel
# Import files
import os
import sys
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

# Add folders to the PYTHONPATH
sys.path.append(os.path.abspath('DeepSurf_Files'))
sys.path.append(os.path.abspath('PUResNet_Files'))

from PUResNet_Files.data import Featurizer, make_grid
from PUResNet_Files.PUResNet import PUResNet
from train_functions import get_grids, get_training_data, get_grids_V2, get_training_data_V2, DiceLoss
from PUResNet_Files.PUResNet import PUResNet


def modify_PUResNet(new_input_shape, new_output_shape, weights_path):
    # Inicializar el modelo original
    original_model = PUResNet()

    # Crear una nueva capa de entrada con la forma deseada
    new_input = Input(shape=new_input_shape, name="new_input")
    
    # Añadir nuevas capas para ajustar la nueva entrada a la forma de entrada original
    x = UpSampling3D(size=(2, 2, 2))(new_input)  # Primer upsample a (32, 32, 32)
    x = ZeroPadding3D(padding=(2, 2, 2))(x)  # Rellenar dimensiones a (36, 36, 36)
    x = Conv3D(18, (3, 3, 3), activation='relu', padding='same', name='initial_conv')(x)

    # Utilizar los métodos de la clase PUResNet para construir el modelo
    f = 18  # El número de filtros inicial
    x = original_model.conv_block(x, [f, f, f], stage=2, block='a', strides=(1,1,1))
    x = original_model.identity_block(x, [f, f, f], stage=2, block='b')
    x1 = original_model.identity_block(x, [f, f, f], stage=2, block='c')

    x = original_model.conv_block(x, [f*2, f*2, f*2], stage=4, block='a', strides=(2,2,2))
    x = original_model.identity_block(x, [f*2, f*2, f*2], stage=4, block='b')
    x2 = original_model.identity_block(x, [f*2, f*2, f*2], stage=4, block='f')

    x = original_model.conv_block(x, [f*4, f*4, f*4], stage=5, block='a', strides=(2,2,2))
    x = original_model.identity_block(x, [f*4, f*4, f*4], stage=5, block='b')
    x3 = original_model.identity_block(x, [f*4, f*4, f*4], stage=5, block='c')

    x = original_model.conv_block(x, [f*8, f*8, f*8], stage=6, block='a', strides=(3,3,3))
    x = original_model.identity_block(x, [f*8, f*8, f*8], stage=6, block='b')
    x4 = original_model.identity_block(x, [f*8, f*8, f*8], stage=6, block='c')

    x = original_model.conv_block(x, [f*16, f*16, f*16], stage=7, block='a', strides=(3,3,3))
    x = original_model.identity_block(x, [f*16, f*16, f*16], stage=7, block='b')

    x = original_model.up_conv_block(x, [f*16, f*16, f*16], stage=8, block='a', size=(3,3,3), padding='same')
    x = original_model.identity_block(x, [f*16, f*16, f*16], stage=8, block='b')
    x = Concatenate(axis=4)([x, x4])

    x = original_model.up_conv_block(x, [f*8, f*8, f*8], stage=9, block='a', size=(3,3,3), stride=(1,1,1))
    x = original_model.identity_block(x, [f*8, f*8, f*8], stage=9, block='b')
    x = Concatenate(axis=4)([x, x3])

    x = original_model.up_conv_block(x, [f*4, f*4, f*4], stage=10, block='a', size=(2,2,2), stride=(1,1,1))
    x = original_model.identity_block(x, [f*4, f*4, f*4], stage=10, block='b')
    x = Concatenate(axis=4)([x, x2])

    x = original_model.up_conv_block(x, [f*2, f*2, f*2], stage=11, block='a', size=(2,2,2), stride=(1,1,1))
    x = original_model.identity_block(x, [f*2, f*2, f*2], stage=11, block='b')
    x = Concatenate(axis=4)([x, x1])

    # Añadir la última capa de convolución
    x = Convolution3D(
        filters=1,
        kernel_size=1,
        kernel_regularizer=l2(1e-4),
        activation='sigmoid',
        name='pocket'
    )(x)

    # Ajustar la salida original a la forma de salida deseada
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='final_conv')(x)
    x = Cropping3D(cropping=((10, 10), (10, 10), (10, 10)), name='final_crop')(x)
    new_output = Conv3D(filters=new_output_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same', name='new_output_conv')(x)

    # Construir el nuevo modelo
    modified_model = Model(inputs=new_input, outputs=new_output)

    # Printing the shapes of the model's layers (for debugging)
    for layer in modified_model.layers:
        print(layer.name, layer.output_shape)

    # Cargar los pesos del modelo original, omitiendo las capas de entrada y salida que no coincidan
    modified_model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return modified_model


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

def train_function(data_folder_path, X, y, batch_size, epochs, loss_function, final_weights_file, model_name):
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
    original_weights_path = '/home/ubuntu/tfg/data/whole_trained_model1.hdf'
    original_model.save_weights(original_weights_path)

    new_input_shape = (16, 16, 16, 18)
    new_output_shape = (16, 16, 16, 1)

    # Modify the model
    modified_model = modify_PUResNet(new_input_shape, new_output_shape, original_weights_path)
    print("The modified model is created")
    # Print the summary of the modified model
    print(modified_model.summary())
    
    modified_model.compile(loss=loss_function, optimizer = "adam", metrics=["accuracy", Recall(), Precision()])

    # # OJO
    # precision = tf.metrics.precision(labels, predictions)
    # recall = tf.metrics.recall(labels, predictions)
    # f1_score = 2 * ((precision * recall) / (precision + recall))

    
    # Train the modified model
    history = modified_model.fit(X, y, 
            batch_size=batch_size, epochs=epochs, 
            validation_split=0.1, shuffle=True,
            callbacks=[checkpoint_cb, early_stopping_cb, csv_logger_cb])

    # Save the trained weights
    file_path = os.path.join(final_weights_file, f"final_weights_{model_name}.hdf")
    modified_model.save_weights(file_path)
    print(f"Final trained weights saved to {file_path}")

    # Optionally, save the training history to a file
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(final_weights_file, f"training_history{model_name}.csv"), index=False)
    
    # Plot the training history
    log_file_path = os.path.join(final_weights_file, f"training_history{model_name}.csv")
    plot_file_path = os.path.join(final_weights_file, f"training_history_plot_{model_name}.png")
    plot_training_history(log_file_path, plot_file_path)


def evaluate_model(modified_model, X, y):
    # Faltaria cargar el modelo
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


if __name__ == "__main__":
    
    data_folder_path = "../data/train/final_data" 
    final_weights_file = "../data"
    ## DEFINE VARIABLES ##
    batch_size = 5
    epochs = 20
    loss_function = DiceLoss
    k = 4  # Number of folds for K-fold cross-validation

    # Prepare the data
    # To not see any warnings: 
    pybel.ob.obErrorLog.StopLogging()
    # To see warnings: pybel.ob.obErrorLog.StartLogging()
    # proteins, binding_sites, _ = get_training_data(data_folder_path, proteins_pkl='../data/proteins.pkl', binding_sites_pkl='../data/binding_sites.pkl') 
    proteins, binding_sites, _ = get_training_data_V2(data_folder_path, proteins_pkl='../data/GOODproteins.pkl', binding_sites_pkl='../data/GOODbinding_sites.pkl') 

    # Check that the two sets have the same number of training parameters
    print(proteins.shape) # It should give (2368, 16, 16, 16, 18)
    print(binding_sites.shape) # It should give (2368, 16, 16, 16, 1)

    # Call train function
    train_function(data_folder_path, proteins, binding_sites, batch_size, epochs, loss_function, final_weights_file, model_name = "DeepSurfGRIDs")
