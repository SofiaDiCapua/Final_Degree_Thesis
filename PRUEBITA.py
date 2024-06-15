import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, UpSampling3D, Cropping3D, ZeroPadding3D

# Define the original PUResNet model with input shape (36, 36, 36, 18)
def PUResNet():
    inputs = Input(shape=(36, 36, 36, 18))
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    outputs = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
    model = Model(inputs, outputs)
    return model

# Function to modify the PUResNet model
def modify_PUResNet(original_model, new_input_shape, new_output_shape):
    # Create new input layer with the desired shape
    new_input = Input(shape=new_input_shape)
    
    # Add new layers to adjust the new input to the original input shape
    x = UpSampling3D(size=(2, 2, 2))(new_input)  # First upsample to (32, 32, 32)
    x = ZeroPadding3D(padding=(2, 2, 2))(x)  # Pad dimensions to (36, 36, 36)
    x = Conv3D(18, (3, 3, 3), activation='relu', padding='same')(x)  # Until now the only layer that has parameters

    # Apply each layer of the original model to the new input sequentially
    for layer in original_model.layers[1:]:  # Skip the original input layer
        x = layer(x)
    
    # Adjust the original output to the desired output shape
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = Cropping3D(cropping=((10, 10), (10, 10), (10, 10)))(x)  # Adjust size to match the new output shape
    new_output = Conv3D(filters=new_output_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x) #Ajusta el 1
      

    # Build the new model
    modified_model = Model(inputs=new_input, outputs=new_output)
    
    return modified_model

# Create the original PUResNet model
original_model = PUResNet()

# Define new input and output shapes
new_input_shape = (16, 16, 16, 18)
new_output_shape = (16, 16, 16, 1)

# Modify the original model
modified_model = modify_PUResNet(original_model, new_input_shape, new_output_shape)

# Print the summary of the modified model
modified_model.summary()
