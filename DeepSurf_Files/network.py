#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:13:37 2020

@author: smylonas
"""

import sys
import numpy as np, os
import tensorflow as tf
from tensorflow.contrib import slim

from features import KalasantyFeaturizer

# Obtener la ruta del directorio en el que se encuentra multi_predict.py
current_dir = os.path.dirname(__file__)

# Obtener la ruta del directorio principal (proyecto)
base_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Añadir el directorio principal al sys.path
sys.path.append(base_dir)

# Añadir el directorio PUResNet_Files al sys.path
puresnet_path = os.path.join(base_dir, 'PUResNet_Files')
sys.path.append(puresnet_path)

# Ahora puedes importar los módulos necesarios
from PUResNet import PUResNet

from keras.models import Model
from keras.layers import Input, Convolution3D, Conv3D, UpSampling3D, Cropping3D, ZeroPadding3D, BatchNormalization, Activation, Add, Concatenate
from keras.regularizers import l2

class ModifiedPUResNet:
    def __init__(self, new_input_shape, new_output_shape, weights_path):
        self.model = self.build_model(new_input_shape, new_output_shape, weights_path)

    def build_model(self, new_input_shape, new_output_shape, weights_path):
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
        x = Conv3D(
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
        modified_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        return modified_model

    def predict(self, input_data):
        predictions = self.model.predict(input_data)
        # Squeeze the predictions to remove singleton dimensions
        squeezed_predictions = np.squeeze(predictions, axis=-1)
        return squeezed_predictions


class Network:
    def __init__(self,model_path,model,voxelSize):
        gridSize = 16
        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32,shape=(None,gridSize,gridSize,gridSize,18))
        
        # New implementation
        if model == 'PUResNet':
            self.model = ModifiedPUResNet(
                new_input_shape=(gridSize, gridSize, gridSize, 18),
                new_output_shape=(gridSize, gridSize, gridSize, 1),
                weights_path=os.path.join(model_path, 'final_weights_1GRID.hdf') # CAMBIAR ESTO A PESOS MODELO TOCHO
            )
        # Old implementation
        else :
            if model=='orig':
                from net.resnet_3d import resnet_arg_scope, resnet_v1_18
            elif model=='lds':
                from net.resnet_lds_3d_bottleneck import resnet_arg_scope, resnet_v1_18
            
            with slim.arg_scope(resnet_arg_scope()):  
                self.net, self.end_points = resnet_v1_18(self.inputs, 1, is_training=False)
            

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.sess.run(tf.global_variables_initializer()) 

        
        saver = tf.train.Saver()
        if model=='orig':
                saver.restore(self.sess,os.path.join(model_path,'resnet18'))
        elif model=='lds':
                saver.restore(self.sess,os.path.join(model_path,'bot_lds_resnet18'))
        
        self.featurizer = KalasantyFeaturizer(gridSize,voxelSize) 
        
    def get_lig_scores(self, prot, batch_size):
        # First get the features
        self.featurizer.get_channels(prot.mol) 

        gridSize = 16
        lig_scores = []
        input_data = np.zeros((batch_size,gridSize,gridSize,gridSize,18))  
        batch_cnt = 0
        for p,n in zip(prot.surf_points,prot.surf_normals):
            # Then make the grids
            input_data[batch_cnt,:,:,:,:] = self.featurizer.grid_feats(p,n,prot.heavy_atom_coords)  
            batch_cnt += 1
            if batch_cnt == batch_size:
                if hasattr(self, 'end_points'):
                    output = self.sess.run(self.end_points, feed_dict={self.inputs: input_data})
                    lig_scores += list(output['probs'])
                else:
                    output = self.model.predict(input_data)
                    center_values = [out[8, 8, 8] for out in output]
                    lig_scores += list(center_values)
                batch_cnt = 0
                
        if batch_cnt>0:
            if hasattr(self, 'end_points'):
                output = self.sess.run(self.end_points, feed_dict={self.inputs: input_data[:batch_cnt, :, :, :, :]})
                if batch_cnt==1:
                    lig_scores.append(output['probs'])
                else:
                    lig_scores += list(output['probs'])
            else:
                output = self.model.predict(input_data[:batch_cnt, :, :, :, :])
                center_values = [out[8, 8, 8] for out in output]
                lig_scores += list(center_values)

        return np.array(lig_scores)
        
