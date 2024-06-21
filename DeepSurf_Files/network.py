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

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Añadir la ruta del directorio 'PUResNet_Files' al path
puresnet_path = os.path.join(base_dir, 'PUResNet_Files')
sys.path.append(puresnet_path)

# Intentar importar el módulo
from PUResNet import PUResNet

from ..train_PUResNet import modify_PUResNet

class Network:
    def __init__(self,model_path,model,voxelSize):
        gridSize = 16
        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32,shape=(None,gridSize,gridSize,gridSize,18))
        
        # New implementation
        if model == 'PUResNet':
            self.model = self.modify_PUResNet(
                new_input_shape=(gridSize, gridSize, gridSize, 18),
                new_output_shape=(gridSize, gridSize, gridSize, 1),
                weights_path=os.path.join(model_path, 'whole_trained_model1.hdf')
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

        
        # New implementation that uses keras (as PUResNet does)
        if model == 'PUResNet':
            self.model.load_weights(os.path.join(model_path, 'final_weights_1GRID.hdf'))
        # Old implementation that used tensorflow 
        else :
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
            if batch_cnt==batch_size:
                output = self.sess.run(self.end_points,feed_dict={self.inputs:input_data}) 
                lig_scores += list(output['probs'])
                batch_cnt = 0
                
        if batch_cnt>0:
            output = self.sess.run(self.end_points,feed_dict={self.inputs:input_data[:batch_cnt,:,:,:,:]}) 
            if batch_cnt==1:
                lig_scores.append(output['probs'])
            else:
                lig_scores += list(output['probs'])
        
        return np.array(lig_scores)
        
