"""
On the test dataset Coach420 we will test the original version of DeepSurf 
in other to be able to make a proper comparison analysis. 
"""

#!conda install -y -c conda-forge openbabel
#Import files
import os
import warnings
import numpy as np
import keras
import keras.backend as K
from openbabel import pybel, openbabel
from sklearn.model_selection import train_test_split
import argparse, os


from data import Featurizer, make_grid
from .net.PUResNet import PUResNet
from train_functions import get_grids, get_training_data, DiceLoss
from network import Network
from protein import Protein
from bsite_extraction import Bsite_extractor
from predict_DeepSurf import parse_args
from PUResNet_metrics import get_PLI

data_folder_path = "./data/test/coach420" # Poner nombre del zip

proteins, binding_sites, _ = get_training_data(data_folder_path) # Deberiamos usar el de DeepSurf no ?

# Upload testing data
proteins = np.load(data_folder_path+'_proteins.npy')
binding_sites = np.load(data_folder_path+'_binding_sites.npy')

# Check that the two sets have the same number of testing parameters
print(proteins.shape)
print(binding_sites.shape)


args = parse_args()

if not os.path.exists(args.prot_file):
    raise IOError('%s does not exist.' % args.prot_file)
if not os.path.exists(args.model_path):
    raise IOError('%s does not exist.' % args.model_path)
if not os.path.exists(args.output):
    os.makedirs(args.output)

prot = Protein(args.prot_file,args.protonate,args.expand,args.f,args.output, args.discard_points, args.seed)

nn = Network(args.model_path,args.model,args.voxel_size) 

lig_scores = nn.get_lig_scores(prot,args.batch)

extractor = Bsite_extractor(args.T)

extractor.extract_bsites(prot,lig_scores)

# Qu es lig y que es PKT ? Se tiene que hacer con un for en el dataset Coach420

get_PLI(lig,pkt,resolution = 1)
