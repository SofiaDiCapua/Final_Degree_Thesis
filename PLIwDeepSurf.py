#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
On the test dataset Coach420 we will test the original version of DeepSurf 
in order to be able to make a proper comparison analysis. 

Attributes:
    lig : .pdb file with the ligand of the molecule, should be passed with the protein.pdb file
    pkt: .pdb file with the pocket predicted by the ANN
"""

import numpy as np
import os
from PUResNet_files.metrics import get_PLI

def calculate_pli_scores(ligand_base_path, pocket_base_path):
    # Verify that the base paths exist
    if not os.path.exists(ligand_base_path):
        raise IOError(f'The directory {ligand_base_path} does not exist.')
    if not os.path.exists(pocket_base_path):
        raise IOError(f'The directory {pocket_base_path} does not exist.')
    
    # List all subdirectories in the base directory for ligands and sort them
    ligand_subdirs = sorted([d for d in os.listdir(ligand_base_path) if os.path.isdir(os.path.join(ligand_base_path, d))])
    
    # Initialize a list to store the PLI results
    pli_scores = []

    for subdir in ligand_subdirs:
        ligand_file_path = os.path.join(ligand_base_path, subdir, 'ligand.pdb')
        pocket_file_path = os.path.join(pocket_base_path, subdir, 'protein', 'pocket1.pdb')
        
        # Verify that both files exist
        if os.path.exists(ligand_file_path) and os.path.exists(pocket_file_path):
            print(f'Calculating PLI for {ligand_file_path} and {pocket_file_path}')
            pli_score = get_PLI(ligand_file_path, pocket_file_path)
            pli_scores.append(pli_score)
        else:
            print(f'Both files were not found in the directory {subdir}')
    
    return np.array(pli_scores)

# Use the function
ligand_base_path = '/home/lmc/Documents/Sofia_TFG/Final_Degree_Thesis/data/test/coach420'
pocket_base_path = '/home/lmc/Documents/Sofia_TFG/Final_Degree_Thesis/data/Results4PLI'
pli_scores = calculate_pli_scores(ligand_base_path, pocket_base_path)

print('PLI Scores:', pli_scores)
