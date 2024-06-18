import os
import sys
import numpy as np
import pickle
import keras
import keras.backend as K
from openbabel import pybel
from sklearn.model_selection import train_test_split

# Añade las carpetas al PYTHONPATH
sys.path.append(os.path.abspath('DeepSurf_Files'))
sys.path.append(os.path.abspath('PUResNet_Files'))

from PUResNet_files.data import Featurizer, make_grid
from PUResNet_files.PUResNet import PUResNet
from DeepSurf_Files.protein import Protein
from DeepSurf_Files.features import KalasantyFeaturizer

def get_grids(file_type, prot_input_file, bs_input_file=None,
              grid_resolution=2, max_dist=35, 
              featurizer=Featurizer(save_molecule_codes=False)):
    """
    Converts both a protein file (PDB or mol2) and its ligand (if specified)
    to a grid.

    To make a 16x16x16x18 grid, max_dist should be 7.5 and grid_resolution = 1
    because make_grid returns np.ndarray, shape = (M, M, M, F) and 
    M is equal to (2 * `max_dist` / `grid_resolution`) + 1  
    36x36x36x18 --> max_dist = 35 and grid_resolution = 2
    
    Parameters
    ----------
    file_type: "pdb", "mol2"
    prot_input_file, ligand_input_file: protein and ligand files
    grid_resolution: float, optional
        Resolution of a grid (in Angstroms).
    max_dist: float, optional
        Maximum distance between atom and box center. Resulting box has size of
        2*`max_dist`+1 Angstroms and atoms that are too far away are not
        included.
    """

    # Convert file into pybel object and get the features of the molecule. 
    # If binding site, features is an array of 1s (indicating that bs is present)
    prot_input_file = prot_input_file.replace('.ipynb_checkpoints/', '')
    bs_input_file = bs_input_file.replace('.ipynb_checkpoints/', '')

    if not os.path.exists(prot_input_file):
        raise IOError("No such file: '%s'" % prot_input_file)
    if bs_input_file and not os.path.exists(bs_input_file):
        raise IOError("No such file: '%s'" % bs_input_file)

    # Convert to Protein object --> simplify_dms + KMeans
    # prot = Protein(prot_input_file, output=prot_input_file) 

    prot = next(pybel.readfile(file_type, prot_input_file)) # THIS PART SHOLD BE REMOVED?
    # THIS PART SHOULD BE CHANGED TO THE A NETWORK CLASS OBJECT  ?
    prot_coords, prot_features = featurizer.get_features(prot) 
    
    # Change all coordinates to be respect the center of the protein
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid
    # Create the grid (we want to make more than one)
    prot_grid = make_grid(prot_coords, prot_features,
                          max_dist=max_dist,
                          grid_resolution=grid_resolution)
    
    # Do the same for the binding site, if input file specified
    if bs_input_file:
        bs = next(pybel.readfile(file_type, bs_input_file))
        bs_coords, _ = featurizer.get_features(bs)
        # BS just has 1 feature: an array of 1s for each atom, indicating the
        # atom is present in that position
        # The objective of the ANN is to annotate a 1 in the atoms that form the biding site
        bs_features = np.ones((len(bs_coords), 1))
        bs_coords -= centroid
        bs_grid = make_grid(bs_coords, bs_features,
                            max_dist=max_dist,
                            grid_resolution=grid_resolution)
        print("Created binding site grid for:", bs_input_file)
    else:
        bs_grid = None
    
    return prot_grid, bs_grid, centroid

def to_pickle(data, fname):
    """Save data in a given file.
    Parameters
    ----------
    data: Any
        Data to be saved
    fname: str
       Path to file in which data will be saved
    """
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def from_pickle(fname):
    """Load data from a given file.
    Parameters
    ----------
    fname: str
       Path to file from which data will be loaded
    Returns
    -------
    data: Any
        Loaded data
    """
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def get_training_data(input_folder, proteins_pkl='proteins.pkl', binding_sites_pkl='binding_sites.pkl'):
    """
    Returns a np array containing the protein grids, one np array with the binding_sites grids,
    and the centroid coordinates for each one. 
    """   
    # Try to load from pickle files first
    try:
        proteins = from_pickle(proteins_pkl)
        binding_sites = from_pickle(binding_sites_pkl)
        print("Data loaded from pickle files.")
        return proteins, binding_sites, []
    except (FileNotFoundError, pickle.UnpicklingError):
        print("No valid pickle files found, processing data...")

    proteins = None
    binding_sites = None
    centroids = []

    for root, dirs, _ in os.walk(input_folder, topdown=False):
        for dir in dirs:
            protein_file = os.path.join(root, dir, "protein.pdb")
            site_file = os.path.join(root, dir, "cavity6.pdb")
            
            print("Processing protein file:", protein_file)
            print("Processing binding site file:", site_file)

            try:
                # A new get_grids should be implemented and used here
                prot_grid, bs_grid, centroid = get_grids("pdb", protein_file, site_file, grid_resolution=1, max_dist=7.5)
                if prot_grid is not None:
                    # Remove the unnecessary extra dimension (axis=1)
                    prot_grid = np.squeeze(prot_grid, axis=0)
                    bs_grid = np.squeeze(bs_grid, axis=0)
                    if proteins is None:
                        print("Inside the if where proteins is None")
                        proteins = np.expand_dims(prot_grid, axis=0)
                        binding_sites = np.expand_dims(bs_grid, axis=0) if bs_grid is not None else None
                    else:
                        proteins = np.concatenate((proteins, np.expand_dims(prot_grid, axis=0)), axis=0)
                        if bs_grid is not None:
                            if binding_sites is None:
                                binding_sites = np.expand_dims(bs_grid, axis=0)
                            else:
                                binding_sites = np.concatenate((binding_sites, np.expand_dims(bs_grid, axis=0)), axis=0)
                    
                    centroids.append(centroid)
                else:
                    print("Failed to create grid for:", protein_file)
            except Exception as e:
                print(f"Error processing {protein_file}: {e}")
        
    if proteins is not None:
        print("Number of proteins to train the model:", proteins.shape[0])
    else:
        print("No proteins found to train the model.")

    # Save the data to pickle files
    to_pickle(proteins, proteins_pkl)
    to_pickle(binding_sites, binding_sites_pkl)

    return proteins, binding_sites, centroids



def DiceLoss(targets, inputs, smooth=1e-6):
    '''
    Loss function to use to train the data
    call with: model.compile(loss=Diceloss)
    DiceLoss is used as it was
    '''
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #reshape to 2D matrices
    inputs = K.reshape(inputs, (-1, 1))
    targets = K.reshape(targets, (-1, 1))
    
    intersection = K.sum(targets * inputs)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def get_grids_V2(file_type, prot_input_file, bs_input_file=None,
              grid_resolution=2, max_dist=35, 
              featurizer=Featurizer(save_molecule_codes=False)):
    """
    Converts both a protein file (PDB or mol2) and its ligand (if specified)
    to a grid.
    """
    prot_input_file = prot_input_file.replace('.ipynb_checkpoints/', '')
    bs_input_file = bs_input_file.replace('.ipynb_checkpoints/', '')

    if not os.path.exists(prot_input_file):
        raise IOError("No such file: '%s'" % prot_input_file)
    if bs_input_file and not os.path.exists(bs_input_file):
        raise IOError("No such file: '%s'" % bs_input_file)

    # Create Protein object and process ASA with DMS and K-Means
    protein = Protein(prot_input_file)
    
    # To achieve (16,16,16) max_dist should be 7.5, max_dist = (gridSize-1)*voxelSize/2
    gridSize = 16
    voxelSize = 1
    featurizer = KalasantyFeaturizer(gridSize,voxelSize) 
    featurizer.get_channels(protein.mol) 

    prot_grids = []
    for p,n in zip(protein.surf_points,protein.surf_normals):
        # Then make the grids
        # grid_feats --> tfbio_data.make_grid()
        grid = self.featurizer.grid_feats(p,n,protein.heavy_atom_coords)  
        prot_grids.append(grid)
    
    # Convert list of grids to a numpy array
    prot_grids = np.array(prot_grids)
     
    if bs_input_file:
        bs = next(pybel.readfile(file_type, bs_input_file))
        # _get_channels --> featurizer.get_features --> return coords, features
        # But we only need the coord because it's the output
        bs_coords, bs_features = featurizer.get_channels(bs)
        bs_features = np.ones((len(bs_coords), 1))
        # bs_coords -= centroid
        bs_grid = make_grid(bs_coords, bs_features, max_dist=max_dist, grid_resolution=grid_resolution)
        print("Created binding site grid for:", bs_input_file)
    else:
        bs_grid = None
    
    return prot_grids, bs_grid, _

def get_training_data_V2(input_folder, proteins_pkl='GOODproteins.pkl', binding_sites_pkl='binding_sites.pkl'):
    """
    Returns a np array containing the protein grids, one np array with the binding_sites grids,
    and the NOT centroid coordinates for each one, is returned empty on purpose. 
    """   
    # Try to load from pickle files first
    try:
        proteins = from_pickle(proteins_pkl)
        binding_sites = from_pickle(binding_sites_pkl)
        print("Data loaded from pickle files.")
        return proteins, binding_sites, []
    except (FileNotFoundError, pickle.UnpicklingError):
        print("No valid pickle files found, processing data...")

    proteins = None
    binding_sites = None
    centroids = []

    for root, dirs, _ in os.walk(input_folder, topdown=False):
        for dir in dirs:
            protein_file = os.path.join(root, dir, "protein.pdb")
            site_file = os.path.join(root, dir, "cavity6.pdb")
            
            print("Processing protein file:", protein_file)
            print("Processing binding site file:", site_file)

            try:
                # A new get_grids should be implemented and used here
                prot_grids, bs_grid, centroid = get_grids_V2("pdb", protein_file, site_file, grid_resolution=1, max_dist=7.5)
                if prot_grids is None:
                    print("Failed to create grid for:", protein_file)
                    continue
                else: 
                    for prot_grid in prot_grids:
                        proteins = np.concatenate((proteins, prot_grid), axis=0)
                    binding_sites = np.concatenate((binding_sites, bs_grid), axis=0)
                
                centroids.append(centroid)
               
            except Exception as e:
                print(f"Error processing {protein_file}: {e}")
        
    if proteins is not None:
        print("Number of proteins to train the model:", proteins.shape[0])
    else:
        print("No proteins found to train the model.")

    # Save the data to pickle files
    to_pickle(proteins, proteins_pkl)
    to_pickle(binding_sites, binding_sites_pkl)

    return proteins, binding_sites, centroids
