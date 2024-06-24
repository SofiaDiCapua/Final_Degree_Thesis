import os
import sys
import numpy as np
import pickle
import keras
import random
import keras.backend as K
from openbabel import pybel
from sklearn.model_selection import train_test_split

# Add folders to the PYTHONPATH
sys.path.append(os.path.abspath('DeepSurf_Files'))
sys.path.append(os.path.abspath('PUResNet_Files'))

import tfbio_data 
from PUResNet_Files.data import Featurizer, make_grid
from PUResNet_Files.PUResNet import PUResNet
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


def create_binding_site_features(p, bs_coords):
    """
    Create binding site features where coordinates match bs_coords.
    """
    bs_features = 0
    if any(np.allclose(p, bs_coord, atol=4) for bs_coord in bs_coords):   
        bs_features = 1
    return bs_features

def reduce_surface_points(surf_points, bs_list, num_points):
    """
    Reduce the number of surface points randomly, excluding those in bs_list.
    """
    # Flatten bs_list to get a list of coordinates
    bs_coords = [item[1] for item in bs_list]
    # Filter out points that are in bs_coords
    filtered_points = [p for p in surf_points if not any(np.allclose(p, bs_coord, atol=4) for bs_coord in bs_coords)]
    # If there are not enough points to reduce to the desired number, return all filtered points
    if len(filtered_points) <= num_points:
        return filtered_points
    
    # Randomly sample the required number of points
    reduced_points = random.sample(filtered_points, num_points)
    
    return reduced_points

def get_grids_V2(file_type, prot_input_file, bs_input_file=None,
                 grid_resolution=2, max_dist=35, 
                 featurizer=Featurizer(save_molecule_codes=False)):
    """
    Converts both a protein file (PDB or mol2) and its ligand (if specified)
    to a grid.
    """
    prot_input_file = prot_input_file.replace('.ipynb_checkpoints/', '')
    if bs_input_file:
        bs_input_file = bs_input_file.replace('.ipynb_checkpoints/', '')

    if not os.path.exists(prot_input_file):
        raise IOError("No such file: '%s'" % prot_input_file)
    if bs_input_file and not os.path.exists(bs_input_file):
        raise IOError("No such file: '%s'" % bs_input_file)

    # Set grid size and voxel size
    gridSize = 16
    voxelSize = 1
    featurizer = KalasantyFeaturizer(gridSize, voxelSize) 
    featurizer4bs = tfbio_data.Featurizer(save_molecule_codes=False)

    # Get binding site coordinates and features
    bs = next(pybel.readfile(file_type, bs_input_file))
    bs_coords, _ = featurizer4bs.get_features(bs)

    # Create Protein object and process ASA with DMS and K-Means
    protein = Protein(prot_file=prot_input_file, protonate=False, expand_residue=False, f=10, save_path="../data/ShitGeneratedBYdefault", discard_points=False)

    featurizer.get_channels(protein.mol)

    # Find the binding site points
    bs_list = []
    for idx, p in enumerate(protein.surf_points):
        if any(np.allclose(p, bs_coord, atol=4) for bs_coord in bs_coords):
            bs_list.append(p)

    # Reduce the surface points
    num_bs_points = len(bs_list)
    print("Num bs points: ", num_bs_points)
    # Reducing surface points to get 50/50 class balance data
    reduced_points = reduce_surface_points(protein.surf_points, bs_list, num_bs_points)

    # Combine bs_list and reduced_points for grid creation
    combined_points = bs_list + reduced_points
    
    # Assert that the number of reduced points is equal to the number of binding site points
    assert len(reduced_points) == len(bs_list), f"Error: The number of reduced points ({len(reduced_points)}) is not equal to the number of binding site points ({len(bs_list)})."

    # Create grids for combined points
    prot_grids = []
    bs_grids = []

    for p in combined_points:
        normal_idx = np.where(np.all(protein.surf_points == p, axis=1))[0][0]
        n = protein.surf_normals[normal_idx]
        # Create the protein grid
        grid = featurizer.grid_feats(p, n, protein.heavy_atom_coords)
        prot_grids.append(grid)

        bs_features = create_binding_site_features(p, bs_coords)
        # Create the binding site grid
        bs_grid = featurizer.grid_feats(p, n, protein.heavy_atom_coords, bs_features)
        bs_grids.append(bs_grid)

    # Convert lists of grids to numpy arrays
    prot_grids = np.array(prot_grids)
    bs_grids = np.array(bs_grids)
    print("All grids converted to numpy arrays.")

    return prot_grids, bs_grids, reduced_points


def get_training_data_V2(input_folder, proteins_pkl='GOODproteins.pkl', binding_sites_pkl='GOODbinding_sites.pkl'):
    """
    Processes each protein directory to generate and save protein and binding site grids.
    """   
    proteins = None
    binding_sites = None
    count = 0
    for root, dirs, _ in os.walk(input_folder, topdown=False):
        for dir in dirs[:1000]:
            protein_file = os.path.join(root, dir, "protein.pdb")
            site_file = os.path.join(root, dir, "ligand.pdb")
            protein_pkl = os.path.join(root, dir, proteins_pkl)
            binding_site_pkl = os.path.join(root, dir, binding_sites_pkl)
            
            print("Processing directory:", os.path.join(root, dir))
            print("Protein file:", protein_file)
            print("Binding site file:", site_file)
            print("===============================================")
            print("MISSING proteins to process:", len(dirs[:1000]) - count)
            count += 1

            if os.path.exists(protein_pkl) and os.path.exists(binding_site_pkl):
                print(f"Pickle files already exist for {dir}. Skipping processing.")
                prot_grids = from_pickle(protein_pkl)
                bs_grid = from_pickle(binding_site_pkl)
                if proteins is None:
                    proteins = prot_grids
                    binding_sites = bs_grid
                else:
                        proteins = np.concatenate((proteins, prot_grids), axis=0)
                        binding_sites = np.concatenate((binding_sites, bs_grid), axis=0)
                continue

            try:
                print("No valid pickle files found, processing data...")
                # Calculate grids if the pickle files do not exist
                prot_grids, bs_grid, centroid = get_grids_V2("pdb", protein_file, site_file, grid_resolution=1, max_dist=7.5)
                print("Prot_grid_new : \n" , prot_grids.shape)
                print("BS_grid_new : \n" , bs_grid.shape)
                if prot_grids is None:
                    print(f"Failed to create grid for {protein_file}. Skipping directory.")
                    continue

                if proteins is None:
                    proteins = prot_grids
                    binding_sites = bs_grid
                else:
                    proteins = np.concatenate((proteins, prot_grids), axis=0)
                    binding_sites = np.concatenate((binding_sites, bs_grid), axis=0)
                
                # Save the grids to pickle files
                to_pickle(prot_grids, protein_pkl)
                to_pickle(bs_grid, binding_site_pkl)
                print(f"Pickle files saved for {dir}.")

            except Exception as e:
                print(f"Error processing {protein_file}: {e}")

    print("Processing complete.")

    return proteins, binding_sites, _