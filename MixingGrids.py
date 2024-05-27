import numpy as np
from math import ceil
import pybel

from tfbio_data import Featurizer

# Estas funciones las ha hecho CHtgpt y se nota que estan mal
# EL objetivo de este file es caragr las proteinas como estan en PUResNet
# pero que se haga la grid como en DeepSurf
# EN ambos papapers usaron el mismo featurizer !!!!

def make_grid(coords, features, grid_resolution=1.0, max_dist=7.5):
    """Convert atom coordinates and features represented as 2D arrays into a
    fixed-sized 3D box.

    Parameters
    ----------
    coords, features: array-likes, shape (N, 3) and (N, F)
        Arrays with coordinates and features for each atoms.
    grid_resolution: float, optional
        Resolution of a grid (in Angstroms).
    max_dist: float, optional
        Maximum distance between atom and box center. Resulting box has size of
        2*`max_dist`+1 Angstroms and atoms that are too far away are not
        included.

    Returns
    -------
    coords: np.ndarray, shape = (M, M, M, F)
        4D array with atom properties distributed in 3D space. M is equal to
        2 * `max_dist` / `grid_resolution` + 1
    """

    try:
        coords = np.asarray(coords, dtype=np.float32)
    except ValueError:
        raise ValueError('coords must be an array of floats of shape (N, 3)')
    c_shape = coords.shape
    if len(c_shape) != 2 or c_shape[1] != 3:
        raise ValueError('coords must be an array of floats of shape (N, 3)')

    N = len(coords)
    try:
        features = np.asarray(features, dtype=np.float32)
    except ValueError:
        raise ValueError('features must be an array of floats of shape (N, F)')
    f_shape = features.shape
    if len(f_shape) != 2 or f_shape[0] != N:
        raise ValueError('features must be an array of floats of shape (N, F)')

    if not isinstance(grid_resolution, (float, int)):
        raise TypeError('grid_resolution must be float or int')
    if grid_resolution <= 0:
        raise ValueError('grid_resolution must be positive')

    if not isinstance(max_dist, (float, int)):
        raise TypeError('max_dist must be float or int')
    if max_dist <= 0:
        raise ValueError('max_dist must be positive')

    num_features = f_shape[1]
    max_dist = float(max_dist)
    grid_resolution = float(grid_resolution)

    box_size = ceil(2 * max_dist / grid_resolution + 1)

    # move all atoms to the nearest grid point
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)

    # remove atoms outside the box
    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    grid = np.zeros((box_size, box_size, box_size, num_features), dtype=np.float32)
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[x, y, z] += f

    return grid

def get_grids(file_type, prot_input_file, bs_input_file=None,
              grid_resolution=1.0, max_dist=7.5, 
              featurizer=None):
    '''
    Converts both a protein file (PDB or mol2) and its ligand (if specified)
    to a grid.
        
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
    featurizer: Featurizer object, optional
        Object to extract features from molecules.
    
    Returns
    -------
    prot_grid: np.ndarray
        4D array with protein atom properties distributed in 3D space.
    bs_grid: np.ndarray
        4D array with binding site properties distributed in 3D space.
    centroid: np.ndarray
        Coordinates of the centroid of the protein.
    '''

    if featurizer is None:
        featurizer = Featurizer()

    # Convert file into pybel object and get the features of the molecule.
    prot = next(pybel.readfile(file_type, prot_input_file))
    prot_coords, prot_features = featurizer.get_features(prot)
    
    # Change all coordinates to be relative to the center of the protein
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid

    # Create the grid for the protein
    prot_grid = make_grid(prot_coords, prot_features,
                          max_dist=max_dist,
                          grid_resolution=grid_resolution)
    
    # Do the same for the binding site, if input file specified
    if bs_input_file is not None:
        bs = next(pybel.readfile(file_type, bs_input_file))
        bs_coords, _ = featurizer.get_features(bs)
        # Binding site just has 1 feature: an array of 1s for each atom, indicating the
        # atom is present in that position
        bs_features = np.ones((len(bs_coords), 1))
        bs_coords -= centroid
        bs_grid = make_grid(bs_coords, bs_features,
                            max_dist=max_dist,
                            grid_resolution=grid_resolution)
    else:
        bs_grid = None
    
    return prot_grid, bs_grid, centroid
