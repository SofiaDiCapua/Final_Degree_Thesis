import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from train_functions import get_grids, get_grids_V2

# Add folders to the PYTHONPATH
sys.path.append(os.path.abspath('DeepSurf_Files'))
sys.path.append(os.path.abspath('PUResNet_Files'))


def visualize_grid(grid, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(grid, edgecolor='k')
    ax.set_title(title)
    plt.show()

def compare_grids(file_type, prot_input_file, bs_input_file=None):
    try:
        print("Getting old grid")
        # Generar grids usando la función antigua
        prot_grid_old, bs_grid_old, centroid_old = get_grids(file_type, prot_input_file, bs_input_file, grid_resolution=1, max_dist=7.5)
        print("Old grid achieved")
        # Generar grids usando la nueva función
        print("Getting new grids")
        prot_grids_new, bs_grid_new, centroid_new = get_grids_V2(file_type, prot_input_file, bs_input_file, grid_resolution=1, max_dist=7.5)
        print("New grids achieved")


        #print("Prot_grid_old : \n" , prot_grid_old.shape)
        print("Prot_grid_new : \n" , prot_grids_new.shape)
        #print("BS_grid_old : \n" , bs_grid_old.shape)
        print("BS_grid_new : \n" , bs_grid_new.shape)
        print(f'Tamaño de prot_grids_new: {sys.getsizeof(prot_grids_new)} bytes')
        print(f'Tamaño de bs_grids_new: {sys.getsizeof(bs_grid_new)} bytes')

        # # Visualizar los resultados
        # if prot_grid_old is not None:
        #     print("Visualizando grid de la proteína usando la función antigua")
        #     print(prot_grid_old)
        #     visualize_grid(prot_grid_old, "Old Function - Protein Grid")

        # if bs_grid_old is not None:
        #     print("Visualizando grid del sitio de unión usando la función antigua")
        #     visualize_grid(bs_grid_old, "Old Function - Binding Site Grid")

        # if prot_grids_new is not None and len(prot_grids_new) > 0:
        #     print("Visualizando grid de la proteína usando la nueva función")
        #     visualize_grid(prot_grids_new[0], "New Function - Protein Grid")

        # if bs_grid_new is not None:
        #     print("Visualizando grid del sitio de unión usando la nueva función")
        #     visualize_grid(bs_grid_new, "New Function - Binding Site Grid")

    except Exception as e:
        print(f"Error: {e}")

def main():
    file_type = 'pdb'
    prot_input_file = '../data/test/coach420/1a4k/protein.pdb'  # Cambia esto por la ruta correcta
    bs_input_file = '../data/test/coach420/1a4k/ligand.pdb'  # Cambia esto por la ruta correcta, si es necesario

    compare_grids(file_type, prot_input_file, bs_input_file)

if __name__ == "__main__":
    main()

