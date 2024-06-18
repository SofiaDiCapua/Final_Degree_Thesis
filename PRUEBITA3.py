import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Asegúrate de que las funciones get_grids y get_grids_V2 están definidas antes de este script.

def visualize_grid(grid, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(grid, edgecolor='k')
    ax.set_title(title)
    plt.show()

def compare_grids(file_type, prot_input_file, bs_input_file=None):
    try:
        # Generar grids usando la función antigua
        prot_grid_old, bs_grid_old, centroid_old = get_grids(file_type, prot_input_file, bs_input_file, grid_resolution=1, max_dist=7.5)
        # Generar grids usando la nueva función
        prot_grids_new, bs_grid_new, centroid_new = get_grids_V2(file_type, prot_input_file, bs_input_file, grid_resolution=1, max_dist=7.5)

        # Visualizar los resultados
        if prot_grid_old is not None:
            print("Visualizando grid de la proteína usando la función antigua")
            visualize_grid(prot_grid_old, "Old Function - Protein Grid")

        if bs_grid_old is not None:
            print("Visualizando grid del sitio de unión usando la función antigua")
            visualize_grid(bs_grid_old, "Old Function - Binding Site Grid")

        if prot_grids_new is not None and len(prot_grids_new) > 0:
            print("Visualizando grid de la proteína usando la nueva función")
            visualize_grid(prot_grids_new[0], "New Function - Protein Grid")

        if bs_grid_new is not None:
            print("Visualizando grid del sitio de unión usando la nueva función")
            visualize_grid(bs_grid_new, "New Function - Binding Site Grid")

    except Exception as e:
        print(f"Error: {e}")

def main():
    file_type = 'pdb'
    prot_input_file = 'path/to/protein.pdb'  # Cambia esto por la ruta correcta
    bs_input_file = 'path/to/cavity6.pdb'  # Cambia esto por la ruta correcta, si es necesario

    compare_grids(file_type, prot_input_file, bs_input_file)

if __name__ == "__main__":
    main()
