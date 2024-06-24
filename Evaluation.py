# # FOR DEEPSURF'S EVALUATION
# import os
# import sys
# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from openbabel import pybel

# # Add folders to the PYTHONPATH
# sys.path.append(os.path.abspath('DeepSurf_Files'))
# sys.path.append(os.path.abspath('PUResNet_Files'))

# import tfbio_data 

# from PUResNet_Files.data import make_grid

# """
# This file is supossed to use a folder with outputs of DeepSurf algorithm with diferent ANN.
# The metric to evaluate each folder would be Success Rate with DCC
# DCC --> Distance Center Center  
# Success Rate --> Num of sites having DDC <= 4 Amstrongs  /  Total num of sites
# Total num of sites --> if centers.txt is not empty
# """

# # BEFORE RUNNING THIS FILE:
# # python multi_predict.py -p ../../data/test/Test4Both -mp models -o ../../data/Results4PLI
# # python multi_predict.py -p ../../data/test/Test4Both -mp ../../data -m PUResNet  -o ../../data/Results4PLI_1GRID
# # Change weights file in Network.py to test the huge model
# # python multi_predict.py -p ../../data/test/Test4Both -mp ../../data -m PUResNet  -o ../../data/Results4PLI_GRIDs


# def create_binding_site_features(p, bs_coords):
#     """
#     Create binding site features where coordinates match bs_coords.
#     """
#     bs_features = 0
#     if any(np.allclose(p, bs_coord, atol=4) for bs_coord in bs_coords):
#         bs_features = 1
#     return bs_features 

# def load_pdb_and_create_grid(file_path, featurizer, grid_resolution=1, max_dist=8):
#     bs = next(pybel.readfile('pdb', file_path))
#     bs_coords, _ = featurizer.get_features(bs)  
#     bs_features = create_binding_site_features(p, bs_coords) 
#     bs_grid = make_grid(bs_coords, bs_features, max_dist=max_dist, grid_resolution=grid_resolution)
#     return bs_grid


# # Función para obtener la lista de archivos .pdb en una carpeta y sus subcarpetas
# def get_pdb_files(directory, file_name, max_dirs=200):
#     pdb_files = []
#     dir_count = 0
#     processed_dirs = set()
    
#     for root, dirs, files in os.walk(directory):
#         for dir in dirs:
#             if dir_count >= max_dirs:
#                 break
#             if dir in processed_dirs:
#                 continue
            
#             protein_file = os.path.join(root, dir, file_name)
#             if os.path.isfile(protein_file):
#                 pdb_files.append(protein_file)
#                 processed_dirs.add(dir)
#                 dir_count += 1
        
#         if dir_count >= max_dirs:
#             break
    
#     return pdb_files

# def evaluate_directory(true_dir, pred_dir, featurizer, grid_resolution=1, max_dist=8, max_dirs=200):
#     true_files = get_pdb_files(true_dir, 'site.pdb', max_dirs=max_dirs)
#     pred_files = get_pdb_files(pred_dir, 'pocket1.pdb', max_dirs=max_dirs)

#     accuracies, precisions, recalls, f1_scores = [], [], [], []

#     for true_file, pred_file in zip(true_files, pred_files):
#         y_true_grid = load_pdb_and_create_grid(true_file, featurizer, grid_resolution, max_dist)
#         y_pred_grid = load_pdb_and_create_grid(pred_file, featurizer, grid_resolution, max_dist)
        
#         accuracy, precision, recall, f1 = calculate_metrics(y_true_grid, y_pred_grid)
        
#         accuracies.append(accuracy)
#         precisions.append(precision)
#         recalls.append(recall)
#         f1_scores.append(f1)
    
#     avg_accuracy = np.mean(accuracies)
#     avg_precision = np.mean(precisions)
#     avg_recall = np.mean(recalls)
#     avg_f1 = np.mean(f1_scores)

#     return avg_accuracy, avg_precision, avg_recall, avg_f1

# # Definir rutas de las carpetas que contienen las predicciones y las etiquetas verdaderas
# true_dir = "../data/test/Test4Both"
# pred_dir = "../data/Results4PLI"

# # Asegúrate de que tienes una instancia de featurizer definida en tu entorno
# featurizer = tfbio_data.Featurizer(save_molecule_codes=False)

# accuracy, precision, recall, f1 = evaluate_directory(true_dir, pred_dir, featurizer)
# print(f"Average Accuracy: {accuracy}")
# print(f"Average Precision: {precision}")
# print(f"Average Recall: {recall}")
# print(f"Average F1-Score: {f1}")

# # Almacenar resultados
# results = {
#     "Accuracy": accuracy,
#     "Precision": precision,
#     "Recall": recall,
#     "F1 Score": f1
# }

# # Crear un DataFrame con los resultados
# results_df = pd.DataFrame([results])

# # Mostrar la tabla de resultados
# print(results_df)

# # Guardar la tabla de resultados a un archivo CSV
# results_df.to_csv(os.path.join("../data", "DeepSurf_evaluation_results.csv"), index=False)


import numpy as np

def read_predicted_centers(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    centers = [list(map(float, line.strip().split())) for line in lines]
    return np.array(centers)

def read_real_centers_from_pdb(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    coords = []
    for line in lines:
        if line.startswith("ATOM"):
            parts = line.split()
            x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
            coords.append([x, y, z])
    coords = np.array(coords)
    center = np.mean(coords, axis=0)
    print("The center of the .pdb is : ", center)
    return center

def calculate_success_rate(predicted_centers, real_center, threshold=4.0):
    success_count = 0
    for pred_center in predicted_centers:
        distance = np.linalg.norm(pred_center - real_center)
        if distance <= threshold:
            success_count += 1
    print(success_count)
    print(len(predicted_centers))
    success_rate = success_count / len(predicted_centers) if len(predicted_centers) > 0 else 0
    return success_rate

def main():
    predicted_centers_file = '../data/Results4PLI/1ae1_2/protein/centers.txt'
    # HAY QUE VER QUE ARCHIVO .PDB DA EL MEJOR SUCCESS RATE
    real_centers_pdb_file = '../data/test/Test4Both/1ae1_2/ligand.pdb'
    
    predicted_centers = read_predicted_centers(predicted_centers_file)
    real_center = read_real_centers_from_pdb(real_centers_pdb_file)
    
    success_rate = calculate_success_rate(predicted_centers, real_center)
    print(f'Success Rate: {success_rate * 100:.2f}%')

if __name__ == "__main__":
    main()
