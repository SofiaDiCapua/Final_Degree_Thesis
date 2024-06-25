import os
import sys
import numpy as np
from openbabel import pybel

"""
This file is supossed to use a folder with outputs of DeepSurf algorithm with diferent ANN.
The metric to evaluate each folder would be Success Rate with DCC
DCC --> Distance Center Center  
Success Rate --> Num of sites having DDC <= 4 Amstrongs  /  Total num of sites
Total num of sites --> if centers.txt is not empty

DEEPSURF ORIGINAL:
Average Success Rate: 70.15%
Average Success Rate Top-1: 75.82%
F1 Score: 0.76
"""

# BEFORE RUNNING THIS FILE:
# python multi_predict.py -p ../../data/test/Test4Both -mp models -o ../../data/Results4PLI
# python multi_predict.py -p ../../data/test/Test4Both -mp ../../data -m PUResNet  -o ../../data/Results4PLI_1GRID
# Change weights file in Network.py to test the huge model
# python multi_predict.py -p ../../data/test/Test4Both -mp ../../data -m PUResNet  -o ../../data/Results4PLI_GRIDs


def get_files(directory, subfolder, file_name, max_dirs=100):
    pdb_files = []
    dir_count = 0
    processed_dirs = set()
    
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir_count >= max_dirs:
                break
            if dir in processed_dirs:
                continue
            
            protein_file = os.path.join(root, dir, subfolder, file_name)
            if os.path.isfile(protein_file):
                pdb_files.append(protein_file)
                processed_dirs.add(dir)
                dir_count += 1
        
        if dir_count >= max_dirs:
            break
    
    return pdb_files

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
    return center

def calculate_success_rate(predicted_centers, real_center, threshold=4.0):
    success_count = 0
    for pred_center in predicted_centers:
        distance = np.linalg.norm(pred_center - real_center)
        if distance <= threshold:
            success_count += 1
    success_rate = success_count / len(predicted_centers) if len(predicted_centers) > 0 else 0
    return success_rate

def calculate_success_rate_bestBS(predicted_centers, real_center, threshold=4.0):
    if len(predicted_centers) == 0:
        return 0
    # Calcula todas las distancias y selecciona la menor
    distances = [np.linalg.norm(pred_center - real_center) for pred_center in predicted_centers]
    min_distance = min(distances)
    # Comprueba si la menor distancia está dentro del umbral
    success_count = 1 if min_distance <= threshold else 0
    success_rate = success_count / 1  # Solo se evalúa el mejor sitio predicho
    return success_rate


def calculate_metrics(predicted_centers, real_center, threshold=4.0):
    tp = 0
    fp = 0
    fn = 0  # If there's at least one prediction, then FN is 0
    if predicted_centers is None:
        fn = 1 
    distances = [np.linalg.norm(pred_center - real_center) for pred_center in predicted_centers]
    min_distance = min(distances)
    if min_distance <= threshold:
        tp += 1
    else:
        fp += 1
    return tp, fp, fn

def evaluate_directory(true_dir, pred_dir, max_dirs=200):
    true_files_dict = {os.path.basename(os.path.dirname(f)): f for f in get_files(true_dir, '', 'ligand.pdb', max_dirs=max_dirs)}
    pred_files_dict = {os.path.basename(os.path.dirname(os.path.dirname(f))): f for f in get_files(pred_dir, 'protein', 'centers.txt', max_dirs=max_dirs)}

    success_rates1 = []
    success_rates2 = []
    tps, fps, fns = 0, 0, 0
    common_keys = set(true_files_dict.keys()).intersection(set(pred_files_dict.keys()))

    for key in common_keys:
        true_file = true_files_dict[key]
        pred_file = pred_files_dict[key]
        
        real_center = read_real_centers_from_pdb(true_file)
        predicted_centers = read_predicted_centers(pred_file)
        success_rate1 = calculate_success_rate(predicted_centers, real_center)
        success_rate2 = calculate_success_rate_bestBS(predicted_centers, real_center)
        success_rates1.append(success_rate1)
        success_rates2.append(success_rate2)
        
        tp, fp, fn = calculate_metrics(predicted_centers, real_center)
        tps += tp
        fps += fp
        fns += fn

    avg_success_rate1 = np.mean(success_rates1) if success_rates1 else 0
    avg_success_rate2 = np.mean(success_rates2) if success_rates2 else 0
    precision = tps / (tps + fps) if (tps + fps) > 0 else 0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    return avg_success_rate1, avg_success_rate2, precision, recall, f1

# Definir rutas de las carpetas que contienen las predicciones y las etiquetas verdaderas
true_dir = "../data/test/Test4Both"
pred_dir = "../data/Results4PLI_1GRIDs"

avg_success_rate1, avg_success_rate2, precision, recall, f1 = evaluate_directory(true_dir, pred_dir)
print(f"Average Success Rate: {avg_success_rate1 * 100:.2f}%")
print(f"Average Success Rate Top-1: {avg_success_rate2 * 100:.2f}%")
print(f"Precision: {f1:.2f}")
print(f"Recall: {f1:.2f}")
print(f"F1 Score: {f1:.2f}")