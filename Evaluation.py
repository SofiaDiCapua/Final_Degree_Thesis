# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import pandas as pd
# import numpy as np

# # Definir rutas de los modelos y la carpeta de test
# model_paths = [
#     "path/to/model1.h5",
#     "path/to/model2.h5",
#     "path/to/model3.h5",
#     "path/to/model4.h5"
# ]
# test_data_dir = "path/to/test_data"


# # Inicializar una lista para almacenar los resultados
# results = []

# # Evaluar cada modelo
# for model_path in model_paths:
#     model = load_model(model_path)
#     y_pred_prob = model.predict(X_test)
#     y_pred = np.argmax(y_pred_prob, axis=1)

#     # Calcular métricas
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='binary')
#     recall = recall_score(y_true, y_pred, average='binary')
#     f1 = f1_score(y_true, y_pred, average='binary')
    
#     # Almacenar resultados
#     results.append({
#         "Model": model_path,
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall,
#         "F1 Score": f1
#     })

# # Crear un DataFrame con los resultados
# results_df = pd.DataFrame(results)

# # Mostrar la tabla de resultados
# print(results_df)

# # Guardar la tabla de resultados a un archivo CSV
# results_df.to_csv("model_evaluation_results.csv", index=False)

# FOR DEEPSURF'S EVALUATION
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Bio.PDB import PDBParser

# Función para extraer información de etiquetas desde archivos .pdb
def extract_labels_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb_file)
    # Ejemplo de extracción de etiquetas basado en el nombre del archivo.
    # Reemplaza esto con tu lógica específica para extraer las etiquetas correctas.
    label = int(pdb_file.split('_')[-1].replace('.pdb', ''))  # Ejemplo ficticio
    return label

# Definir rutas de las carpetas que contienen las predicciones y las etiquetas verdaderas
y_true_dir = "../data/test/Test4Both"
y_pred_dir = "../data/Results4PLI"

# Función para obtener la lista de archivos .pdb en una carpeta y sus subcarpetas
def get_pdb_files(directory, file_name, max_dirs=200):
    pdb_files = []
    dir_count = 0
    processed_dirs = set()
    
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir_count >= max_dirs:
                break
            if dir in processed_dirs:
                continue
            
            protein_file = os.path.join(root, dir, file_name)
            if os.path.isfile(protein_file):
                pdb_files.append(protein_file)
                processed_dirs.add(dir)
                dir_count += 1
        
        if dir_count >= max_dirs:
            break
    
    return pdb_files
# Leer las predicciones y las etiquetas verdaderas
y_pred_files = get_pdb_files(y_pred_dir, file_name="pocket1.pdb")
y_true_files = get_pdb_files(y_true_dir, file_name= "site.pdb")

# Inicializar diccionarios para almacenar las predicciones y etiquetas verdaderas
y_pred_dict = {}
y_true_dict = {}

# Cargar las predicciones y etiquetas verdaderas
for y_pred_file in y_pred_files:
    protein_name = os.path.basename(os.path.dirname(y_pred_file))
    y_pred_dict[protein_name] = extract_labels_from_pdb(y_pred_file)

for y_true_file in y_true_files:
    protein_name = os.path.basename(os.path.dirname(y_true_file))
    y_true_dict[protein_name] = extract_labels_from_pdb(y_true_file)

# Asegurarse de que solo se comparan las proteínas que están en ambas listas
common_proteins = set(y_pred_dict.keys()).intersection(y_true_dict.keys())

# Inicializar listas para almacenar las predicciones y etiquetas verdaderas
y_pred = []
y_true = []

# Rellenar las listas con las etiquetas correspondientes
for protein in common_proteins:
    y_pred.append(y_pred_dict[protein])
    y_true.append(y_true_dict[protein])

# Convertir las listas a arrays de numpy
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Calcular métricas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

# Almacenar resultados
results = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

# Crear un DataFrame con los resultados
results_df = pd.DataFrame([results])

# Mostrar la tabla de resultados
print(results_df)

# Guardar la tabla de resultados a un archivo CSV
results_df.to_csv(os.path.join("../data", "DeepSurf_evaluation_results.csv"), index=False)
