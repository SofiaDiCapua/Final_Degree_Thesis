import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# Definir rutas de los modelos y la carpeta de test
model_paths = [
    "path/to/model1.h5",
    "path/to/model2.h5",
    "path/to/model3.h5",
    "path/to/model4.h5"
]
test_data_dir = "path/to/test_data"


# Inicializar una lista para almacenar los resultados
results = []

# Evaluar cada modelo
for model_path in model_paths:
    model = load_model(model_path)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Almacenar resultados
    results.append({
        "Model": model_path,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results)

# Mostrar la tabla de resultados
print(results_df)

# Guardar la tabla de resultados a un archivo CSV
results_df.to_csv("model_evaluation_results.csv", index=False)
