import os
import shutil

def obtener_nombres_carpetas(ruta_carpeta):
    nombres_carpetas = []
    for item in os.listdir(ruta_carpeta):
        item_path = os.path.join(ruta_carpeta, item)
        if os.path.isdir(item_path):
            # Eliminar la parte después del guion bajo
            nombres_carpetas.append(item.split('_')[0])
    return nombres_carpetas


def leer_nombres_archivo(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        nombres = archivo.read().splitlines()
    return nombres


def obtener_nombres_carpetas(ruta_carpeta):
    nombres_carpetas = []
    for item in os.listdir(ruta_carpeta):
        item_path = os.path.join(ruta_carpeta, item)
        if os.path.isdir(item_path):
            nombre_sin_extra = item.split('_')[0]  # Eliminar la parte después del guion bajo
            nombres_carpetas.append(nombre_sin_extra)
    return nombres_carpetas


def mover_carpetas(lista_proteinas, carpeta_origen, carpeta_destino):
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    
    moved_folders = set()  # Para asegurarnos de mover solo las primeras 250 únicas proteínas
    
    for proteina in lista_proteinas:
        for item in os.listdir(carpeta_origen):
            if item.startswith(proteina) and proteina not in moved_folders:
                origen_path = os.path.join(carpeta_origen, item)
                destino_path = os.path.join(carpeta_destino, item)
                shutil.move(origen_path, destino_path)
                moved_folders.add(proteina)
                print(f"Carpeta {item} movida a {carpeta_destino}")
                if len(moved_folders) >= 250:
                    return


ruta_archivo = "../data/training_subset_of_scpdb.proteins"  # Reemplaza con la ruta de tu archivo
nombres_proteinas = leer_nombres_archivo(ruta_archivo)

ruta_carpeta = "../data/train/final_data"  # Reemplaza con la ruta de tu carpeta
nombres_carpetas = obtener_nombres_carpetas(ruta_carpeta)

# Convertir la segunda lista a un conjunto para mejorar la eficiencia de búsqueda
set2 = set(nombres_proteinas)

# Contar los elementos de lista1 que no están en lista2
elementos_no_en_lista2 = [elemento for elemento in nombres_carpetas if elemento not in set2]
cantidad_no_en_lista2 = len(elementos_no_en_lista2)
print(f"Cantidad de elementos en lista1 que no están en lista2: {cantidad_no_en_lista2}")

# Mover las carpetas
mover_carpetas(elementos_no_en_lista2, carpeta_origen=ruta_carpeta, carpeta_destino="../data/test/Test4Both")

# Checking that the changes were done succesfully
nombres_carpetas_movidas = obtener_nombres_carpetas("../data/test/Test4Both")
set2 = set(nombres_carpetas)
elementos_no_en_lista2 = [elemento for elemento in nombres_carpetas_movidas if elemento not in set2]
cantidad_no_en_lista2 = len(elementos_no_en_lista2)
print(f"Cantidad de elementos en lista1 que no están en lista2: {cantidad_no_en_lista2}")

