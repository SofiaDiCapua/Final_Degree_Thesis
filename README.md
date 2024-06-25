# Final_Degree_Thesis
Code for my final thesis about the evaluation an integration of ANN for prediction improvement of ligand-protein biding sites.

## DeepSurf
A surface-based deep learning approach for the prediction of ligand binding sites on proteins (https://doi.org/10.1093/bioinformatics/btab009)

Setup
---------------

Experiments were conducted on an Ubuntu 18.04 machine with Python 3.6.9 and CUDA 10.0 

1) Install dependencies
```
sudo apt-get update && apt-get install python3-venv, p7zip, swig, libopenbabel-dev, g++
```
2) Clone this repository
```
git clone https://github.com/stemylonas/DeepSurf
cd DeepSurf
```
3) Create environment and install python dependencies
```
python3 -m venv venv --prompt DeepSurf
source venv/bin/activate
pip install -r ./Final_Degree_Thesis/requirements.txt
```
4) Compile custom LDS module
```
cd lds
chmod a+x compile.sh
./compile.sh
cd ..
```
5) Download pretrained models
```
pip install gdown
gdown 1nIBoD3_5nuMqgRGx4G1OHZwLsiUjb7JG
p7zip -d models.7z
```
6) Collect and install DMS
```
wget www.cgl.ucsf.edu/Overview/ftp/dms.zip
unzip dms.zip
rm dms.zip
cd dms
sudo make install
cd ..
```

================================================================
Para reinstalar Python 3.6.15 y crear un entorno virtual con esa versión, sigue estos pasos:

1. **Descargar los archivos**: Asegúrate de tener los archivos tar y tgz de Python 3.6.15 en tu directorio de descargas:
https://www.python.org/downloads/release/python-3615/

2. **Extraer los archivos**:
    ```bash
    cd ~/Baixades
    tar -xzf Python-3.6.15.tgz
    ```

3. **Instalar Python 3.6.15**:
    ```bash
    cd Python-3.6.15
    ./configure --enable-optimizations --prefix=$HOME/python3.6.15
    make -j 8
    make altinstall
    ```

4. **Crear el entorno virtual**:
    ```bash
    cd ~/Documents/TFG
   /home/sofia-capua/python3.6.15/bin/python3.6 -m venv ~/Documents/TFG/tfg
    ```

5. **Activar el entorno virtual**:
    ```bash
    source tfg/bin/activate
    ```

6. **Actualizar pip y setuptools**:
    ```bash
    pip install --upgrade pip setuptools wheel
    ```

7. **Instalar los requerimientos**:
    ```bash
    pip install -r Final_Degree_Thesis/requirements.txt
    ```

Estos pasos deberían permitirte reinstalar Python 3.6.15 y crear un entorno virtual correctamente.

In case of "Segmentation fault (core dumped)" use: https://stackoverflow.com/questions/74553858/how-to-fix-python-pip-segmentation-fault-core-dumped-response-from-virtualenv
================================================================================

Usage example
---------------

```
python predict.py -p protein.pdb -mp model_path -o output_path
```

For more input options, check 'predict.py'. All other molecules (waters, ions, ligands) should be removed from the structure. If the input protein has not been protonated, add --protonate to the execution command.\
The provided models have been trained on a subset of scPDB (training_subset_of_scpdb.proteins)

Usage example in my computer
------------------------------

For DeepSurf:

```
python Final_Degree_Thesis/DeepSurf_Files/predict.py -p Final_Degree_Thesis/data/test/coach420/1a2k/protein.pdb -mp Final_Degree_Thesis/models -o .

python predict.py -p ../data/test/coach420/1a4k/protein.pdb -mp models -o 1a4k
```

For PUResNet:
```
python predict.py -ftype pdb -m 0 -i test.pdb -mpath whole_trained_model1.hdf -otype mol2 -o output -gpu 1
```
