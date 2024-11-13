from rdkit import Chem
import pandas as pd

# load dataset
# path = "/home/jovyan/PharmaBench/data/final_datasets/bbb_cls_final_data.csv"
path = "/home/jovyan/PharmaBench/data/final_datasets/logd_reg_final_data.csv"
data = pd.read_csv(path)

scaffold_training = data[data['scaffold_train_test_label'] == 'train']
scaffold_test = data[data['scaffold_train_test_label'] == 'test']

# random_training = data[data['random_train_test_label'] == 'train']
# random_test = data[data['scaffold_train_test_label'] == 'test']

scaffold_training = scaffold_training.reset_index()
scaffold_test = scaffold_test.reset_index()


import os
import numpy as np
import pandas as pd
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import pickle
import glob
from multiprocessing import Pool
from collections import defaultdict
def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi,cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def inner_smi2coords(content):
    smi = content
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d

    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt)
        # add 2d conf
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
    
    # 获取最短路径矩阵 SPD
    distance_matrix = AllChem.GetDistanceMatrix(mol) 
 
    # 获取边矩阵 Edge
    adjacency_matrix = AllChem.GetAdjacencyMatrix(mol)
    
    return pickle.dumps({'atoms': atoms, 'coordinates': coordinate_list, "SPD": distance_matrix,\
                         "edge": adjacency_matrix, 'smi': smi }, protocol=-1)


def smi2coords(content):
    try:
        return inner_smi2coords(content)
    except:
        print("failed smiles: {}".format(content[0]))
        return None


def write_lmdb(smiles_list, job_name, seed=42, outpath='./results', nthreads=8):
    os.makedirs(outpath, exist_ok=True)
    output_name = os.path.join(outpath,'{}.lmdb'.format(job_name))
    try:
        os.remove(output_name)
    except:
        pass
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for inner_output in tqdm(pool.imap(smi2coords, smiles_list)):
            if inner_output is not None:
                txn_write.put(f'{i}'.encode("ascii"), inner_output)
                i += 1
        print('{} process {} lines'.format(job_name, i))
        txn_write.commit()
        env_new.close()


        
print(len(scaffold_training['Smiles_unify']))
max_shape = 0 
num = 0
smi_list = []
for i in range(len(scaffold_training['Smiles_unify'])):
    shape = len(scaffold_training['Smiles_unify'][i])
    smi_list.append(scaffold_training['Smiles_unify'][i])
    num += 1
print(f"number of the valuable data is {num}.")


print(len(scaffold_test['Smiles_unify']))
max_shape = 0 
num = 0
test_smi_list = []
for i in range(len(scaffold_test['Smiles_unify'])):
    shape = len(scaffold_test['Smiles_unify'][i])
    test_smi_list.append(scaffold_test['Smiles_unify'][i])
    num += 1
print(f"number of the valuable data is {num}.")


seed = 42
data_path = './results'  # replace to your data path
batch_size=16
conf_size=11  # default 10 3d + 1 2d
results_path=data_path   # replace to your save path
print("start preprocessing...")
# job_name = 'bbb_train'
job_name = 'logd_train'
write_lmdb(smi_list, job_name=job_name, seed=seed, outpath=data_path)
# job_name = 'bbb_test'
job_name = 'logd_test'
write_lmdb(test_smi_list, job_name=job_name, seed=seed, outpath=data_path)
print("Generate successfully!")