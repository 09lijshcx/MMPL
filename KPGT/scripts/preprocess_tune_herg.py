import sys
sys.path.append("..")

import numpy as np
from multiprocessing import Pool
from rdkit import Chem
from scipy import sparse as sp
import argparse 

from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args

def preprocess_dataset(smiless, name, n_jobs=16):
    # with open(f"{args.data_path}/smiles.smi", 'r') as f:
    #         lines = f.readlines()
    #         smiless = [line.strip('\n') for line in lines]

    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(f"/home/jovyan/prompts_learning/pretrain_data/{name}_rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"/home/jovyan/prompts_learning/pretrain_data/{name}_molecular_descriptors.npz",md=arr[:,1:])

if __name__ == '__main__':
   # load dataset
    import pandas as pd
    path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_train_data.csv"
    data = pd.read_csv(path)

    print(len(data))
    num = 0
    smi_list = []
    for i in range(len(data)):
        smi_list.append(data['smiles'][i])
        num += 1
    print(f"number of the valuable data is {num}.")

    path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_valid_data.csv"
    data = pd.read_csv(path)
    print(len(data))
    num = 0
    val_smi_list = []
    for i in range(len(data)):
        val_smi_list.append(data['smiles'][i])
        num += 1
    print(f"number of the valuable data is {num}.")

    path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week1_1201.csv"
    data = pd.read_csv(path)
    print(len(data))
    num = 0
    week1_smi_list = []
    for i in range(len(data)):
        week1_smi_list.append(data['smiles'][i])
        num += 1
    print(f"number of the valuable data is {num}.")

    path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week2_1201.csv"
    data = pd.read_csv(path)
    print(len(data))
    num = 0
    week2_smi_list = []
    for i in range(len(data)):
        week2_smi_list.append(data['smiles'][i])
        num += 1
    print(f"number of the valuable data is {num}.")

    path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week3_1201.csv"
    data = pd.read_csv(path)
    print(len(data))
    num = 0
    week3_smi_list = []
    for i in range(len(data)):
        week3_smi_list.append(data['smiles'][i])
        num += 1
    print(f"number of the valuable data is {num}.")


    path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week4_1201.csv"
    data = pd.read_csv(path)
    print(len(data))
    num = 0
    week4_smi_list = []
    for i in range(len(data)):
        week4_smi_list.append(data['smiles'][i])
        num += 1
    print(f"number of the valuable data is {num}.")


    
    preprocess_dataset(smi_list, "herg_cls_train")
    preprocess_dataset(val_smi_list, "herg_cls_val")
    preprocess_dataset(week1_smi_list, "herg_cls_week1")
    preprocess_dataset(week2_smi_list, "herg_cls_week2")
    preprocess_dataset(week3_smi_list, "herg_cls_week3")
    preprocess_dataset(week4_smi_list, "herg_cls_week4")