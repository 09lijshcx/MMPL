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
    path = "/home/jovyan/PharmaBench/data/final_datasets/bbb_cls_final_data.csv"
    # path = "/home/jovyan/PharmaBench/data/final_datasets/logd_reg_final_data.csv"
    import pandas as pd
    data = pd.read_csv(path)

    scaffold_training = data[data['scaffold_train_test_label'] == 'train']
    scaffold_test = data[data['scaffold_train_test_label'] == 'test']
    scaffold_training = scaffold_training.reset_index()
    scaffold_test = scaffold_test.reset_index()

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


    
    preprocess_dataset(smi_list, "bbb_train")
    preprocess_dataset(test_smi_list, "bbb_test")