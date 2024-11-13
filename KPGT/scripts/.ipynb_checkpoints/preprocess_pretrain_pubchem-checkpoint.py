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

def preprocess_dataset(smiless, n_jobs=16):
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
    sp.save_npz(f"/home/jovyan/prompts_learning/pretrain_data/rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"/home/jovyan/prompts_learning/pretrain_data/molecular_descriptors.npz",md=arr[:,1:])

if __name__ == '__main__':
    # load original dataset
    import torch
    from torch_geometric.data import InMemoryDataset
    class PubChemDataset(InMemoryDataset):
        def __init__(self, path):
            super(PubChemDataset, self).__init__()
            self.data, self.slices = torch.load(path)

        def __getitem__(self, idx):
            return self.get(idx)


    smi_list = []
    # text_list = []
    # mol_list = []
    dataset = PubChemDataset('/home/jovyan/prompts_learning/pretrain_data/PubChem324kV2/pretrain.pt')
    for i in range(len(dataset)):
        smi = dataset[i]['smiles']
        # text = dataset[i]['text']
        smi_list.append(smi)
        # mol_list.append([smi, text])
    # args = parse_args()
    preprocess_dataset(smi_list)