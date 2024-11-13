import lmdb
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import pickle

import pandas as pd
data = pd.read_csv("/home/jovyan/prompts_learning/bbb_cls_final_data_multi_class.csv")
# data = pd.read_csv("/home/jovyan/PharmaBench/data/final_datasets/bbb_cls_final_data.csv")
scaffold_training = data[data['scaffold_train_test_label'] == 'train']
scaffold_test = data[data['scaffold_train_test_label'] == 'test']
scaffold_training = scaffold_training.reset_index()
scaffold_test = scaffold_test.reset_index()



class Pretrain_LMDBDataset(Dataset):
    def __init__(self, conf, pad_len=150, mode="herg"):
        if mode == "herg":
            db_path = "/home/jovyan/prompts_learning/results/herg_cls_train.lmdb"
            csv_path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_train_data.csv"
            pd_data = pd.read_csv(csv_path)
        elif mode == "bbb":
            db_path = "/home/jovyan/prompts_learning/results/bbb_train.lmdb"
            csv_path = "/home/jovyan/prompts_learning/bbb_cls_final_data_multi_class.csv"
            data = pd.read_csv(csv_path)
            scaffold_training = data[data['scaffold_train_test_label'] == 'train']
            scaffold_training = scaffold_training.reset_index()
            pd_data = scaffold_training
        elif mode == "logd":
            db_path = "/home/jovyan/prompts_learning/results/logd_train.lmdb"
            csv_path = "/home/jovyan/PharmaBench/data/final_datasets/logd_reg_final_data.csv"
            data = pd.read_csv(csv_path)
            scaffold_training = data[data['scaffold_train_test_label'] == 'train']
            scaffold_training = scaffold_training.reset_index()
            pd_data = scaffold_training
        elif mode == "bbb_test":
            db_path = "/home/jovyan/prompts_learning/results/bbb_test.lmdb"
            csv_path = "/home/jovyan/prompts_learning/bbb_cls_final_data_multi_class.csv"
            data = pd.read_csv(csv_path)
            scaffold_training = data[data['scaffold_train_test_label'] == 'test']
            scaffold_training = scaffold_training.reset_index()
            pd_data = scaffold_training
        
        
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        
        # get correponding label from csv
        self.value_list = []
        self.max_len = 0

        
        self.max_value = 0.
        self.min_value = 999.
        for i in range(len(self._keys)):
            datapoint_pickled = env.begin().get(self._keys[i])
            data = pickle.loads(datapoint_pickled)
            current_len = len(data['atoms'])
            if self.max_len < current_len:
                self.max_len = current_len
            if mode == "herg":
                idx = pd_data['smiles'][pd_data['smiles']==data['smi']].index[0]
                self.value_list.append(pd_data['class'][idx])
            else:
                idx = pd_data['Smiles_unify'][pd_data['Smiles_unify']==data['smi']].index[0]
                self.value_list.append(pd_data['value'][idx])
            
        self.atom_dict = {'Br': 0, 'C': 1, 'H': 2, 'N': 3, 'O': 4, 'F': 5, 'Cl': 6, 'S': 7, 'P': 8, 'I': 9, 'B': 10, 'Se': 11, 'Ar': 12, 'Kr': 13, 'Li': 14, 'Ne': 15, 'Xe': 16, 'Si': 17, 'Na': 18, 'Mask': 19, "Pad": 20}
        print(self.atom_dict)
        self.mask_token_id = self.atom_dict['Mask']
        self.pad_token_id = self.atom_dict["Pad"]
        
        self.pad_len = pad_len
        self.conf = conf # conformation
        self.mode = mode
        print(f"{mode} set is initialized successfully. The max length of the atom is {self.max_len}. The number of dataset is {len(self._keys)}. Padding length is {self.pad_len}.")
        
                    

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)
    
    def min_max_norm(self, x):
        _min = x.min()
        _max = x.max()
        x = (x - _min) / (_max - _min)
        return x
    
    def random_mask(self, atoms, mask_ratio=0.75):
        # print("before", atoms)
        len_masked = int(len(atoms) * mask_ratio)
        noise = np.random.rand(len(atoms))  # noise in [0, 1]
        
        # sort noise for each sample
        ids_masked = np.argsort(noise)[:len_masked]  # ascend: small is keep, large is remove
        output_label = []
        
        output_label = atoms[ids_masked]
        atoms[ids_masked] = self.mask_token_id
        # print("masked", atoms)
        # print("lab", output_label)

        return atoms, output_label, ids_masked

    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        coordinates = torch.tensor(np.array(data['coordinates']), dtype=torch.float32)[:self.conf, :, :]
        emb_idx = torch.tensor([self.atom_dict[atom] for atom in data['atoms']], dtype=torch.long)
        cur_len = len(emb_idx)
        
        # shortest path distance
        spd = torch.tensor(np.array(data["SPD"]), dtype=torch.float32)
        edge = torch.tensor(np.array(data["edge"]), dtype=torch.float32) + torch.eye(cur_len)
        
        # random dropout atoms
        if cur_len > 100:
            num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num = random.choice(num_list)
            emb_idx = emb_idx[-num:]
            coordinates = coordinates[:, -num:,:]
            spd = spd[-num:, -num:]
            edge = edge[-num:, -num:]
            cur_len = len(emb_idx)
            
        
        # padding
        if cur_len < self.pad_len:
            new_emb = torch.full(size=(self.pad_len,), fill_value=self.pad_token_id, dtype=torch.long)
            new_emb[:cur_len] = emb_idx
            
            new_cor = torch.full(size=(self.conf, self.pad_len, 3), fill_value=0, dtype=torch.float32)
            new_cor[:, :cur_len, :] = coordinates
            
            new_spd = torch.full(size=(self.pad_len, self.pad_len), fill_value=0, dtype=torch.float32)
            new_spd[:cur_len, :cur_len] = spd
            new_edge = torch.full(size=(self.pad_len, self.pad_len), fill_value=0, dtype=torch.float32)
            new_edge[:cur_len, :cur_len] = edge
        elif cur_len >= self.pad_len:
            new_emb = emb_idx[:self.pad_len]
            new_cor = coordinates[:, :self.pad_len, :]
            new_spd = spd[:self.pad_len, :self.pad_len]
            new_edge = edge[:self.pad_len, :self.pad_len]
        
        # Normalize and augment coordination
      # for random augmentation
        weight_list = [0.1, 0.2, 0.3, 0.5, 0.7, 0.01, 0.001]
        # weight_list = [0.1, 0.2, 0.3, 0.01, 0.001]
        scale = random.choice(weight_list) 
        noise = scale * torch.randn_like(new_cor)
        if np.random.rand() < 0.5:  # for add noise locally
            mask = torch.randint_like(noise, 0, 2, dtype=torch.float32)
            noise = noise * mask
        new_cor = new_cor + noise
            
            
        # to compute the pair relative distance
        atom_expanded = new_cor.unsqueeze(2)  # shape (conf, pad_len, 1, 3)
        coor_expanded = new_cor.unsqueeze(1)   # shape (conf, 1, pad_len, 3)
        # distance = atom_expanded - coor_expanded
        # distance = distance.permute(1, 0, 2, 3).reshape(-1, conf*pad_len*3)   # xyz 
        distance = torch.sqrt((atom_expanded - coor_expanded).pow(2).sum(dim=-1))   # x+y+z
        distance = distance.permute(1, 0, 2).reshape(-1, self.conf*self.pad_len)
        distance = (self.min_max_norm(distance) - 0.5) / 0.5
        
        label = torch.tensor(self.value_list[idx], dtype=torch.float32)
        
        new_cor = (self.min_max_norm(new_cor) - 0.5) / 0.5
        new_cor = new_cor.permute(1, 0, 2).reshape(-1, self.conf*3)
        
        # for pre-training task 1: atom prediction
        new_emb, masked_label, ids_masked = self.random_mask(new_emb)
        # for pre-training task 2: noised distance prediction
        # weight_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        weight_list = [0]
        scale = random.choice(weight_list) 
        noise_gt = scale * torch.randn_like(distance)
        distance = distance # + noise_gt
       
        return {"atoms": new_emb, "coordinate": new_cor, "distance": distance, "SPD": new_spd, "edge": new_edge, "label": label,\
               "masked_label": masked_label, "ids_masked": ids_masked, "noise_gt": noise_gt}

        

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

        
if __name__ == "__main__":
    lmdb_file = './results/bbb_train.lmdb'
    train_dataset = LMDBDataset(lmdb_file, conf=10, pad_len=150, mode="train")
    train_set = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=2,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=0,
                                                # collate_fn=train_dataset.collate_fn,
                                                worker_init_fn=train_dataset.worker_init_fn
                                                )
    for datas in train_set:
        print(datas["atoms"].shape)
        # print(datas["coordinate"].shape)
        # print(datas["label"])