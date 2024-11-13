import lmdb
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import pickle

import pandas as pd
# data = pd.read_csv("/home/jovyan/PharmaBench/data/final_datasets/logd_reg_final_data.csv")
data = pd.read_csv("/home/jovyan/PharmaBench/data/final_datasets/bbb_cls_final_data.csv")
scaffold_training = data[data['scaffold_train_test_label'] == 'train']
scaffold_test = data[data['scaffold_train_test_label'] == 'test']
scaffold_training = scaffold_training.reset_index()
scaffold_test = scaffold_test.reset_index()


class LMDBDataset(Dataset):
    def __init__(self, db_path, conf, pad_len=200, mode="train"):
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
        if mode == "train":
            pd_data = scaffold_training
        else:
            pd_data = scaffold_test
        
        self.max_value = 0.
        self.min_value = 999.
        for i in range(len(self._keys)):
            datapoint_pickled = env.begin().get(self._keys[i])
            data = pickle.loads(datapoint_pickled)
            current_len = len(data['atoms'])
            if self.max_len < current_len:
                self.max_len = current_len
            idx = pd_data['Smiles_unify'][pd_data['Smiles_unify']==data['smi']].index[0]
            self.value_list.append(pd_data['value'][idx])
            
            # for normalization of reprogression task
            if self.max_value < pd_data['value'][idx]:
                self.max_value = pd_data['value'][idx]
            if self.min_value > pd_data['value'][idx]:
                self.min_value = pd_data['value'][idx]
        
        # for classfication task, to balance number of postive/negtive samples
        num_p = 0
        num_n = 0
        for lab in self.value_list:
            if lab == 1: num_p += 1
            else: num_n += 1
        print(f"number of positive/negative samples {num_p}/{num_n}.")
            
        
        # get word embedding index (atoms) ps:only use one time and fix the dict result 
        # self.atom_dict = {}
        # idx = 0 
        # for i in range(len(self._keys)):
        #     datapoint_pickled = env.begin().get(self._keys[i])
        #     data = pickle.loads(datapoint_pickled)
        #     for a in data['atoms']:
        #         if a not in self.atom_dict:
        #             self.atom_dict.update({a:idx})
        #             idx += 1
        self.atom_dict = {'Br': 0, 'C': 1, 'H': 2, 'N': 3, 'O': 4, 'F': 5, 'Cl': 6, 'S': 7, 'P': 8, 'I': 9, 'B': 10, 'Se': 11, 'Ar': 12, 'Kr': 13, 'Li': 14, 'Ne': 15, 'Xe': 16, 'Si': 17}
        print(self.atom_dict)
        
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
        if np.random.rand() < 0.5 and self.mode == "train" and cur_len > 100:
            num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num = random.choice(num_list)
            emb_idx = emb_idx[-num:]
            coordinates = coordinates[:, -num:,:]
            spd = spd[-num:, -num:]
            edge = edge[-num:, -num:]
            cur_len = len(emb_idx)
        
        # padding
        if cur_len < self.pad_len:
            new_emb = torch.full(size=(self.pad_len,), fill_value=self.pad_len-1, dtype=torch.long)
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
        if self.mode == "train": # for random augmentation
            weight_list = [0.1, 0.2, 0.3, 0.5, 0.7, 0.01, 0.001]
            scale = random.choice(weight_list) 
            noise = scale * torch.randn_like(new_cor)
            if np.random.rand() < 0.5:  # for add noise locally
                mask = torch.randint_like(noise, 0, 2, dtype=torch.float32)
                noise = noise * mask
            new_cor = new_cor + noise
            
            # new_cor = new_cor + 1. * torch.rand_like(new_cor)
        # new_cor = (self.min_max_norm(new_cor) - 0.5) / 0.5
            
        # to compute the pair relative distance
        atom_expanded = new_cor.unsqueeze(2)  # shape (conf, pad_len, 1, 3)
        coor_expanded = new_cor.unsqueeze(1)   # shape (conf, 1, pad_len, 3)
        # distance = atom_expanded - coor_expanded
        # distance = distance.permute(1, 0, 2, 3).reshape(-1, conf*pad_len*3)   # xyz 
        distance = torch.sqrt((atom_expanded - coor_expanded).pow(2).sum(dim=-1))   # x+y+z
        distance = distance.permute(1, 0, 2).reshape(-1, self.conf*self.pad_len)
        distance = (self.min_max_norm(distance) - 0.5) / 0.5
        
        label = torch.tensor(self.value_list[idx], dtype=torch.float32)
        label = (label - self.min_value) / (self.max_value - self.min_value)
        
        new_cor = (self.min_max_norm(new_cor) - 0.5) / 0.5
        new_cor = new_cor.permute(1, 0, 2).reshape(-1, self.conf*3)
        # new_spd = (self.min_max_norm(new_spd) - 0.5) / 0.5
        # new_edge = (self.min_max_norm(new_edge) - 0.5) / 0.5
       
        
        return {"atoms": new_emb, "coordinate": new_cor, "distance": distance, "SPD": new_spd, "edge": new_edge, "label": label}
        

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

        
if __name__ == "__main__":
    lmdb_file = './results/bbb_train.lmdb'
    train_dataset = LMDBDataset(lmdb_file, conf=11, pad_len=200, mode="train")
    train_set = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=2,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=0,
                                                # collate_fn=train_dataset.collate_fn,
                                                worker_init_fn=train_dataset.worker_init_fn
                                                )
    for datas in train_set:
        # print(datas["atoms"].shape)
        # print(datas["coordinate"].shape)
        # print(datas["label"])
        exit(0)