from torch.utils.data import Dataset
import os
import numpy as np
import scipy.sparse as sps
import torch
# import dgl.backend as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import dgl
import numpy as np
import os
import random
from KPGT.src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from KPGT.src.data.collator_text import Collator_pretrain
from KPGT.src.model.light import LiGhTPredictor as LiGhT
from KPGT.src.trainer.scheduler import PolynomialDecayLR
from KPGT.src.trainer.pretrain_trainer import Trainer
# from KPGT.src.trainer.evaluator import Evaluator
# from KPGT.src.trainer.result_tracker import Result_Tracker
from KPGT.src.model_config import config_dict
import warnings
warnings.filterwarnings("ignore")
# local_rank = int(os.environ['LOCAL_RANK'])

path = "/home/jovyan/prompts_learning/bbb_cls_final_data_multi_class.csv"
import pandas as pd
data = pd.read_csv(path)

scaffold_training = data[data['scaffold_train_test_label'] == 'train']
scaffold_test = data[data['scaffold_train_test_label'] == 'test']
scaffold_training = scaffold_training.reset_index()
scaffold_test = scaffold_test.reset_index()

# print(len(scaffold_training['Smiles_unify']))
max_shape = 0 
num = 0
smi_list = []
for i in range(len(scaffold_training['Smiles_unify'])):
    shape = len(scaffold_training['Smiles_unify'][i])
    smi_list.append(scaffold_training['Smiles_unify'][i])
    num += 1
# print(f"number of the valuable data is {num}.")


# print(len(scaffold_test['Smiles_unify']))
max_shape = 0 
num = 0
test_smi_list = []
for i in range(len(scaffold_test['Smiles_unify'])):
    shape = len(scaffold_test['Smiles_unify'][i])
    test_smi_list.append(scaffold_test['Smiles_unify'][i])
    num += 1
# print(f"number of the valuable data is {num}.")


def list_min_max_norm(x_list, max_v, min_v):
    x_np = np.array(x_list)
    x_np = (x_np - min_v) / (max_v - min_v)
    
    x_np = np.where(x_np<0.5, x_np-0.3, x_np+0.3)
    x_np[x_np<0.] = 0.
    x_np[x_np>1.] = 1.
    
    x_list = x_np.tolist()
    return x_list


class MoleculeBBBDataset(Dataset):
    def __init__(self, mode, train_num=None):
        fp_path = f"/home/jovyan/prompts_learning/pretrain_data/bbb_{mode}_rdkfp1-7_512.npz"
        md_path = f"/home/jovyan/prompts_learning/pretrain_data/bbb_{mode}_molecular_descriptors.npz"
        # with open(smiles_path, 'r') as f:
        #     lines = f.readlines()
        #     self.smiles_list = [line.strip('\n') for line in lines]
        self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = np.where(np.isnan(mds), 0, mds)
        self.mds = torch.from_numpy(mds)
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]        
        
        if mode == "train":
            self.smiles_list = smi_list
        elif mode == "test":
            self.smiles_list = test_smi_list
            
        
        # get correponding label from csv
        self.value_list = []
        self.logd_list = []
        self.logp_list = []
        self.pka_list = []
        self.pkb_list = []
        self.logsol_list = []
        self.wlogsol_list = []
        self.max_len = 0
        if mode == "train":
            pd_data = scaffold_training
        else:
            pd_data = scaffold_test
        
        self.max_value = 0.
        self.min_value = 999.
        
        if train_num is None:
          for i in range(len(self.smiles_list)):
              idx = pd_data['Smiles_unify'][pd_data['Smiles_unify']==self.smiles_list[i]].index[0]
              self.value_list.append(pd_data['value'][idx])

              self.logd_list.append(pd_data['LogD_pred'][idx])
              self.logp_list.append(pd_data['LogP_pred'][idx])
              self.pka_list.append(pd_data['pKa_class_pred'][idx])
              self.pkb_list.append(pd_data['pKb_class_pred'][idx])
              # self.pka_list.append(pd_data['pKa_pred'][idx])
              # self.pkb_list.append(pd_data['pKb_pred'][idx])
              self.logsol_list.append(pd_data['LogSol_pred'][idx])
              self.wlogsol_list.append(pd_data['wLogSol_pred'][idx])
        else:
          pos_cnt = 0
          neg_cnt = 0
          for i in range(len(self.smiles_list)):
              idx = pd_data['Smiles_unify'][pd_data['Smiles_unify']==self.smiles_list[i]].index[0]
              if pd_data['value'][idx] == 1 and pos_cnt < train_num:
                pos_cnt += 1
              elif pd_data['value'][idx] == 0 and neg_cnt < train_num:
                neg_cnt += 1
              else:
                continue
              self.value_list.append(pd_data['value'][idx])

              self.logd_list.append(pd_data['LogD_pred'][idx])
              self.logp_list.append(pd_data['LogP_pred'][idx])
              self.pka_list.append(pd_data['pKa_class_pred'][idx])
              self.pkb_list.append(pd_data['pKb_class_pred'][idx])
              # self.pka_list.append(pd_data['pKa_pred'][idx])
              # self.pkb_list.append(pd_data['pKb_pred'][idx])
              self.logsol_list.append(pd_data['LogSol_pred'][idx])
              self.wlogsol_list.append(pd_data['wLogSol_pred'][idx])
          
            

        # self.logd_list = list_min_max_norm(self.logd_list, 8.7890625, -3.41796875)
        # self.logp_list = list_min_max_norm(self.logp_list, 12.796875, -5.53125)
        # self.logsol_list = list_min_max_norm(self.logsol_list, 3.017578125, -0.51806640625)
        # self.wlogsol_list = list_min_max_norm(self.wlogsol_list, 3.92578125, -12.3984375)
        
        # for mix training
        self.logd_list = list_min_max_norm(self.logd_list, 8.7890625, -3.41796875)
        self.logp_list = list_min_max_norm(self.logp_list, 12.796875, -5.53125)
        self.logsol_list = list_min_max_norm(self.logsol_list, 3.017578125, -0.54931640625)
        self.wlogsol_list = list_min_max_norm(self.wlogsol_list, 3.92578125, -12.3984375)


        num_p = 0
        num_n = 0
        for lab in self.value_list:
            if lab == 1: num_p += 1
            else: num_n += 1
        print(f"number of postive/negtive samples {num_p}/{num_n}.")
        

    def __len__(self):
        return len(self.value_list)
    
    def __getitem__(self, idx):
        label = self.value_list[idx]
        logd = self.logd_list[idx]
        logp = self.logp_list[idx]
        pka = self.pka_list[idx]
        pkb = self.pkb_list[idx]
        logsol = self.logsol_list[idx]
        wlogsol = self.wlogsol_list[idx]
        return self.smiles_list[idx], self.fps[idx], self.mds[idx],\
    label, logd, logp, pka, pkb, logsol, wlogsol
    
    

class MoleculeHERGDataset(Dataset):
    def __init__(self, mode, train_num=None):
        if mode == "train":
            path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_train_data.csv"
        # elif mode == "val":
        #     path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_valid_data.csv"
        # elif mode == "week1":
        #     path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week1_1201.csv"
        # elif mode == "week2":
        #     path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week2_1201.csv"
        # elif mode == "week3":
        #     path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week3_1201.csv"
        # elif mode == "week4":
        #     path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week4_1201.csv"
        data = pd.read_csv(path)
        self.smiles_list = []
        for i in range(len(data)):
            self.smiles_list.append(data['smiles'][i])
        
        fp_path = f"/home/jovyan/prompts_learning/pretrain_data/herg_cls_{mode}_rdkfp1-7_512.npz"
        md_path = f"/home/jovyan/prompts_learning/pretrain_data/herg_cls_{mode}_molecular_descriptors.npz"
        # with open(smiles_path, 'r') as f:
        #     lines = f.readlines()
        #     self.smiles_list = [line.strip('\n') for line in lines]
        self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = np.where(np.isnan(mds), 0, mds)
        self.mds = torch.from_numpy(mds)
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]        
            
        
        # get correponding label from csv
        self.value_list = []
        self.logd_list = []
        self.logp_list = []
        self.pka_list = []
        self.pkb_list = []
        self.logsol_list = []
        self.wlogsol_list = []
        self.max_len = 0

        pd_data = data
        self.max_value = 0.
        self.min_value = 999.
        
        if train_num is None:
          for i in range(len(self.smiles_list)):
            idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
            self.value_list.append(pd_data['class'][idx])
            
            self.logd_list.append(pd_data['LogD_pred'][idx])
            self.logp_list.append(pd_data['LogP_pred'][idx])
            self.pka_list.append(pd_data['pKa_class_pred'][idx])
            self.pkb_list.append(pd_data['pKb_class_pred'][idx])
            # self.pka_list.append(pd_data['pKa_pred'][idx])
            # self.pkb_list.append(pd_data['pKb_pred'][idx])
            self.logsol_list.append(pd_data['LogSol_pred'][idx])
            self.wlogsol_list.append(pd_data['wLogSol_pred'][idx])
        else:
          pos_cnt = 0
          neg_cnt = 0
          for i in range(len(self.smiles_list)):
              idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
              if pd_data['class'][idx] == 1 and pos_cnt < train_num:
                pos_cnt += 1
              elif pd_data['class'][idx] == 0 and neg_cnt < train_num:
                neg_cnt += 1
              else:
                continue
              
              self.value_list.append(pd_data['class'][idx])

              self.logd_list.append(pd_data['LogD_pred'][idx])
              self.logp_list.append(pd_data['LogP_pred'][idx])
              self.pka_list.append(pd_data['pKa_class_pred'][idx])
              self.pkb_list.append(pd_data['pKb_class_pred'][idx])
              # self.pka_list.append(pd_data['pKa_pred'][idx])
              # self.pkb_list.append(pd_data['pKb_pred'][idx])
              self.logsol_list.append(pd_data['LogSol_pred'][idx])
              self.wlogsol_list.append(pd_data['wLogSol_pred'][idx])
            

        # self.logd_list = list_min_max_norm(self.logd_list, 5.75390625, -0.93310546875)
        # self.logp_list = list_min_max_norm(self.logp_list, 7.89453125, -1.91796875)
        # self.logsol_list = list_min_max_norm(self.logsol_list, 2.880859375, -0.54931640625)
        # self.wlogsol_list = list_min_max_norm(self.wlogsol_list, 3.5625, -9.6171875)
        
        # for mix training
        self.logd_list = list_min_max_norm(self.logd_list, 8.7890625, -3.41796875)
        self.logp_list = list_min_max_norm(self.logp_list, 12.796875, -5.53125)
        self.logsol_list = list_min_max_norm(self.logsol_list, 3.017578125, -0.54931640625)
        self.wlogsol_list = list_min_max_norm(self.wlogsol_list, 3.92578125, -12.3984375)


        num_p = 0
        num_n = 0
        for lab in self.value_list:
            if lab == 1: num_p += 1
            else: num_n += 1
        print(f"number of postive/negtive samples {num_p}/{num_n}.")
        

    def __len__(self):
        return len(self.value_list)
    
    def __getitem__(self, idx):
        label = self.value_list[idx]
        logd = self.logd_list[idx]
        logp = self.logp_list[idx]
        pka = self.pka_list[idx]
        pkb = self.pkb_list[idx]
        logsol = self.logsol_list[idx]
        wlogsol = self.wlogsol_list[idx]
        return self.smiles_list[idx], self.fps[idx], self.mds[idx],\
    label, logd, logp, pka, pkb, logsol, wlogsol
      

class MoleculeHERGFewShotDataset(Dataset):
    def __init__(self, mode):
        if mode == "train":
            path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_train_data.csv"
        data = pd.read_csv(path)
        self.smiles_list = []
        for i in range(len(data)):
            self.smiles_list.append(data['smiles'][i])
        
        fp_path = f"/home/jovyan/prompts_learning/pretrain_data/herg_cls_{mode}_rdkfp1-7_512.npz"
        md_path = f"/home/jovyan/prompts_learning/pretrain_data/herg_cls_{mode}_molecular_descriptors.npz"
        # with open(smiles_path, 'r') as f:
        #     lines = f.readlines()
        #     self.smiles_list = [line.strip('\n') for line in lines]
        self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = np.where(np.isnan(mds), 0, mds)
        self.mds = torch.from_numpy(mds)
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]        
            
        
        # get correponding label from csv
        self.value_list = []
        self.logd_list = []
        self.logp_list = []
        self.pka_list = []
        self.pkb_list = []
        self.logsol_list = []
        self.wlogsol_list = []
        self.max_len = 0

        pd_data = data
        self.max_value = 0.
        self.min_value = 999.
        
        
        
        for i in range(len(self.smiles_list)):
            idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
            self.value_list.append(pd_data['class'][idx])
            
            self.logd_list.append(pd_data['LogD_pred'][idx])
            self.logp_list.append(pd_data['LogP_pred'][idx])
            self.pka_list.append(pd_data['pKa_class_pred'][idx])
            self.pkb_list.append(pd_data['pKb_class_pred'][idx])
            # self.pka_list.append(pd_data['pKa_pred'][idx])
            # self.pkb_list.append(pd_data['pKb_pred'][idx])
            self.logsol_list.append(pd_data['LogSol_pred'][idx])
            self.wlogsol_list.append(pd_data['wLogSol_pred'][idx])
            

        self.logd_list = list_min_max_norm(self.logd_list, 5.75390625, -0.93310546875)
        self.logp_list = list_min_max_norm(self.logp_list, 7.89453125, -1.91796875)
        self.logsol_list = list_min_max_norm(self.logsol_list, 2.880859375, -0.54931640625)
        self.wlogsol_list = list_min_max_norm(self.wlogsol_list, 3.5625, -9.6171875)


        num_p = 0
        num_n = 0
        for lab in self.value_list:
            if lab == 1: num_p += 1
            else: num_n += 1
        print(f"number of postive/negtive samples {num_p}/{num_n}.")
        

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        label = self.value_list[idx]
        logd = self.logd_list[idx]
        logp = self.logp_list[idx]
        pka = self.pka_list[idx]
        pkb = self.pkb_list[idx]
        logsol = self.logsol_list[idx]
        wlogsol = self.wlogsol_list[idx]
        return self.smiles_list[idx], self.fps[idx], self.mds[idx],\
    label, logd, logp, pka, pkb, logsol, wlogsol
    
    

class MoleculeHERGTestDataset(Dataset):
    def __init__(self, mode):
        if mode == "val":
            path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_valid_data.csv"
        elif mode == "week1":
            path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week1_1201.csv"
        elif mode == "week2":
            path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week2_1201.csv"
        elif mode == "week3":
            path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week3_1201.csv"
        elif mode == "week4":
            path = "/home/jovyan/prompts_learning/herg_dataset/hERGDB_cls_week4_1201.csv"
        data = pd.read_csv(path)
        self.smiles_list = []
        for i in range(len(data)):
            self.smiles_list.append(data['smiles'][i])
        
        fp_path = f"/home/jovyan/prompts_learning/pretrain_data/herg_cls_{mode}_rdkfp1-7_512.npz"
        md_path = f"/home/jovyan/prompts_learning/pretrain_data/herg_cls_{mode}_molecular_descriptors.npz"
        # with open(smiles_path, 'r') as f:
        #     lines = f.readlines()
        #     self.smiles_list = [line.strip('\n') for line in lines]
        self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = np.where(np.isnan(mds), 0, mds)
        self.mds = torch.from_numpy(mds)
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]        
            
        
        # get correponding label from csv
        self.value_list = []
        self.max_len = 0

        pd_data = data
        self.max_value = 0.
        self.min_value = 999.
        for i in range(len(self.smiles_list)):
            idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
            self.value_list.append(pd_data['class'][idx])


        num_p = 0
        num_n = 0
        for lab in self.value_list:
            if lab == 1: num_p += 1
            else: num_n += 1
        print(f"number of postive/negtive samples {num_p}/{num_n}.")
        

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
       
        return self.smiles_list[idx], self.fps[idx], self.mds[idx], self.value_list[idx]

    
if __name__ == "__main__":
    config = config_dict['base']
    print(config)
    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(local_rank)
    # torch.distributed.init_process_group(backend='nccl')
    # device = torch.device('cuda', local_rank)
    # set_random_seed(args.seed)
    # print(local_rank)
    
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    collator = Collator_pretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
    train_dataset = MoleculeTextDataset(smi_list, text_list)
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size']// 1, num_workers=16, drop_last=True, collate_fn=collator)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=16, drop_last=True, collate_fn=collator)
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=config['input_drop'],
        attn_drop=config['attn_drop'],
        feat_drop=config['feat_drop'],
        n_node_types=vocab.vocab_size
    ).to("cuda")
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("/home/jovyan/prompts_learning/KPGT/src/models/base.pth").items()})
    print("Pre-trained weights of KPGT were loaded successfully!")
    
    device = "cuda"
    for b_id, batched_data in enumerate(train_loader):
        (smiles, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds, text) = batched_data
        batched_graph = batched_graph.to(device)
        fps = fps.to(device)
        mds = mds.to(device)

        mol_fps_feat = model.generate_fps(batched_graph, fps, mds)
        print(mol_fps_feat.shape)
        print(text.shape)
        break
        
    
    print("okk")