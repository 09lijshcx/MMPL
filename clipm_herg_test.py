import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import gc
from sklearn import metrics
from torch.cuda.amp import autocast, GradScaler
# torch.autograd.set_detect_anomaly(True)
CUDA_LAUNCH_BLOCKING=1


# fixed random seed for reproduction
seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
print('Random seed :', seed)

from collections import OrderedDict
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

from sklearn import metrics
def compute_AUC(y, pred, n_class=1):
    # compute one score
    if n_class == 1:
        auc = metrics.roc_auc_score(y, pred)

    # compute two-class
    elif n_class == 2:
        # pos = pred[:, 1]
        auc = metrics.roc_auc_score(y, pred)
    return auc

pad_len = 150 # best
conf = 10 # best# 

##########################################
#########  construct dataloader  ######### 
##########################################
from tune_cls_pipeline import MoleculeHERGTestDataset
from KPGT.src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from KPGT.src.data.collator_tune import Collator_pretrain, Collator_tune
from KPGT.src.model_config import config_dict
config = config_dict['base']


test_dataset1 = MoleculeHERGTestDataset("week1")
test_dataset2 = MoleculeHERGTestDataset("week2")
test_dataset3 = MoleculeHERGTestDataset("week3")
test_dataset4 = MoleculeHERGTestDataset("week4")
vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
collator = Collator_tune(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
test_set1 = DataLoader(test_dataset1, batch_size=16, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)
test_set2 = DataLoader(test_dataset2, batch_size=16, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)
test_set3 = DataLoader(test_dataset3, batch_size=16, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)
test_set4 = DataLoader(test_dataset4, batch_size=16, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)


##########################################
######  build model and optimizer  ####### 
##########################################
dev = "cuda" if torch.cuda.is_available() else "cpu"
# from CLIP import clip
# clip_model, preprocess = clip.load(name="ViT-B/16", device="cpu", download_root="/home/jovyan/clip_download_root")

from KPGT.src.model.light import LiGhTPredictor as LiGhT
kpgt = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=test_dataset1.d_fps,
        d_md_feats=test_dataset1.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        # input_drop=config['input_drop'], # 0.0
        # attn_drop=config['attn_drop'],   # 0.1
        # feat_drop=config['feat_drop'],   # 0.1
        input_drop=0.,
        attn_drop=0.,
        feat_drop=0.,
        n_node_types=vocab.vocab_size
    )# .to("cuda")
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
kpgt.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("/home/jovyan/prompts_learning/KPGT/src/models/base.pth").items()})
print("Pre-trained weights of KPGT were loaded successfully!")

# from model_zoo import KPGT
# model = KPGT(kpgt).to(dev)

# from model_zoo import CLIPM
# model = CLIPM(kpgt).to(dev)

from model_zoo import MMPL
model = MMPL(kpgt).to(dev)

# path = "/home/jovyan/prompts_learning/trained_weight/MMPL_MTattn_fewshot_tune_herg_Epoch6_val_auc_0.84329.pth" # CLIPM
# path = "/home/jovyan/prompts_learning/trained_weight/KPGT_tune_herg_Epoch4_val_auc_0.88595.pth" # KPGT
path = "/home/jovyan/prompts_learning/trained_weight/MMPL_spcTeacher_tune_herg_Epoch3_val_auc_0.89004.pth"
sd = torch.load(path)
model.load_state_dict(sd)
print(f"pre-trained weights loaded from {path}")

                
    
##########################################
####### start evaluating our model #######
##########################################
model.eval()
print("evaluating...")
with torch.no_grad():
    all_pred = None 
    all_lab = None
    for step_id, batched_data in enumerate(test_set1):
            (_, batched_graph, fps, mds, label) = batched_data
            batched_graph = batched_graph.to(dev)
            fps = fps.to(dev)
            mds = mds.to(dev)
            
            pred = model([batched_graph, fps, mds])
            # pred = ema([batched_graph, fps, mds])

            pred = torch.sigmoid(pred)

            all_pred = pred if all_pred is None else torch.cat([all_pred, pred], dim=0)
            all_lab = label if all_lab is None else torch.cat([all_lab, label], dim=0)
auc = compute_AUC(all_lab.cpu().detach(), all_pred.cpu().detach())
print(f"test AUC: {auc:.5f}")

with torch.no_grad():
    all_pred = None 
    all_lab = None
    for step_id, batched_data in enumerate(test_set2):
            (_, batched_graph, fps, mds, label) = batched_data
            batched_graph = batched_graph.to(dev)
            fps = fps.to(dev)
            mds = mds.to(dev)
            
            pred = model([batched_graph, fps, mds])
            # pred = ema([batched_graph, fps, mds])

            pred = torch.sigmoid(pred)

            all_pred = pred if all_pred is None else torch.cat([all_pred, pred], dim=0)
            all_lab = label if all_lab is None else torch.cat([all_lab, label], dim=0)
auc = compute_AUC(all_lab.cpu().detach(), all_pred.cpu().detach())
print(f"test AUC: {auc:.5f}")

with torch.no_grad():
    all_pred = None 
    all_lab = None
    for step_id, batched_data in enumerate(test_set3):
            (_, batched_graph, fps, mds, label) = batched_data
            batched_graph = batched_graph.to(dev)
            fps = fps.to(dev)
            mds = mds.to(dev)
            
            pred = model([batched_graph, fps, mds])
            # pred = ema([batched_graph, fps, mds])

            pred = torch.sigmoid(pred)

            all_pred = pred if all_pred is None else torch.cat([all_pred, pred], dim=0)
            all_lab = label if all_lab is None else torch.cat([all_lab, label], dim=0)
auc = compute_AUC(all_lab.cpu().detach(), all_pred.cpu().detach())
print(f"test AUC: {auc:.5f}")

with torch.no_grad():
    all_pred = None 
    all_lab = None
    for step_id, batched_data in enumerate(test_set4):
            (_, batched_graph, fps, mds, label) = batched_data
            batched_graph = batched_graph.to(dev)
            fps = fps.to(dev)
            mds = mds.to(dev)
            
            pred = model([batched_graph, fps, mds])
            # pred = ema([batched_graph, fps, mds])

            pred = torch.sigmoid(pred)

            all_pred = pred if all_pred is None else torch.cat([all_pred, pred], dim=0)
            all_lab = label if all_lab is None else torch.cat([all_lab, label], dim=0)
auc = compute_AUC(all_lab.cpu().detach(), all_pred.cpu().detach())
print(f"test AUC: {auc:.5f}")


