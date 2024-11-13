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

from herg_cls_datapipeline import HERG_LMDBDataset 
test_dataset = HERG_LMDBDataset(conf=conf, pad_len=pad_len, mode="week1")
test_set = DataLoader(test_dataset,
                    batch_size=32,
                    drop_last=False,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=4,
                    # collate_fn=train_dataset.collate_fn,
                    worker_init_fn=val_dataset.worker_init_fn
                    )


##########################################
######  build model and optimizer  ####### 
##########################################
dev = "cuda" if torch.cuda.is_available() else "cpu"
from CLIP import clip
clip_model, preprocess = clip.load(name="ViT-B/16", device="cpu", download_root="/home/jovyan/clip_download_root")
from model_zoo import CLIP_Protein
model = CLIP_Protein(clip_model, conf, pad_len=pad_len).to(dev)

path = "/home/jovyan/prompts_learning/trained_weight/A_10_7_cliP_Epoch13_val_auc_0.90789.pth"
sd = torch.load(path)
model.load_state_dict(sd)
print("pre-trained weights loaded...")

                
    
##########################################
####### start evaluating our model #######
##########################################
model.eval()
print("evaluating...")
with torch.no_grad():
    all_pred = None 
    all_lab = None
    for step_id, datas in enumerate(test_set):
            atoms = datas["atoms"].to(dev, non_blocking=True).long()
            # coord = datas["coordinate"].to(dev, non_blocking=True).float()
            pair = datas["distance"].to(dev, non_blocking=True).float()
            spd = datas["SPD"].to(dev, non_blocking=True).float()
            edge = datas["edge"].to(dev, non_blocking=True).float()
            label = datas["label"].to(dev, non_blocking=True).float()
            # label = torch.tanh(label)
            # pred = model(atoms, pair, spd, edge)
            pred = ema(atoms, pair, spd, edge)

            pred = torch.sigmoid(pred)
            # pred = torch.softmax(pred, dim=-1)[:, 1]
            # total_loss += cri_mae(pred, label.unsqueeze(-1))
            all_pred = pred if all_pred is None else torch.cat([all_pred, pred], dim=0)
            all_lab = label if all_lab is None else torch.cat([all_lab, label], dim=0)
auc = compute_AUC(all_lab.cpu().detach(), all_pred.cpu().detach())
print(f"test AUC: {auc:.5f}")


