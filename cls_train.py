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
# from lmdb_datapipeline import LMDBDataset  
from cls_lmdb_datapipeline import LMDBDataset 
# lmdb_file = './results/logd_train.lmdb'
lmdb_file = './results/bbb_train.lmdb'
train_dataset = LMDBDataset(lmdb_file, conf=conf, pad_len=pad_len, mode="train")
train_set = DataLoader(train_dataset,
                    batch_size=16,  # 16 best
                    drop_last=True,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=4,
                    # collate_fn=train_dataset.collate_fn,
                    worker_init_fn=train_dataset.worker_init_fn
                    )

# lmdb_file = './results/logd_test.lmdb'
lmdb_file = './results/bbb_test.lmdb'
val_dataset = LMDBDataset(lmdb_file, conf=conf, pad_len=pad_len, mode="test")
val_set = DataLoader(val_dataset,
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

from copy import deepcopy
ema = deepcopy(model).to(dev)  # Create an EMA of the model for use after training
update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
requires_grad(ema, False)
ema.eval()

# best
lr = 5e-5
wd = 0.

print(f'Set of Optimizer: lr:{lr}, weight_decay:{wd}')
model_params = [# {'params': model.parameters(), 'lr': lr},
                {'params': model.atom_encoder.parameters(), 'lr': lr},
                {'params': model.coor_encoder.parameters(), 'lr': lr},
                {'params': model.fusion_blocks.parameters(), 'lr': lr},
                {'params': model.head.parameters(), 'lr': 1e-3},
                {'params': model.prompts_processor.parameters(), 'lr': lr},
                {'params': model.ppim.parameters(), 'lr': lr},
                # {'params': model.qformer.parameters(), 'lr':1e-5}
               ]

optims = 'adan'
# optims = "sgd"
if optims == 'adan':
    from adan import Adan
    optimizer = Adan(model_params, betas=(0.98, 0.92, 0.99),weight_decay=wd, max_grad_norm=0.)
elif optims == 'sgd':
    optimizer = optim.SGD(model_params, momentum=0.9, weight_decay=wd)
elif optims == 'adamw':
    optimizer = optim.AdamW(model_params, betas=(0.9, 0.999), weight_decay=wd)
elif optims == 'adam':
    optimizer = optim.Adam(model_params, betas=(0.9, 0.999), weight_decay=wd)
print('Current Optimizer is', optims)


###################################################
########   build learning rate scheduler   ######## 
###################################################
scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1./3., total_iters=5) # best
scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=5)
scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[5, 10])
cur_lr = scheduler.get_last_lr() 
print(f"Current learning rate is {cur_lr}.")


##########################################
########   build loss criterion   ######## 
##########################################
cri_mse = nn.MSELoss()
cri_mae = nn.L1Loss()
weight = torch.tensor([6641./2042., 6641./4599.], dtype=torch.float32, device=dev)
cri_ce = nn.CrossEntropyLoss(weight)


##########################################
######## start training our model ######## 
##########################################
gc.collect()
torch.cuda.empty_cache()
Epoch = 1000
best_val = 0.91
print("Let's start training!")

for e in range(0, Epoch):
    start = time.time()
    model.train()
    for step_id, datas in enumerate(train_set):
            break
            # print(datas["atoms"].shape)
            # print(datas["coordinate"].shape)
            # print(datas["label"])
            atoms = datas["atoms"].to(dev, non_blocking=True).long()
            # coord = datas["coordinate"].to(dev, non_blocking=True).float()
            pair = datas["distance"].to(dev, non_blocking=True).float()
            spd = datas["SPD"].to(dev, non_blocking=True).float()
            edge = datas["edge"].to(dev, non_blocking=True).float()
            label = datas["label"].to(dev, non_blocking=True).float()
            # label = torch.tanh(label)
            
            pred = model(atoms, pair, spd, edge)
            # pred = torch.tanh(pred)

            # loss = cri_mae(pred, label.unsqueeze(-1))
            
            # weighted BCE loss
            # loss = F.binary_cross_entropy_with_logits(pred, label.unsqueeze(-1))
            loss = F.binary_cross_entropy_with_logits(pred, label.unsqueeze(-1), reduction='none')
            for i, lab in enumerate(label):
                if lab == 0:
                    # loss[i] = loss[i] * (6641./2042.)
                    loss[i] = loss[i] * (4599./6641)  # best
                else:
                    # loss[i] = loss[i] * (6641./4599.)
                    loss[i] = loss[i] * (2042./6641)
            loss = loss.mean()
            
            # CE loss
            # loss = cri_ce(pred, label.long())
            

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            update_ema(ema, model, 0.997) # 0.997 best

            if not (step_id+1) % 40:
                print(f"epoch: {e+1} / {Epoch},step {step_id} / {len(train_set)}, loss: {loss.detach().cpu().numpy():.4f}")
                
    
    ##########################################
    ####### start evaluating our model #######
    ##########################################
    model.eval()
    print("evaluating...")
    with torch.no_grad():
        all_pred = None 
        all_lab = None
        for step_id, datas in enumerate(val_set):
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
    print(f"epoch: {e+1} / {Epoch}, test AUC: {auc:.5f}")
    if auc > best_val:
        best_val = auc
        # torch.save(model.state_dict(), f'./trained_weight/cliP_Epoch{e+1}_val_auc_{best_val:.5f}.pth') 
        torch.save(ema.state_dict(), f'./trained_weight/cliP_Epoch{e+1}_val_auc_{best_val:.5f}.pth') 


    end = time.time()
    print(f"epoch: {e+1} end ; cost time: {(end - start)/60.:.4f} min")
    gc.collect()
    torch.cuda.empty_cache()
    scheduler.step()
    cur_lr = scheduler.get_last_lr() 
    print(f"Current learning rate is {cur_lr}.")