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


def print_loss(loss, loss_name):
    print(f"{loss_name}: {loss.detach().cpu().numpy():.4f}; ", end='', flush=True)
    # print('\r', end='', flush=True)

    
pad_len = 150 # best
conf = 10 # best# 

##########################################
#########  construct dataloader  ######### 
##########################################
train_bz = 64
 
from pretrain_datapipeline import Pretrain_LMDBDataset
train_dataset = Pretrain_LMDBDataset(conf=conf, pad_len=pad_len, mode="logd")
train_set = DataLoader(train_dataset,
                    batch_size=train_bz,  # 16 best
                    drop_last=True,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=4,
                    # collate_fn=train_dataset.collate_fn,
                    worker_init_fn=train_dataset.worker_init_fn
                    )

train_dataset3 = Pretrain_LMDBDataset(conf=conf, pad_len=pad_len, mode="herg")
train_set3 = DataLoader(train_dataset3,
                    batch_size=train_bz,  # 16 best
                    drop_last=True,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=4,
                    # collate_fn=train_dataset.collate_fn,
                    worker_init_fn=train_dataset3.worker_init_fn
                    )

train_dataset2 = Pretrain_LMDBDataset(conf=conf, pad_len=pad_len, mode="bbb")
train_set2 = DataLoader(train_dataset2,
                    batch_size=train_bz,  # 16 best
                    drop_last=True,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=4,
                    # collate_fn=train_dataset.collate_fn,
                    worker_init_fn=train_dataset2.worker_init_fn
                    )

val_dataset = Pretrain_LMDBDataset(conf=conf, pad_len=pad_len, mode="bbb_test")
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
from model_zoo import CLIPM_Stage1
model = CLIPM_Stage1(clip_model, conf, pad_len=pad_len).to(dev)


from copy import deepcopy
ema = deepcopy(model).to(dev)  # Create an EMA of the model for use after training
update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
requires_grad(ema, False)
ema.eval()

# best
lr = 5e-5
wd = 0.

print(f'Set of Optimizer: lr:{lr}, weight_decay:{wd}')
model_params = [
    {'params': model.parameters(), 'lr': lr},
                # {'params': model.atom_encoder.parameters(), 'lr': lr},
                # {'params': model.coor_encoder.parameters(), 'lr': lr},
                # {'params': model.fusion_blocks.parameters(), 'lr': lr},
                # {'params': model.pretrain_head1.parameters(), 'lr': lr},
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
# scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1./3., total_iters=5) # best
# scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=95)
# scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[5, 100])
# cur_lr = scheduler.get_last_lr() 
# print(f"Current learning rate is {cur_lr}.")



##########################################
######## start training our model ######## 
##########################################
gc.collect()
torch.cuda.empty_cache()
Epoch = 100
print("Let's start training!")
all_len = len(train_set)+len(train_set2)+len(train_set3)
for e in range(0, Epoch):
    start = time.time()
    model.train()
    for step_id, datas in enumerate(train_set):
            atoms = datas["atoms"].to(dev, non_blocking=True).long()
            coord = datas["coordinate"].to(dev, non_blocking=True).float()
            pair = datas["distance"].to(dev, non_blocking=True).float()
            spd = datas["SPD"].to(dev, non_blocking=True).float()
            edge = datas["edge"].to(dev, non_blocking=True).float()
            masked_label = datas["masked_label"].to(dev, non_blocking=True).long()
            ids_masked = datas["ids_masked"].to(dev, non_blocking=True).long()
            n_gt = datas["noise_gt"].to(dev, non_blocking=True).float()
            
            pre_atom, pre_dist = model(atoms, pair, spd, edge, ids_masked)
            loss_atom = F.cross_entropy(pre_atom.reshape(-1, 21), masked_label.reshape(-1))
            loss_dist = F.mse_loss(pre_dist.permute(1, 0, 2), coord)
            loss = loss_atom + loss_dist

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            update_ema(ema, model, 0.997) # 0.997 best

            if not (step_id+1) % 40:
                # print(f"epoch: {e+1} / {Epoch},step {step_id} / {len(train_set)}, loss: {loss.detach().cpu().numpy():.4f}")
                # print(f"epoch: {e+1} / {Epoch},step {step_id} / {len(train_set)+len(train_set2)}, loss: {loss.detach().cpu().numpy():.4f}")
                print(f"epoch: {e+1} / {Epoch},step {step_id} / {all_len}, loss: {loss.detach().cpu().numpy():.4f}")
                print_loss(loss_atom, "Atom")
                print_loss(loss_dist, "Dist")
                print()
    
    for step_id, datas in enumerate(train_set2):
            atoms = datas["atoms"].to(dev, non_blocking=True).long()
            coord = datas["coordinate"].to(dev, non_blocking=True).float()
            pair = datas["distance"].to(dev, non_blocking=True).float()
            spd = datas["SPD"].to(dev, non_blocking=True).float()
            edge = datas["edge"].to(dev, non_blocking=True).float()
            masked_label = datas["masked_label"].to(dev, non_blocking=True).long()
            ids_masked = datas["ids_masked"].to(dev, non_blocking=True).long()
            n_gt = datas["noise_gt"].to(dev, non_blocking=True).float()
            
            pre_atom, pre_dist = model(atoms, pair, spd, edge, ids_masked)
            loss_atom = F.cross_entropy(pre_atom.reshape(-1, 21), masked_label.reshape(-1))
            loss_dist = F.mse_loss(pre_dist.permute(1, 0, 2), coord)
            loss = loss_atom + loss_dist

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            update_ema(ema, model, 0.997) # 0.997 best

            if not (step_id+1) % 40:
                print(f"epoch: {e+1} / {Epoch},step {step_id+len(train_set)} / {all_len}, loss: {loss.detach().cpu().numpy():.4f}")
                print_loss(loss_atom, "Atom")
                print_loss(loss_dist, "Dist")
                print()
                
    for step_id, datas in enumerate(train_set3):
            atoms = datas["atoms"].to(dev, non_blocking=True).long()
            coord = datas["coordinate"].to(dev, non_blocking=True).float()
            pair = datas["distance"].to(dev, non_blocking=True).float()
            spd = datas["SPD"].to(dev, non_blocking=True).float()
            edge = datas["edge"].to(dev, non_blocking=True).float()
            masked_label = datas["masked_label"].to(dev, non_blocking=True).long()
            ids_masked = datas["ids_masked"].to(dev, non_blocking=True).long()
            n_gt = datas["noise_gt"].to(dev, non_blocking=True).float()
            
            pre_atom, pre_dist = model(atoms, pair, spd, edge, ids_masked)
            loss_atom = F.cross_entropy(pre_atom.reshape(-1, 21), masked_label.reshape(-1))
            loss_dist = F.mse_loss(pre_dist.permute(1, 0, 2), coord)
            loss = loss_atom + loss_dist

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            update_ema(ema, model, 0.997) # 0.997 best

            if not (step_id+1) % 40:
                print(f"epoch: {e+1} / {Epoch},step {step_id+len(train_set)+len(train_set2)} / {all_len}, loss: {loss.detach().cpu().numpy():.4f}")
                print_loss(loss_atom, "Atom")
                print_loss(loss_dist, "Dist")
                print()
                
    
    ##########################################
    ####### start evaluating our model #######
    ##########################################
    model.eval()
    print("evaluating...")
    with torch.no_grad():
        all_atom_loss = None 
        all_dist_loss = None
        for step_id, datas in enumerate(val_set):
                atoms = datas["atoms"].to(dev, non_blocking=True).long()
                coord = datas["coordinate"].to(dev, non_blocking=True).float()
                pair = datas["distance"].to(dev, non_blocking=True).float()
                spd = datas["SPD"].to(dev, non_blocking=True).float()
                edge = datas["edge"].to(dev, non_blocking=True).float()
                masked_label = datas["masked_label"].to(dev, non_blocking=True).long()
                ids_masked = datas["ids_masked"].to(dev, non_blocking=True).long()
                n_gt = datas["noise_gt"].to(dev, non_blocking=True).float()
                
                pre_atom, pre_dist = model(atoms, pair, spd, edge, ids_masked)
                loss_atom = F.cross_entropy(pre_atom.reshape(-1, 21), masked_label.reshape(-1)).unsqueeze(0)
                loss_dist = F.mse_loss(pre_dist.permute(1, 0, 2), coord).unsqueeze(0)
                
                all_atom_loss = loss_atom if all_atom_loss is None else torch.cat([all_atom_loss, loss_atom], dim=0)
                all_dist_loss = loss_dist if all_dist_loss is None else torch.cat([all_dist_loss, loss_dist], dim=0)
                
    all_atom_loss = all_atom_loss.mean()
    all_dist_loss = all_dist_loss.mean()
    print(f"epoch: {e+1} / {Epoch}, test atom loss: {all_atom_loss:.5f}; test dist loss: {all_dist_loss:.5f}")
    if not (e+1) % 1:
        check_point = {
            'atom_encoder': model.atom_encoder.state_dict(),
            'coor_encoder': model.coor_encoder.state_dict(),
            'fusion_blocks': model.fusion_blocks.state_dict(),
            'pretrain_head1': model.pretrain_head1.state_dict(),
            'pretrain_head2': model.pretrain_head2.state_dict(),
        }
        
        torch.save(check_point, f'./trained_weight/cliPM_pre_stage1_Epoch{e+1}.pth') 


    end = time.time()
    print(f"epoch: {e+1} end ; cost time: {(end - start)/60.:.4f} min")
    gc.collect()
    torch.cuda.empty_cache()
    # scheduler.step()
    # cur_lr = scheduler.get_last_lr() 
    # print(f"Current learning rate is {cur_lr}.")