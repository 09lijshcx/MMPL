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

import os
local_rank = int(os.environ['LOCAL_RANK'])

# fixed random seed for reproduction
seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

torch.backends.cudnn.benchmark = True
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl')
device = torch.device('cuda', local_rank)
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


##########################################
#########  construct dataloader  ######### 
##########################################
 
from text_pretrain_datapipeline import MoleculeTextDataset
from KPGT.src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from KPGT.src.data.collator_text import Collator_pretrain
from KPGT.src.model_config import config_dict
config = config_dict['base']

vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
collator = Collator_pretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
train_dataset = MoleculeTextDataset()
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, drop_last=True, collate_fn=collator)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
from torch.utils.data.distributed import DistributedSampler
train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset), batch_size=16 // 1, num_workers=8, worker_init_fn=seed_worker, drop_last=True, collate_fn=collator)


##########################################
######  build model and optimizer  ####### 
##########################################
from CLIP import clip
clip_model, preprocess = clip.load(name="ViT-B/16", device="cpu", download_root="/home/jovyan/clip_download_root")
from KPGT.src.model.light import LiGhTPredictor as LiGhT
kpgt = LiGhT(
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
        # input_drop=config['input_drop'],
        # attn_drop=config['attn_drop'],
        # feat_drop=config['feat_drop'],
        input_drop=0.,
        attn_drop=0.,
        feat_drop=0.,
        n_node_types=vocab.vocab_size
    )# .to("cuda")
kpgt.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("/home/jovyan/prompts_learning/KPGT/src/models/base.pth").items()})
print("Pre-trained weights of KPGT were loaded successfully!")


from model_zoo import CLIPM_KPGT
dev = torch.device('cuda', local_rank)
model = CLIPM_KPGT(kpgt, clip_model).to(dev)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# from copy import deepcopy
# ema = deepcopy(model).to(dev)  # Create an EMA of the model for use after training
# update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
# requires_grad(ema, False)
# ema.eval()

# best
# lr = 1e-4
# wd = 0.

# best
lr = 5e-5
wd = 1e-6

print(f'Set of Optimizer: lr:{lr}, weight_decay:{wd}')
model_params = [
                {'params': model.parameters(), 'lr': lr},
                # {'params': model.text_proc.parameters(), 'lr': 1e-4},
                # {'params': model.mol_encoder.parameters(), 'lr': 5e-5},
               ]


optims = 'adan'
# optims = "adam"
if optims == 'adan':
    from adan import Adan
    optimizer = Adan(model_params, betas=(0.98, 0.92, 0.99),weight_decay=wd, max_grad_norm=5.)
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
from KPGT.src.trainer.scheduler import PolynomialDecayLR
scheduler = PolynomialDecayLR(optimizer, warmup_updates=15000, tot_updates=300000,lr=lr, end_lr=1e-9,power=1)
cur_lr = scheduler.get_last_lr() 
print(f"Current learning rate is {cur_lr}.")



##########################################
######## start training our model ######## 
##########################################
gc.collect()
torch.cuda.empty_cache()
Epoch = 20
print("Let's start distributed training!")
n_steps = 15000
all_len = len(train_loader)
for e in range(0, Epoch):
    start = time.time()
    model.train()
    train_loader.sampler.set_epoch(e)
    for step_id, batched_data in enumerate(train_loader):
        (_, batched_graph, fps, mds, _, _, _, text) = batched_data
        batched_graph = batched_graph.to(dev)
        fps = fps.to(dev)
        mds = mds.to(dev)
        text = text.to(dev)
        
        loss = model([batched_graph, fps, mds], text)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        # update_ema(ema, model, 0.997)
        
        if step_id >= n_steps:
        # if step_id >= 0:
            if local_rank == 0:
                check_point = {
                'mol_encoder': model.module.mol_encoder.state_dict(),
                'text_proc': model.module.text_proc.state_dict(),
                }
                torch.save(check_point, f'./trained_weight/cliPM_KPGT_Epoch{e+1}_step{step_id}.pth') 
            break

        if not (step_id+1) % 1000:
            print(f"epoch: {e+1} / {Epoch},step {step_id} / {all_len}, loss: {loss.detach().cpu().numpy():.8f}")
            gc.collect()
            torch.cuda.empty_cache()
                
    
    ##########################################
    ####### start evaluating our model #######
    ##########################################
    # check_point = {
    #         'mol_encoder': model.mol_encoder.state_dict(),
    #         'text_proc': model.text_proc.state_dict(),
    #     }
    # torch.save(check_point, f'./trained_weight/cliPM_KPGT_Epoch{e+1}.pth') 


    end = time.time()
    print(f"epoch: {e+1} end ; cost time: {(end - start)/60.:.4f} min")
    
    # scheduler.step()
    cur_lr = scheduler.get_last_lr() 
    print(f"Current learning rate is {cur_lr}.")