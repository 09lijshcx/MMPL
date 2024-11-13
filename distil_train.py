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


@torch.no_grad()
def teacher_forward(teacher, mol):
    feats = teacher.get_feats(mol)
    return feats
    

##########################################
#########  construct dataloader  ######### 
##########################################
from tune_cls_pipeline import *
from KPGT.src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from KPGT.src.data.collator_tune import Collator_pretrain, Collator_tune
from KPGT.src.model_config import config_dict
config = config_dict['base']

vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
collator = Collator_pretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
train_dataset = MoleculeHERGDataset("train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, drop_last=True, collate_fn=collator)


val_dataset = MoleculeHERGTestDataset("val")
collator = Collator_tune(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
val_set = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)


collator = Collator_pretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
train_dataset2 = MoleculeBBBDataset("train")
train_loader2 = DataLoader(train_dataset2, batch_size=16, shuffle=True, num_workers=8, drop_last=True, collate_fn=collator)

val_dataset2 = MoleculeBBBDataset("test")
val_set2 = DataLoader(val_dataset2, batch_size=16, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)


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
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        # input_drop=config['input_drop'], # 0.0
        # attn_drop=config['attn_drop'],  # 0.1
        # feat_drop=config['feat_drop'],  # 0.1
        input_drop=0.0,
        attn_drop=0.,
        feat_drop=0.,
        n_node_types=vocab.vocab_size
    )# .to("cuda")
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
# kpgt.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("/home/jovyan/prompts_learning/KPGT/src/models/base.pth").items()})
# print("Pre-trained weights of KPGT were loaded successfully!")

from model_zoo import MMPL, KPGT
model = MMPL(kpgt).to(dev)

teacher_bbb = KPGT(kpgt).to(dev)
sd = torch.load("/home/jovyan/prompts_learning/trained_weight/KPGT_tune_bbb_Epoch2_val_auc_0.93055.pth")
teacher_bbb.load_state_dict(sd)
teacher_bbb.eval()
requires_grad(teacher_bbb, False)

teacher_herg = KPGT(kpgt).to(dev)
sd = torch.load("/home/jovyan/prompts_learning/trained_weight/KPGT_tune_herg_Epoch5_val_auc_0.88827.pth")
teacher_herg.load_state_dict(sd)
teacher_herg.eval()
requires_grad(teacher_herg, False)

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
                # {'params': model.parameters(), 'lr': lr, "weight_decay": wd},
    {'params': model.mol_encoder.parameters(), 'lr': 1e-5, "weight_decay": wd},
    {'params': model.text_proc.parameters(), 'lr': 1e-5, "weight_decay": wd},
    {'params': model.heads.parameters(), 'lr': 1e-3, "weight_decay": 1e-2},
    
    # {'params': model.mpim.parameters(), 'lr': 5e-5, "weight_decay": wd},
    {'params': model.mpim.model.parameters(), 'lr': 5e-5, "weight_decay": wd},
    {'params': model.mpim.attn_model.parameters(), 'lr': 5e-5, "weight_decay": wd},
    {'params': model.mpim.prom_emb.parameters(), 'lr': 3e-5, "weight_decay": wd},
    {'params': model.mpim.mol_emb.parameters(), 'lr': 3e-5, "weight_decay": wd},
    {'params': model.mpim.positional_embedding, 'lr': 5e-5, "weight_decay": wd},
    
    # {'params': model.mpim.moltex_attn.parameters(), 'lr': 1e-3, "weight_decay": 1e-2},
    {'params': model.mpim.moltex_attn.parameters(), 'lr': 1e-3, "weight_decay": wd},
    
    # {'params': model.mpim.attr_heads.parameters(), 'lr': 1e-3, "weight_decay": 1e-2}
               ]


optims = 'adan'
# optims = "sgd"
if optims == 'adan':
    from adan import Adan
    optimizer = Adan(model_params, betas=(0.98, 0.92, 0.99), max_grad_norm=5.)
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
# from KPGT.src.trainer.scheduler import PolynomialDecayLR
# scheduler = PolynomialDecayLR(optimizer, warmup_updates=355, tot_updates=355*15,lr=lr, end_lr=1e-9,power=1)
# cur_lr = scheduler.get_last_lr() 
# print(f"Current learning rate is {cur_lr}.")

# scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1./3., total_iters=5) # best
# scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=5)
# scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[5, 10])
# cur_lr = scheduler.get_last_lr() 
# print(f"Current learning rate is {cur_lr}.")


##########################################
########   build loss criterion   ######## 
##########################################
# attr_loss = 'l1'
# attr_loss = 'mse'
attr_loss = 'bce'
# attr_loss = 'ce'

# red = 'sum'
red = 'mean'

print(f"attribution loss is {attr_loss}, and reduction method is {red}.")
if attr_loss == 'l1':
    # cri_attr = nn.L1Loss(reduction=red)
    cri_attr = nn.SmoothL1Loss(reduction=red)
elif attr_loss == 'mse':
    cri_attr = nn.MSELoss(reduction=red)
elif attr_loss == 'bce':
    cri_attr = nn.BCEWithLogitsLoss(reduction=red)
elif attr_loss == 'ce':
    cri_attr = nn.CrossEntropyLoss(reduction=red)


##########################################
######## start training our model ######## 
##########################################
gc.collect()
torch.cuda.empty_cache()
Epoch = 100
best_val = 0.8
print("Let's start training!")

for e in range(0, Epoch):
    start = time.time()
    model.train()
    for step_id, batched_data in enumerate(train_loader):
        (_, batched_graph, fps, mds, _, _, _, label, logd, logp, pka, pkb, logsol, wlogsol) = batched_data
        batched_graph = batched_graph.to(dev)
        fps = fps.to(dev)
        mds = mds.to(dev)
        
        pred, attr_list, feat = model([batched_graph.clone(), fps, mds], ind=0)
        
        feat_t = teacher_forward(teacher_herg, [batched_graph.clone(), fps.detach(), mds.detach()])
        loss_distill = F.mse_loss(feat, feat_t)
         
        # loss_cls = F.binary_cross_entropy_with_logits(pred, label.unsqueeze(-1).to(dev))
        # weighted BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, label.unsqueeze(-1).to(dev), reduction='none')
        for i, lab in enumerate(label):
            if lab == 0:
                loss[i] = loss[i] * (2809./5687.)  # best
            else:
                loss[i] = loss[i] * (2878./5687.)
        if red == 'mean':
            loss_cls = loss.mean()
        elif red == 'sum':
            loss_cls = loss.sum()


#         loss_pka = F.binary_cross_entropy_with_logits(attr_list[2], pka.unsqueeze(-1).to(dev), reduction=red)
#         loss_pkb = F.binary_cross_entropy_with_logits(attr_list[3], pkb.unsqueeze(-1).to(dev), reduction=red)

#         loss_logd = cri_attr(attr_list[0], logd.unsqueeze(-1).to(dev))
#         loss_logp = cri_attr(attr_list[1], logp.unsqueeze(-1).to(dev))
        # loss_pka = cri_attr(attr_list[2], pka.unsqueeze(-1).to(dev))
        # loss_pkb = cri_attr(attr_list[3], pkb.unsqueeze(-1).to(dev))
        # loss_logsol = cri_attr(attr_list[4], logsol.unsqueeze(-1).to(dev))
        # loss_wlogsol = cri_attr(attr_list[5], wlogsol.unsqueeze(-1).to(dev))

        # loss_pka = cri_attr(attr_list[2], pka.to(dev))
        # loss_pkb = cri_attr(attr_list[3], pkb.to(dev))
        # loss_logd = cri_attr(attr_list[0], logd.to(dev))
        # loss_logp = cri_attr(attr_list[1], logp.to(dev))
        # loss_logsol = cri_attr(attr_list[4], logsol.to(dev))
        # loss_wlogsol = cri_attr(attr_list[5], wlogsol.to(dev))

        # loss_attr = loss_logd + loss_logp  + loss_logsol + loss_pka + loss_pkb + loss_wlogsol
            
        loss = loss_cls*0.5 + loss_distill * 1. # + loss_attr

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        update_ema(ema, model, 0.997)
        

        if not (step_id+1) % 40:
                print(f"epoch: {e+1} / {Epoch},step {step_id} / {len(train_loader)}, loss:{loss.detach().cpu().numpy():.4f}")
                print_loss(loss_cls, "hERG cls")
                # print_loss(loss_logd, "LogD")
                # print_loss(loss_logp, "LogP")
                # print_loss(loss_pka, "pKa")
                # print_loss(loss_pkb, "pKb")
                # print_loss(loss_logsol, "LogSol")
                # print_loss(loss_wlogsol, "wLogSol")
                print_loss(loss_distill, "Distillation")
                print()
                
    
    
    for step_id, batched_data in enumerate(train_loader2):
        (_, batched_graph, fps, mds, _, _, _, label, logd, logp, pka, pkb, logsol, wlogsol) = batched_data
        batched_graph = batched_graph.to(dev)
        fps = fps.to(dev)
        mds = mds.to(dev)
        
        pred, attr_list, feat = model([batched_graph.clone(), fps, mds], 1)
        
        feat_t = teacher_forward(teacher_bbb, [batched_graph.clone(), fps.detach(), mds.detach()])
        loss_distill = F.mse_loss(feat, feat_t)
         
        # loss_cls = F.binary_cross_entropy_with_logits(pred, label.unsqueeze(-1).to(dev))
        # weighted BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, label.unsqueeze(-1).to(dev), reduction='none')
        lambda_ = 2.
        for i, lab in enumerate(label):
            if lab == 0:
                # loss[i] = loss[i] * (6641./2042.)
                loss[i] = loss[i] * (4599./6641)  # best
                # loss[i] = loss[i] * (4599./6641) * torch.pow(torch.sigmoid(pred[i]), lambda_)
            else:
                # loss[i] = loss[i] * (6641./4599.)
                loss[i] = loss[i] * (2042./6641)
                # loss[i] = loss[i] * (2042./6641) * torch.pow(1. - torch.sigmoid(pred[i]), lambda_)# focal loss
        if red == 'mean':
            loss_cls = loss.mean()
        elif red == 'sum':
            loss_cls = loss.sum()


        # loss_pka = F.binary_cross_entropy_with_logits(attr_list[2], pka.unsqueeze(-1).to(dev), reduction=red)
        # loss_pkb = F.binary_cross_entropy_with_logits(attr_list[3], pkb.unsqueeze(-1).to(dev), reduction=red)

        # loss_logd = cri_attr(attr_list[0], logd.unsqueeze(-1).to(dev))
        # loss_logp = cri_attr(attr_list[1], logp.unsqueeze(-1).to(dev))
        # loss_pka = cri_attr(attr_list[2], pka.unsqueeze(-1).to(dev))
        # loss_pkb = cri_attr(attr_list[3], pkb.unsqueeze(-1).to(dev))
        # loss_logsol = cri_attr(attr_list[4], logsol.unsqueeze(-1).to(dev))
        # loss_wlogsol = cri_attr(attr_list[5], wlogsol.unsqueeze(-1).to(dev))

        # loss_pka = cri_attr(attr_list[2], pka.to(dev))
        # loss_pkb = cri_attr(attr_list[3], pkb.to(dev))
        # loss_logd = cri_attr(attr_list[0], logd.to(dev))
        # loss_logp = cri_attr(attr_list[1], logp.to(dev))
        # loss_logsol = cri_attr(attr_list[4], logsol.to(dev))
        # loss_wlogsol = cri_attr(attr_list[5], wlogsol.to(dev))

        # loss_attr = loss_logd + loss_logp  + loss_logsol + loss_pka + loss_pkb + loss_wlogsol
            
        loss = loss_cls*0.5 + loss_distill # + loss_attr

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        update_ema(ema, model, 0.997)
        

        if not (step_id+1) % 40:
                print(f"epoch: {e+1} / {Epoch},step {step_id} / {len(train_loader)}, loss:{loss.detach().cpu().numpy():.4f}")
                print_loss(loss_cls, "BBB cls")
                # print_loss(loss_logd, "LogD")
                # print_loss(loss_logp, "LogP")
                # print_loss(loss_pka, "pKa")
                # print_loss(loss_pkb, "pKb")
                # print_loss(loss_logsol, "LogSol")
                # print_loss(loss_wlogsol, "wLogSol")
                print_loss(loss_distill, "Distillation")
                print()
    ##########################################
    ####### start evaluating our model #######
    ##########################################
    model.eval()
    print("evaluating...")
    with torch.no_grad():
        all_pred = None 
        all_lab = None
        for step_id, batched_data in enumerate(val_set):
            (_, batched_graph, fps, mds, label) = batched_data
            batched_graph = batched_graph.to(dev)
            fps = fps.to(dev)
            mds = mds.to(dev)
            
            # pred = model([batched_graph, fps, mds])
            pred = ema([batched_graph, fps, mds], 0)

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
        torch.save(ema.state_dict(), f'./trained_weight/MMPL_spcTeacher_tune_herg_Epoch{e+1}_val_auc_{best_val:.5f}.pth') 
        
    
    with torch.no_grad():
        all_pred = None 
        all_lab = None
        for step_id, batched_data in enumerate(val_set2):
            (_, batched_graph, fps, mds, _, _, _, label, logd, logp, pka, pkb, logsol, wlogsol) = batched_data
            batched_graph = batched_graph.to(dev)
            fps = fps.to(dev)
            mds = mds.to(dev)
            
            # pred = model([batched_graph, fps, mds])
            pred = ema([batched_graph, fps, mds], 1)

            pred = torch.sigmoid(pred)
            # pred = torch.softmax(pred, dim=-1)[:, 1]
            # total_loss += cri_mae(pred, label.unsqueeze(-1))
            all_pred = pred if all_pred is None else torch.cat([all_pred, pred], dim=0)
            all_lab = label if all_lab is None else torch.cat([all_lab, label], dim=0)
    auc = compute_AUC(all_lab.cpu().detach(), all_pred.cpu().detach())
    print(f"epoch: {e+1} / {Epoch}, test AUC: {auc:.5f}")


    end = time.time()
    print(f"epoch: {e+1} end ; cost time: {(end - start)/60.:.4f} min")
    gc.collect()
    torch.cuda.empty_cache()
    
    # scheduler.step()
    # cur_lr = scheduler.get_last_lr() 
    # print(f"Current learning rate is {cur_lr}.")