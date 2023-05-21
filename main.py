import hydra
import os
import wandb
import torch
import torch.nn as nn
from models.ppgn import PPGN
from optimizer import Lamb
from utils import set_seed, update_cfg_dims, get_name, set_gpu
from dataset import get_loaders

def set_batch_device(batch,device):
    data, mask, label, n_list = batch
    data = data.to(device)
    mask = mask.to(device)
    label = label.to(device)
    n_list = n_list.to(device)
    return data, mask, label, n_list
    


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(cfg):

    # set seed
    if cfg.seed > 0:
        set_seed(cfg)

    cfg = update_cfg_dims(cfg)
    print(cfg)

    exp_name = get_name(cfg)
    print(exp_name)
    if cfg.wandb:
        wandb.init(project=cfg.project, name=exp_name+'_table')


    torch.set_num_threads(1)
    set_gpu('-1')
    device = torch.device("cuda:"+os.environ["CUDA_VISIBLE_DEVICES"])
    
    mae_loss = nn.L1Loss()
    train_loader, val_loader, test_loader = get_loaders(cfg)

    if cfg.model == 'ppgn':
        net = PPGN(cfg).to(device)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("total number of parametets {}".format(pytorch_total_params))
    if cfg.wandb:
        wandb.log({"parameter number": pytorch_total_params})

    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(),lr=cfg.lr)
    elif cfg.optimizer == 'lamb':
        optimizer = Lamb(net.parameters(),lr=cfg.lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=cfg.lr_reduce_factor,
                                                    patience=cfg.lr_schedule_patience,
                                                    verbose=True)

    for epoch in range(cfg.epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_test_loss = 0
        
        for batch_idx, batch in enumerate(train_loader, start=1):
            data, mask, label, n_list = set_batch_device(batch, device)

            optimizer.zero_grad()
            pred = net(data, mask, n_list)
            loss = mae_loss(pred, label)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= (batch_idx)

        for batch_idx, batch in enumerate(val_loader, start=1):
            data, mask, label, n_list = set_batch_device(batch, device)

            pred = net(data, mask, n_list)
            loss = mae_loss(pred, label)

            epoch_val_loss += loss.item()
        epoch_val_loss /= (batch_idx)

        for batch_idx, batch in enumerate(test_loader, start=1):
            data, mask, label, n_list = set_batch_device(batch, device)

            pred = net(data, mask, n_list)
            loss = mae_loss(pred, label)

            epoch_test_loss += loss.item()
        epoch_test_loss /= (batch_idx)
    
        
        if cfg.wandb:
            wandb.log({"train loss": epoch_train_loss, "val loss": epoch_val_loss,
            "test loss": epoch_test_loss, "epoch":epoch})
        print("Epoch {}, Train Loss {}, Val Loss {}, Test Loss {}".format(epoch,
        epoch_train_loss, epoch_val_loss, epoch_test_loss))
        
        # scheduler.step()
        scheduler.step(epoch_val_loss)
        if cfg.wandb:
            wandb.log({'lr':optimizer.param_groups[0]['lr'], "epoch":epoch})
        if optimizer.param_groups[0]['lr'] < cfg.min_lr:
            break


if __name__ == '__main__':
    main()