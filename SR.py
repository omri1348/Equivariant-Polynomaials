import hydra
import os
import wandb
import torch
import torch.nn as nn
from models.ppgn import PPGN
from optimizer import Lamb
from utils import set_seed, update_cfg_dims, get_name, set_gpu
from dataset import get_loaders


def model_eval(model, device, data_loader):
    model.eval()
    pred_arr = []
    with torch.no_grad():
        for _, (data, mask, y, n_list) in enumerate(data_loader):
            data = data.to(device)
            mask = mask.to(device)
            n_list = n_list.to(device)

            print(data.shape)
            pred = model(data,mask,n_list)
            pred_arr.append(pred)

        pred = torch.cat(pred_arr,dim=0)
        mm = torch.pdist(pred, p=2)
        wrong = (mm < 0.01).sum().item()
        metric = wrong / mm.shape[0]

    return metric    


@hydra.main(version_base=None, config_path='.', config_name='SR')
def main(cfg):

    # set seed
    if cfg.seed > 0:
        set_seed(cfg)

    cfg = update_cfg_dims(cfg)
    print(cfg)



    torch.set_num_threads(1)
    set_gpu('-1')
    device = torch.device("cuda:"+os.environ["CUDA_VISIBLE_DEVICES"])
    
    train_loader, _, _ = get_loaders(cfg)

    if cfg.model == 'ppgn':
        net = PPGN(cfg).to(device)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("total number of parametets {}".format(pytorch_total_params))
    metric = model_eval(net, device, train_loader)
    print('non-distingushable models percentage in {} is {}'.format(cfg.d_name, metric))


    

if __name__ == '__main__':
    main()