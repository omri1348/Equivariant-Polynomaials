import torch_geometric.datasets as datasets
from torch_geometric.data.separate import separate
from torch_geometric.utils import to_dense_adj
import torch
import torch.nn.functional as F
import pickle
import os
from utils import compute_poly
from torch_geometric.datasets import TUDataset
from torch.utils.data import DataLoader
from torch_geometric.utils.convert import from_networkx
import networkx as nx

SR_ARR =  ['sr16622','sr251256', 'sr261034','sr281264', 'sr291467', 'sr351668', 'sr351899','sr361446', 'sr401224']


def get_loaders(cfg):
    if cfg.dataset == 'ZINC':
        train_loader = DataLoader(ZincDataset(cfg, 'data/ZINC', subset=cfg.subset),
        batch_size=cfg.batch_size, drop_last=True, collate_fn=custom_collate_fn, shuffle=True)
        val_loader = DataLoader(ZincDataset(cfg, 'data/ZINC', subset=cfg.subset, split='val'), 
        batch_size=cfg.batch_size, drop_last=False, collate_fn=custom_collate_fn, shuffle=False)
        test_loader = DataLoader(ZincDataset(cfg, 'data/ZINC', subset=cfg.subset, split='test'), 
        batch_size=cfg.batch_size, drop_last=False, collate_fn=custom_collate_fn, shuffle=False)
    elif cfg.dataset == 'Alchemy':
        train_dataset, val_dataset, test_dataset = get_alchemy_dataset('data/Alchemy')
        train_loader = DataLoader(AlchemyDataset(cfg, train_dataset),
        batch_size=cfg.batch_size, drop_last=True, collate_fn=custom_collate_fn, shuffle=True)
        val_loader = DataLoader(AlchemyDataset(cfg, val_dataset, split='val'), 
        batch_size=cfg.batch_size, drop_last=True, collate_fn=custom_collate_fn, shuffle=True)
        test_loader = DataLoader(AlchemyDataset(cfg, test_dataset, split='test'), 
        batch_size=cfg.batch_size, drop_last=True, collate_fn=custom_collate_fn, shuffle=True)
    elif cfg.dataset == 'SR':
        train_loader = DataLoader(PPGNSRDataset(cfg.d_name, 'data/SR/raw', cfg.use_poly_data),
        batch_size=cfg.batch_size, drop_last=False, collate_fn=custom_collate_fn, shuffle=True)
        val_loader = None
        test_loader = None
    return train_loader, val_loader, test_loader

def custom_collate_fn(batch_list):
    n_max = 0
    n_list = []
    for (data,y) in batch_list:
        n = data.shape[-1]
        n_list.append(n)
        if n > n_max:
            n_max = n
    data_list = []
    mask_list = []
    y_list = []
    for (data,y) in batch_list:
        n = data.shape[-1]
        mask = torch.ones((n,n)).unsqueeze(0)
        pad_dif = n_max - n
        if pad_dif > 0:
            data_list.append(F.pad(data,(0, pad_dif, 0, pad_dif), "constant", 0))
            mask_list.append(F.pad(mask,(0, pad_dif, 0, pad_dif), "constant", 0))
        else:
            data_list.append(data)
            mask_list.append(mask)
        y_list.append(y)
    data_list = torch.stack(data_list, dim=0)
    mask_list = torch.stack(mask_list, dim=0)
    y_list = torch.cat(y_list, dim=0)
    return data_list, mask_list, y_list, torch.tensor(n_list).view(-1,1).float()

def analyze_poly_response(path):
    for f in os.listdir(path):
        response = 0
        slice_active = 0
        if f.endswith('.pth'):
            if 'filter' in f:
                continue
            print(f)
            where_arr = []
            poly_arr = torch.load(os.path.join(path, f))
            for poly in poly_arr:
                if poly.max() >0:
                    response += 1
                    slice_active += (poly.max(dim=-1)[0].max(dim=-1)[0] > 0).sum() / poly.shape[0]
                    where_arr.append(torch.argwhere(poly.max(dim=-1)[0].max(dim=-1)[0] > 0).view(-1))
            where = torch.unique(torch.cat(where_arr, dim=0))
            slice_active = slice_active / response
            
            print('{} - active graphs  {} / {}'.format(f, response, len(poly_arr)))
            print('{} total slices'.format(poly.shape[0]))
            print('{} active slices'.format(slice_active))
            print('total used indices {}'.format(len(where)))
            print(where)
            filtered_arr = [poly[where,:,:] for poly in poly_arr]
            print(filtered_arr[0].shape)
            torch.save(filtered_arr,os.path.join(path,f.split('.')[0]+'_filter.pth'))

def parse_sr_ppgn_poly_data(src_path,root):
    for d in SR_ARR:
        dataset = PPGNSRDataset(d,root)
        for deg in [5,6]:
            data_list = []
            name = os.path.join(src_path,'data/equivaraint_polynomials','ppgn_{}.pkl'.format(deg))
            poly_list = pickle.load(open(name,'rb'))
            for adj in dataset:
                poly = compute_poly(adj[0], deg, poly_list)
                data_list.append(poly.cpu())
            torch.save(data_list, 'data/SR/poly/ppgn/poly_{}_{}.pth'.format(d,deg))

def parse_alchemy_ppgn_poly_data(src_path, root):
    split_arr = ['train', 'val','test']
    dataset_arr = get_alchemy_dataset(os.path.join(src_path, root))
    for i,dataset in enumerate(dataset_arr):
        split = split_arr[i]
        for deg in [5,6]:
            data_list = []
            name = os.path.join(src_path,'data/equivaraint_polynomials','ppgn_{}.pkl'.format(deg))
            poly_list = pickle.load(open(name,'rb'))
            for data in dataset:
                adj = to_dense_adj(data.edge_index)
                poly = compute_poly(adj, deg, poly_list)
                data_list.append(poly.cpu())
            torch.save(data_list, 'data/Alchemy/poly/ppgn/poly_{}_{}.pth'.format(deg,split))

def parse_zinc_ppgn_poly_data(src_path, root, subset=False):
    for split in ['train', 'val','test']:
        base_dataset = datasets.ZINC(os.path.join(src_path, root), subset, split)
        for deg in [5,6]:
            data_list = []
            name = os.path.join(src_path,'data/equivaraint_polynomials','ppgn_{}.pkl'.format(deg))
            poly_list = pickle.load(open(name,'rb'))
            for i in range(base_dataset.len()):
                data = separate(
                cls=base_dataset.data.__class__,
                batch=base_dataset.data,
                idx=i,
                slice_dict=base_dataset.slices,
                decrement=False,)
                adj = to_dense_adj(data.edge_index)
                poly = compute_poly(adj, deg, poly_list)
                data_list.append(poly.cpu())
            if subset:
                torch.save(data_list, 'data/ZINC/subset_poly/ppgn/poly_{}_{}.pth'.format(deg,split))
            else:
                torch.save(data_list, 'data/ZINC/full_poly/ppgn/poly_{}_{}.pth'.format(deg,split))

def get_alchemy_dataset(path):
    infile = open(os.path.join(path,"train_al_10.index"), "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    infile = open(os.path.join(path,"val_al_10.index"), "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open(os.path.join(path,"test_al_10.index"), "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    indices = indices_train
    indices.extend(indices_val)
    indices.extend(indices_test)

    dataset = TUDataset(path, name="alchemy_full")[indices]
    print('Num points:', len(dataset))

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std

    train_dataset = dataset[0:10000]
    val_dataset = dataset[10000:11000]
    test_dataset = dataset[11000:]
    return train_dataset, val_dataset, test_dataset

class PPGNSRDataset(torch.utils.data.Dataset):
    def __init__(self,name,root, use_poly_data=False):
        """
            Loading SR datasets
        """
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = root
        if not os.path.isfile(os.path.join(data_dir,name+'.pkl')):
            g_path = os.path.join(data_dir,'{}.g6'.format(name))
            G_list = [from_networkx(g) for g in nx.read_graph6(g_path)]
            with open(os.path.join(data_dir,name+'.pkl'),"wb") as f_tmp:
                pickle.dump(G_list, f_tmp)
        if use_poly_data:
            poly_data = [torch.load('data/SR/poly/ppgn/poly_{}_{}_filter.pth'.format(name, d)) for d in range(5,7)]
            self.poly_data = [torch.cat([poly_data[k][j] for k in range(len(poly_data))],dim=0)for j in range(len(poly_data[0]))]
        with open(os.path.join(data_dir,name+'.pkl'),"rb") as f:
            self.data = pickle.load(f)
            if use_poly_data:
                self.data = [torch.cat([to_dense_adj(g.edge_index),self.poly_data[i]],dim=0)
                              for i,g in enumerate(self.data)]
            else:
                self.data = [to_dense_adj(g.edge_index) for g in self.data]
        print("[I] Finished loading.")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return (self.data[idx], torch.ones(1))
    

        

class AlchemyDataset():
    def __init__(self, cfg, dataset, split='train'):
        self.data_list = []
        self.cfg = cfg
        print('process {} data'.format(split))
        if cfg.use_poly_data:
            poly_data = [torch.load('data/Alchemy/poly/{}/poly_{}_{}_filter.pth'.format(cfg.model, d, split)) for d in range(5,cfg.d_expressive+1)]
            self.poly_data = [torch.cat([poly_data[k][j] for k in range(len(poly_data))],dim=0)for j in range(len(poly_data[0]))]
        for i in range(len(dataset)):
            data = dataset[i]
            self.data_list.append((self.pyg_to_tensor(data, i), data.y))
        print('Done')

    def pyg_to_tensor(self, data, i):
        adj = to_dense_adj(data.edge_index)
        if self.cfg.use_poly_data:
            poly_data = self.poly_data[i]
            if self.cfg.normalize_features:
                poly_norm = poly_data.norm(dim=0,keepdim=True)
                poly_norm = torch.where(poly_norm > 0, poly_norm, 1)
                poly_data = poly_data/poly_norm
                poly_norm = poly_data.norm(dim=[1,2],keepdim=True)
                poly_norm = torch.where(poly_norm > 0, poly_norm, 1)
                poly_data = poly_data/poly_norm
            adj = torch.cat((poly_data, adj), dim=0)
        weighted_adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze(0) # n,n,d
        weighted_adj = weighted_adj.transpose(1,2).transpose(0,1)
        node_features = torch.diag_embed(data.x.t())
        tensor_rep = torch.cat((adj, weighted_adj, node_features),dim=0)
        return tensor_rep

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data    


class ZincDataset():
    def __init__(self, cfg, root, subset=False, split='train', transform=None, pre_transform=None, pre_filter=None):
        base_dataset = datasets.ZINC(root, subset, split, transform, pre_transform, pre_filter)
        self.data_list = []
        self.cfg = cfg
        print('process {} data'.format(split))
        if cfg.use_poly_data:
            if cfg.subset:
                prefix = 'data/ZINC/subset_poly/{}'.format(cfg.model)
            else: 
                prefix = 'data/ZINC/full_poly/{}'.format(cfg.model)
            poly_data = [torch.load(prefix + '/poly_{}_{}_filter.pth'.format(d, split)) for d in range(5,cfg.d_expressive+1)]
            self.poly_data = [torch.cat([poly_data[k][j] for k in range(len(poly_data))],dim=0)for j in range(len(poly_data[0]))]
        for i in range(base_dataset.len()):
            data = separate(
            cls=base_dataset.data.__class__,
            batch=base_dataset.data,
            idx=i,
            slice_dict=base_dataset.slices,
            decrement=False,)
            self.data_list.append((self.pyg_to_tensor(data, i), data.y))
        print('Done')

    def pyg_to_tensor(self, data, i):
        adj = to_dense_adj(data.edge_index)
        if self.cfg.use_poly_data:
            poly_data = self.poly_data[i]
            if self.cfg.normalize_features:
                poly_norm = poly_data.norm(dim=0,keepdim=True)
                poly_norm = torch.where(poly_norm > 0, poly_norm, 1)
                poly_data = poly_data/poly_norm
                poly_norm = poly_data.norm(dim=[1,2],keepdim=True)
                poly_norm = torch.where(poly_norm > 0, poly_norm, 1)
                poly_data = poly_data/poly_norm
            adj = torch.cat((poly_data, adj), dim=0)
        weighted_adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)
        node_features = torch.diag(data.x.squeeze() + 1).unsqueeze(0)
        tensor_rep = torch.cat((adj, weighted_adj, node_features),dim=0)
        return tensor_rep

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data

if __name__ == "__main__":
    src_path = '/home/omrip/data/Repositories/equivariant_polynomials/'
    path = os.path.join(src_path, 'data/ZINC')
    parse_zinc_ppgn_poly_data(src_path,'data/ZINC',subset=True)
    analyze_poly_response(os.path.join(src_path,'data/ZINC/subset_poly/ppgn'))
    parse_alchemy_ppgn_poly_data(src_path,'data/Alchemy')
    analyze_poly_response(os.path.join(src_path,'data/Alchemy/poly/ppgn'))
    parse_sr_ppgn_poly_data(src_path,'data/SR/raw')
    analyze_poly_response(os.path.join(src_path,'data/SR/poly/ppgn'))