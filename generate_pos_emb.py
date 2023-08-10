import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_add_pool, GCNConv, SAGEConv
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
import os.path as osp
from tqdm import tqdm


import argparse
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


import argparse
import time

# from lib.logger import Logger
# from lib.eval import get_split, SVMEvaluator
# from lib.logger import get_logger
from lib.utils import *
from models.position_emb import position_emb

from models.node2vec import Node2Vec

import os 
import configparser
from datetime import datetime


#**************************************************************************#
#*******************************Main File**********************************#
#**************************************************************************#

MODE = 'train'
DEBUG = 'True'
DATASET = 'ogbn-arxiv'
DEVICE = 'cuda:3'

# get_configuration
config_file = './configs/pos_emb.conf'
print('Read Configuration file: %s' % (config_file))
# parser
config = configparser.ConfigParser()
config.read(config_file)

args = argparse.ArgumentParser(description="Graph Transformer with global workspace and node2vec")
args.add_argument('--mode', default=MODE, type=str)
args.add_argument('--device', default=config['log']['device'], type=str, help='indices of GPUs')
args.add_argument('--debug', default=config['log']['debug'], type=eval)
args.add_argument('--cuda', default=True, type=bool)

# data
args.add_argument('--dataset', type=str, default=config['data']['dataset'])
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)

# Graph Transformer Embedding

args.add_argument('--pos_emb', default=config['TransEmb']['pos_emb'], type=str)
args.add_argument('--pos_emb_dim', default=config['TransEmb']['pos_emb_dim'], type=int)
args.add_argument('--struc_emb', default=config['TransEmb']['struc_emb'])


# node2vec module argumentation
args.add_argument('--node2vec_emb_dim', default=config['node2vec']['embedding_dim'], type=int)
args.add_argument('--node2vec_walk_length', default=config['node2vec']['walk_length'], type=int)
args.add_argument('--node2vec_context_size', default=config['node2vec']['context_size'], type=int)
args.add_argument('--node2vec_walks_per_node', default=config['node2vec']['walks_per_node'], type=int)
args.add_argument('--node2vec_batch_size', default=config['node2vec']['batch_size'], type=int)
args.add_argument('--node2vec_path', default=config['node2vec']['path'], type=str)
args.add_argument('--node2vec_epochs', default=config['node2vec']['epochs'], type=int)
# parser.add_argument('--device', type=int, default=0)
# parser.add_argument('--dataset', type=str, default='ogb-arxiv')
args.add_argument('--node2vec_load', default=config['node2vec']['load_model'], type=eval)
args.add_argument('--node2vec_save', default=config['node2vec']['save_model'], type=eval)


# Training details
args.add_argument('--lr', default=config['train']['lr_init'], type=float)
args.add_argument('--epochs',  default=config['train']['epochs'], type=int)
# parser.add_argument('--log_steps', type=int, default=1)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)


args = args.parse_args()


init_seed(args.seed)

dataset = load_dataset('dataset', args.dataset)
data = dataset[0]
data['pe'] = position_emb(args, data=data, pe_method=args.pos_emb).to(args.device)
num_class = len(torch.unique(data.y))


init_seed(args.seed)
dataset = load_dataset('dataset', args.dataset)
