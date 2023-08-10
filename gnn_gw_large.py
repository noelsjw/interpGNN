import argparse
import configparser
from datetime import datetime
import csv
import copy
import random
import numpy as np
import pickle
import torch
import networkx as nx
import os
import numpy as np 
from ogb.nodeproppred import PygNodePropPredDataset
from dgl.data import citation_graph
# from dgl.data import AmazonCoBuy, RedditDataset
# from dgl import add_self_loop

from lib.utils import generate_mask_random


from models.model import GCN_GW
from lib.logger import get_logger
from lib.utils import init_seed

from lib.utils_large import train_model, load_aminer

from torch_geometric.datasets import Planetoid, WikiCS
from torch_geometric.datasets import  Amazon, Coauthor,CitationFull, Flickr, Reddit, Reddit2, Actor
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx, add_self_loops

from torch_geometric.transforms import NormalizeFeatures


#*******************************MAIN BODY*************************************#

MODE = 'train'
DEBUG = 'False'
DATASET = 'cora'
DEVICE = 'cuda:3'

# Get configuration
config_file = 'large_graph.conf'
print('Read Configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)
# parser
args = argparse.ArgumentParser(description='arguments')

# args.add_argument('--dataset', default=DATASET,type=str)
args.add_argument('--mode', default=MODE, type=str)
args.add_argument('--device', default=config['log']['device'], type=str, help='indices of GPUs')
args.add_argument('--debug', default=config['log']['debug'], type=eval)
args.add_argument('--cuda', default=True, type=bool)

# data
args.add_argument("--seed", default=config['data']['seed'], help="seed",type=int)
args.add_argument("--dataset", default=config['data']['dataset'], help="dataset", type=str)
args.add_argument("--valid_ratio", default=config['data']['valid_ratio'], help="valid num",type=float)
args.add_argument("--test_ratio", default=config['data']['test_ratio'], help="test num",type=float)
args.add_argument("--num_parts", default=config['data']['num_parts'], help="num_clusters",type=int)




args.add_argument("--train_ratio", default=config['training']['train_ratio'], type=float)
args.add_argument("--batch_size", default=config['training']['batch_size'], type=int)


# Augments for model
args.add_argument('--model', default=config['model']['model'], type=str)
args.add_argument('--epoch', default=config['model']['epoch'], type=int)
args.add_argument('--lr', default=config['model']['lr'], type=float)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--gnn_hidden_dim', default=config['model']['gnn_hidden_dim'], type=int)

args.add_argument('--node2vec_emb_dim', default=config['model']['emb_dim'], type=int)
args.add_argument('--gw_ratio', default=config['model']['gw_ratio'], type=float)
args.add_argument('--init_memory', default=config['model']['init_memory'], type=eval)
args.add_argument('--shared_memory_attention', default=config['model']['shared_memory_attention'], type=eval)
args.add_argument('--shared_memory_percentage', default=config['model']['shared_memory_percentage'], type=float)
args.add_argument('--mem_slots', default=config['model']['mem_slots'], type=int)
args.add_argument('--encoder_attention_heads', default=config['model']['encoder_attention_heads'], type=int)
args.add_argument('--encoder_embed_dim', default=config['model']['encoder_embed_dim'], type=int)
args.add_argument('--encoder_ffn_embed_dim', default=config['model']['encoder_ffn_embed_dim'], type=int)
args.add_argument('--attention_dropout', default=config['model']['attention_dropout'], type=float)
args.add_argument('--topk_ratio', default=config['model']['topk_ratio'], type=float)
args.add_argument('--encoder_normalize_before', default=config['model']['encoder_normalize_before'], type=eval)
args.add_argument('--use_nfm', default=config['model']['use_nfm'], type=eval)
args.add_argument('--null_attention', default=config['model']['null_attention'], type=eval)
args.add_argument('--self_attention', default=config['model']['self_attention'], type=eval)
args.add_argument('--use_topk', default=config['model']['use_topk'], type=eval)
args.add_argument('--topk', default=config['model']['topk'], type=int)
args.add_argument('--num_steps', default=config['model']['topk'], type=int)
args.add_argument('--regressive', default=config['model']['regressive'], type=eval)
args.add_argument('--dropout', default=config['model']['dropout'], type=float)



#log
args.add_argument('--log_dir', default=config['log']['log_dir'], type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)

args = args.parse_args()


init_seed(args.seed)
print(args.dataset)

#Config log path
current_time = datetime.now().strftime('%mM%dD%H:%M')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir, args.log_dir, args.dataset, current_time)
logger = get_logger(log_dir, debug=args.debug)
for arg, value in sorted(vars(args).items()):
    logger.info("%s: %r", arg, value)

# args.batch_size=128




# data
if args.dataset in ["cora", "Pubmed", 'citeSeer']:
    dataset = Planetoid(root='./dataset/Planetoid', name=args.dataset, split='public')
elif args.dataset in ['corafull']:
    dataset = CitationFull('./dataset','Cora' )
elif args.dataset in ["Computers", "Photo"]:
    dataset = Amazon('./dataset', args.dataset)
    # data = AmazonCoBuy('computers')
elif args.dataset == 'reddit':
    dataset = Reddit(root = './dataset/reddit')
elif args.dataset == 'reddit2':
    dataset = Reddit2(root = './dataset/reddit2')
elif args.dataset == 'ogbn-products':
    dataset = PygNodePropPredDataset(name='ogbn-products', root='./dataset')
elif args.dataset == 'aminer':
    dataset = load_aminer()
elif args.dataset == 'flickr':
    dataset = Flickr('./dataset/flickr')
elif args.dataset == 'wikics':
    dataset = WikiCS('./dataset/wikics', transform=NormalizeFeatures())
else:
    raise NotImplementedError(f"Dataset {args.dataset} not supported.")


# Data prepare

# graph_with_features = data[0]
data = dataset[0]
if args.dataset == 'ogbn-products':
    data.y = data.y.squeeze(1)
    args.dataset = 'ogbn_products'

node_features = data.x
node_num = data.x.size(0)
num_features = node_features.shape[1]
node_labels = data.y
num_labels = int(node_labels.max().item() + 1)

# Load node2vec positional embedding
pe_path = os.path.join('dataset', args.dataset, 'node2vec', str(args.node2vec_emb_dim)+'_pos_emb.pt')
pe = torch.load(pe_path)
data.pe = pe

importance_list = list(range(node_num))
train_mask, valid_mask, test_mask = generate_mask_random(
    importance_list.copy(),
    train_num=int (args.train_ratio * node_num),
    valid_num=int(args.valid_ratio * node_num),
    test_num=int (args.test_ratio * node_num)
)




data.train_mask = train_mask
data.test_mask = test_mask
data.val_mask = valid_mask

modelclass = GCN_GW
# Add self-loop
data.edge_index = add_self_loops(data.edge_index)[0]






best_test_acc, tpr, fpr = train_model(modelclass, 
                                    args.num_layers,
                                    data.train_mask, 
                                    data.val_mask, 
                                    data.test_mask, 
                                    # graph_with_features,
                                    data,  
                                    node_features,
                                    node_labels,
                                    num_features,
                                    num_labels,
                                    epoch_num=args.epoch, 
                                    groups=None,
                                    seed=args.seed, 
                                    lr=args.lr,
                                    device=args.device,
                                    args=args,                                    
                                    logger=logger)



logger.info("overall accuraccy: ")
logger.info(best_test_acc)
# logger.info("struc_closeness: ")
# logger.info(struc_closeness)

