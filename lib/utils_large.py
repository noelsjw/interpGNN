from typing import *
import os
import torch
import dgl
import random
import numpy as np
import networkx as nx
import json

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
import pickle
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data 
from torch_geometric_autoscale.models import GCN as GAS


def load_aminer():
    path = os.path.join(
        './dataset', 'aminer'
    )
    adj = pickle.load(open(os.path.join(path, 'aminer.adj.sp.pkl'), 'rb'))
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    edge_index = edge_index.to(torch.int64)
    features = pickle.load(
            open(os.path.join(path, "aminer.features.pkl"), "rb"))
    features = torch.from_numpy(features).to(torch.float32)
    
    labels = pickle.load(
            open(os.path.join(path, "aminer.labels.pkl"), "rb"))
    labels = np.argmax(labels, axis=1)
    labels = torch.from_numpy(labels).to(torch.int64)
    data = Data(x=features, edge_index=edge_index, y=labels)
    return [data]



def train_model(model_class,
                num_layers,
                train_mask,
                valid_mask,
                test_mask,
                data,
                node_features,
                node_labels,
                n_features,
                n_labels,
                epoch_num=1000,
                groups=None,
                seed=42,
                device="cuda:0",
                lr=0.01,
                args=None,
                logger=None):
    '''
    :param model_class: One of MLP, GAT, GCN
    :param train_mask: get from generate_mask
    :param valid_mask: get from generate_mask
    :param test_mask: get from generate_mask
    :param graph: get from dgl.data graph with features
    :param node_features:
    :param node_labels:
    :param n_features:
    :param n_labels:
    :param epoch_num:
    :param groups: checkmethod
    '''
    cluster_data = ClusterData(data, num_parts=args.num_parts, recursive=False, save_dir='./dataset/{}/processed'.format(args.dataset))
    train_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=True, num_workers=12)
    # subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
                                #   shuffle=False, num_workers=12)

    best_valid_acc = 0
    best_test_acc = 0
    best_epoch = 0
    tpr = []
    fpr = []
    model = model_class(n_nodes=data.num_nodes,
                        in_feats=n_features,
                        out_feats=n_labels,
                        num_layers=num_layers,
                        n_units=args.gnn_hidden_dim,
                        dropout=0.5,
                        args=args).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    # acc_best = 0
    for epoch in range(epoch_num):
        model.train()
        total_loss = total_nodes = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.pe, batch.edge_index)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward() 
            opt.step()
            
        # data = data.to(device)
        # node_features = node_features.to(device)
        # node_labels = node_labels.to(device)
        # logits = model(graph, node_features)
        
            nodes = batch.train_mask.sum().item()
            total_loss += loss.item() * nodes 
            total_nodes += nodes
        epoch_loss = total_loss / total_nodes
        if (epoch + 1) % 10 == 0:
            test_acc = evaluate(model, train_loader, args)            
            logger.info("epoch:{}/{} loss: {}".format(epoch, epoch_num, epoch_loss))
            # logger.info("valid acc: {}".format(acc))
            logger.info("test acc: {}".format(test_acc))         
            if test_acc.item() > best_test_acc:
                best_test_acc = test_acc.item()
        # acc, _, _ = evaluate(model,
        #                      node_labels,
        #                      valid_mask,
        #                      data,
        #                      node_features,
        #                      groups=groups)

        # if best_valid_acc < acc:
        #     best_valid_acc = acc
        #     acc_test, tpr_tmp, fpr_tmp = evaluate(model,
        #                                           node_labels,
        #                                           test_mask,
        #                                           data,
        #                                           node_features,
        #                                           groups=groups)
        #     best_epoch = epoch
        #     tpr = tpr_tmp
        #     fpr = fpr_tmp
        #     best_test_acc = acc_test
        # if (epoch + 1) % 1 == 0:
            # logger.info("epoch:{}/{} loss: {}".format(epoch, epoch_num, loss.item()))
            # logger.info("valid acc: {}".format(acc))
            # logger.info("test acc: {}".format(acc_test)) 
    logger.info('best_test_acc: {}'.format(best_test_acc))
    logger.info('best epoch: {}'.format(best_epoch))
    logger.info(tpr)
    logger.info(fpr)
    return best_test_acc, tpr, fpr


@torch.no_grad()
def evaluate(model, test_loader, args):
    model.eval()
    # out = model.inference(data.x)
    total_test_examples = 0
    total_correct = 0
    for data in test_loader:
        data = data.to(args.device)
        if data.test_mask.sum()==0:
            continue
        # pred = model(data)[data.test_mask, data.ed]
        out = model(data.x, data.pe, data.edge_index, plot=data)[data.test_mask]

        _, indices = torch.max(out, dim=1)
        labels = data.y[data.test_mask]
        correct = torch.sum(indices == labels)
        num_test_examples = data.test_mask.sum().item()
        total_test_examples += num_test_examples
        total_correct += correct
        # total_correct+= pred.argmax(dim=-1).eq(y).sum().item()
    return total_correct / total_test_examples
    