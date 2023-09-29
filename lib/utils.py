from typing import *
import os
import torch
import dgl
import random
import numpy as np
import networkx as nx
import json

import torch.nn.functional as F
from lib.utils_large import load_aminer

# from torch_geometric.datasets import Planetoid,  Reddit, Reddit2
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.datasets import Planetoid, Amazon, WikiCS, Coauthor,CitationFull, Flickr, Reddit, Reddit2,  Actor
from torch_geometric.transforms import NormalizeFeatures


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def generate_mask_random(
    n, 
    train_num=500,
    valid_num=500,
    test_num=1000):
    """_summary_

    Args:
        node_importance (_type_): _description_
        node_subgroup (_type_): _description_
        node_num_each (int, optional): _description_. Defaults to 20.
        train_num (int, optional): _description_. Defaults to 500.
        valid_num (int, optional): _description_. Defaults to 500.
        test_num (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """

    perm = torch.as_tensor(np.random.permutation(n))

    train_idx = perm[:train_num]
    valid_idx = perm[train_num:train_num + valid_num]
    test_idx = perm[train_num + valid_num:]
    
    train_mask = torch.zeros(n)
    valid_mask = torch.zeros(n)
    test_mask = torch.zeros(n)
    train_mask[train_idx] = 1
    valid_mask[valid_idx] = 1
    test_mask[test_idx] = 1
    train_mask = train_mask.bool()
    valid_mask = valid_mask.bool()
    test_mask = test_mask.bool()
    return train_mask, valid_mask, test_mask




def correction_helper(model, data, node_features, mask, node_labels):
    model.eval()
    with torch.no_grad():
        logits = model(data.x,None ,data.edge_index)
        logits = logits[mask] # 2708 * 7 
        _, indices = torch.max(logits, dim=1)
        node_labels_tmp = node_labels[mask]
        correct = torch.sum(indices == node_labels_tmp)
    acc = correct.item() * 1.0 / len(node_labels_tmp)
    return acc, node_labels_tmp, indices


def evaluate(model, node_labels, mask, data, node_features, groups=None):
   
    acc, node_labels_new, indices = correction_helper(model, data,
                                                      node_features, mask,
                                                      node_labels)
    if groups is None:
        # evaluate on label group
        tpr_list = []
        fpr_list = []
        for label in range(int(torch.max(node_labels_new)) + 1):
            if float(torch.sum(node_labels_new == label)) != 0:
                tpr_list.append(
                    float(
                        torch.sum((indices == label)
                               & (node_labels_new == label))) /
                    float(torch.sum(node_labels_new == label)))
            else:
                tpr_list.append(1)
            if float(torch.sum(indices == label)) != 0:
                fpr_list.append(
                    float(
                        torch.sum((indices == label)
                               & (node_labels_new != label))) /
                    float(torch.sum(indices == label)))
            else:
                fpr_list.append(0)
        return acc, tpr_list, fpr_list
    else:
        # evaluate on customized group
        tpr = []
        model.eval()
        with torch.no_grad():
            logits = model(data.x, None, data.edge_index)
            _, indices = torch.max(logits, dim=1)
        for group in groups:
            mask_group = torch.zeros(len(node_labels))
            mask_group[group] = 1
            mask_group = mask_group.bool()
            new_mask = (mask & mask_group)
            if sum(new_mask) == 0:
                tpr.append(1)
                continue
            indices_g = indices[new_mask]
            node_labels_tmp = node_labels[new_mask]
            correct = torch.sum(indices_g == node_labels_tmp)
            acc_group = correct.item() * 1.0 / len(node_labels_tmp)
            tpr.append(acc_group)
            # logger.info(len(node_labels_tmp))
        return acc, tpr, None


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


    best_valid_acc = 0
    best_test_acc = 0
    best_epoch = 0
    tpr = []
    fpr = []
    model = model_class(in_feats=n_features,
                        out_feats=n_labels,
                        num_layers=num_layers,
                        n_units=args.gnn_hidden_dim,
                        dropout=0.5,
                        args=args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
        print(torch.norm(parameters, p=2))
    
    data = data.to(device)
    node_features = node_features.to(device)
    node_labels = node_labels.to(device)
    # acc_best = 0
    for epoch in range(epoch_num):
        model.train()
        # logits = model(graph, node_features)
        out = model(data.x, None,  data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc, _, _ = evaluate(model,
                             node_labels,
                             valid_mask,
                             data,
                             node_features,
                             groups=groups)
        if (epoch + 1) % 100 == 0:
            logger.info("epoch:{}/{} loss: {}".format(epoch, epoch_num, loss.item()))
            logger.info("valid acc: {}".format(acc))
        if best_valid_acc < acc:
            best_valid_acc = acc
            acc_test, tpr_tmp, fpr_tmp = evaluate(model,
                                                  node_labels,
                                                  test_mask,
                                                  data,
                                                  node_features,
                                                  groups=groups)
            best_epoch = epoch
            tpr = tpr_tmp
            fpr = fpr_tmp
            best_test_acc = acc_test
    
            # logger.info("test acc: {}".format(acc_test)) 
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
        print(torch.norm(parameters, p=2))
    logger.info('best_test_acc: {}'.format(best_test_acc))
    logger.info('best epoch: {}'.format(best_epoch))
    logger.info(tpr)
    logger.info(fpr)
    return best_test_acc, tpr, fpr




