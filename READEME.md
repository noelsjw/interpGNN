# Requirements
Please make sure to install the following dependencies:

networx == 2.6.3
ogb == 1.3.5
pytorch_geometric == 2.0.1
Pytorch == 1.9.0
dgl == 0.7.1

# Useage
## Download datasets

- **CoraFull, Wiki-CS, Amazon-Computer, Flickr, Reddit**: Implemented with PyTorch Geometric (PyG).
- **Amazon2M**: Implemented with [Open Graph Benchmark (OGB)](https://ogb.stanford.edu).
- **Aminer-CS**: Please download from the [GRAND-plus repository](https://github.com/wzfhaha/GRAND-plus).

## Pre-train node2vec positional embedding

We have uploaded all pre-trained positional embeddings in the `dataset/'name_of_dataset/node2vec'` folder except for Amazon, 128-dim PE for Aminer-CS, and 128-dim PE for Reddit (due to exceeding the size limit of GitHub). However, you may pretrain the positional embeddings yourself with different hyper-parameters by modifying the configuration in pos_emb.conf. Please place the generated .pt file in above folder. 

Optionally, run: 
```bash
python generae_pos_emb.py
```


## Train the InterpGNN
The running logs for each dataset are available and will be stored in the `experiments_rebuttal/'name_of_dataset'/'time_of_experiments'/run.log'`, where all the hyper-parameters are listed. Before training the model, you may change the hyperparameters in `large_graph.conf` to reproduce the results from the paper.

To start training, run:

```bash
python gnn_gw_large.py
```
