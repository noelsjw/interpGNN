import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn.pytorch import GraphConv, GATConv, SGConv, APPNPConv
import types
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn import ModuleList



from lib.transformer_utilities.transformer_layer import TransformerEncoderLayerVanilla
from lib.transformer_utilities.pos_enc import PositionEncoder
from lib.transformer_utilities.GroupLinearLayer import GroupLinearLayer
import math


class GCN_GW(nn.Module):
    def __init__(self, 
                 n_nodes,
                 in_feats,
                 out_feats,
                 n_units=64,
                 num_layers=2,
                 dropout=0.1,
                 activation='relu',
                 fc_units=128,
                 args=None):
        super(GCN_GW, self).__init__()
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        in_feats = in_feats + args.node2vec_emb_dim

        self.conv1 = GCNConv(in_feats, n_units)
        self.conv2 = GCNConv(n_units + in_feats  , out_feats)
        self.memory_layer1 = TransformerEncoderLayerVanilla(args)
        self.layer_norm = torch.nn.LayerNorm(n_units)        
        self.shared_memory_attention = args.shared_memory_attention
        self.shared_memory_percentage = args.shared_memory_percentage
        self.use_topk = args.use_topk
        self.topk = args.topk
        self.gw_ratio = args.gw_ratio
        self.init_memory = args.init_memory



    def forward(self, x, pe, edge_index, plot=None):
        x = torch.cat((x, pe), dim=1)
        residual = x
        x = self.conv1(x, edge_index) # num_nodes * n_unit
        x = x.relu() 
        x = F.dropout(x, p=0.2, training=self.training)
        x_gw = x.unsqueeze(1)
        memory_size = int(self.shared_memory_percentage * x_gw.size(2))
        memory = torch.randn(memory_size, 1, x_gw.size(2)).to(x.device)
        if self.memory_layer1.self_attn.memory is not None:
            # self.memory_layer.self_attn.init_memory(x_gw.size(1), device=x.device)
            self.memory_layer1.self_attn.memory = self.memory_layer1.self_attn.memory.detach()
        if self.init_memory:
            self.memory_layer1.self_attn.init_memory(x_gw.size(1), device=x.device)
        x_gw, memory = self.memory_layer1(x_gw, None, memory=memory, plot=plot)
        x_gw = x_gw.squeeze(1)
        x = self.layer_norm(x)
        x = torch.add(self.gw_ratio*x_gw, x)
        # x = torch.add(x, x_gw * 0.2)
        x = torch.cat((x, residual),dim=1)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)
                

if __name__ == "__main__":
    # x = torch.randn(8, 20, 256).cuda()
    
    model = GCN_GW(1433, 32).to("cuda:0")
    from dgl.data import citation_graph
    from dgl import add_self_loop
    graph_with_features = add_self_loop(citation_graph.load_cora()[0]).to("cuda:0")
    node_features = graph_with_features.ndata['feat'].float().cuda()
    model(graph_with_features, node_features)