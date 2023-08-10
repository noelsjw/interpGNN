
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import lib.transformer_utilities.fairseq_utils as utils 

from .layer_norm import LayerNorm
from .multihead_attention import MultiheadAttention



import random


from .fairseq_dropout import FairseqDropout
from torch import Tensor

import torch.nn.functional as F


class TransformerEncoderLayerVanilla(nn.Module):

    def __init__(self, args, out_proj = None):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        # self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before
        self.final_layer_norm = LayerNorm(self.embed_dim)

        if out_proj is not None:
            self.final_linear = nn.Linear(args.encoder_embed_dim, out_proj)
        else:
            self.final_linear = None

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=args.self_attention,
            shared_memory_attention = args.shared_memory_attention,
            use_topk = args.use_topk,
            topk = args.topk,
            num_steps = args.num_steps,
            mem_slots = args.mem_slots,
            null_attention = args.null_attention,
            regressive = args.regressive
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None, state = None, memory = None, plot=None):

        residual = x
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        x, memory, _ = self.self_attn(
            query=state if state is not None else x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            memory = memory,
            plot=plot
        ) # memory shape: 2 * 8 * 256
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, memory

