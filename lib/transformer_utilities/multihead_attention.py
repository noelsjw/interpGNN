
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import time
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter


#import models.fairseq_util
import lib.transformer_utilities.fairseq_utils as utils
#from fairseq.incremental_decoding_utils import with_incremental_state
from .fairseq_dropout import FairseqDropout



# from .group_linear_layer import GroupLinearLayer
from .relational_memory_volatile import RelationalMemory
#from .relational_memory_lstm import RelationalMemory
# from .relational_memory_regressive import RelationalMemory as RelationalMemoryRegressive
#from fairseq.modules.shared_group_linear_layer import SharedGroupLinearLayer as GroupLinearLayer

class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        nblocks=1,
        top_k_ratio=None,
        use_value_competition=True,
        shared_memory_attention = False,
        use_topk = False,
        topk = 3,
        num_steps = 5,
        mem_slots = 4,
        null_attention = False,
        regressive = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        self.shared_memory_attention = shared_memory_attention

        print('total heads', self.num_heads)
        print('head dim', self.head_dim)

        self.use_topk = use_topk
        self.topk = topk

        print('use topk?' + str(self.use_topk))
        print('topk:'+str(self.topk))

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.onnx_trace = False
        self.tpu = False

        if self.shared_memory_attention:
            print('MEM SLOTS:' + str(mem_slots))
            print('Null attention:' + str(null_attention))
            print('USING SHARED MEMORY ATTENTION +++++++++')
            #self.num_heads = 1
            self.regressive = regressive
            self.relational_memory = RelationalMemory(
                    mem_slots=mem_slots,
                    head_size=self.head_dim , #128
                    input_size=embed_dim,
                    output_size=embed_dim,
                    num_heads=self.num_heads, #1
                    num_blocks=1,
                    forget_bias=1,
                    input_bias=0,
                    gate_style="unit",
                    attention_mlp_layers=1,
                    key_size=16,
                    return_all_outputs=False,
                    use_topk = self.use_topk,
                    topk = self.topk,
                    num_steps = num_steps,
                    null_attention = null_attention
                )
            
            self.memory_size = 32 #self.head_dim * self.num_heads

            self.memory = None


    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            if self.shared_memory_attention:
                nn.init.xavier_uniform_(self.k_proj_memory.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.v_proj_memory.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.q_proj_memory.weight, gain=1 / math.sqrt(2))

        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

            #if self.shared_memory_attention:
            #    nn.init.xavier_uniform_(self.k_proj_memory.weight)
            #    nn.init.xavier_uniform_(self.v_proj_memory.weight)
            #    nn.init.xavier_uniform_(self.q_proj_memory.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        #if self.shared_memory_attention:
        #    nn.init.xavier_uniform_(self.out_proj_memory.weight)
        
        if  self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

        #if self.shared_memory_attention and self.out_proj_memory.bias is not None:
        #    nn.init.constant_(self.out_proj.bias, 0.)
        
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        #if self.shared_memory_attention:
        #    if self.bias_k is not None:
        #        nn.init.xavier_normal_(self.bias_k_memory)
        #    if self.bias_v is not None:
        #        nn.init.xavier_normal_(self.bias_v_memory)


    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        comp = None,
        memory = None,
        plot=None
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True


        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]


        saved_state = None
        t1 = time.time()
        if self.memory is None:
            self.memory = self.relational_memory.initial_state(query.size(1), query.size(0)).to(query.device)
        key = key.transpose(1, 0)
        _,_, self.memory, out_hx_mem_new = self.relational_memory(
                inputs=key,
                memory=self.memory,#.cuda(),
                plot=plot
            )


        return out_hx_mem_new.transpose(0, 1), memory, None

    def init_memory(self, bs, ts = None, device = None):
        if not self.regressive:
            self.memory = self.relational_memory.initial_state(bs).to(device)
        else:
            self.memory = self.relational_memory.initial_state(bs, ts).to(device)


