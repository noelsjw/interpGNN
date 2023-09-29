import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .pos_enc import PositionEncoder
from .GroupLinearLayer import GroupLinearLayer

# this class largely follows the official sonnet implementation
# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
class RepeatLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_steps):
        super().__init__()
        self.pe = PositionEncoder(in_dim)
        self.num_steps = num_steps
        self.w = nn.Parameter(torch.randn(in_dim))
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        w = self.w.unsqueeze(0).repeat(self.num_steps, 1)
        w = self.w.unsqueeze(0).repeat(x.size(0), 1, 1)
        #w = self.pe(w)

        x = torch.relu(w * x)

        x = torch.mean(x, dim = 1)


        x = self.linear(x)

        return x



def count_parameters(name, model):
    k = 0
    for p in model.parameters():
        k += p.numel()

    print(name, end = ':')
    print(k)


class RelationalMemory(nn.Module):

    def __init__(self, mem_slots, head_size, input_size, output_size, num_heads=1, num_blocks=1, forget_bias=1., input_bias=0.,
                 gate_style='unit', attention_mlp_layers=2, key_size=None, return_all_outputs=False, use_topk = False, topk = 3, num_steps = 5,
                 null_attention = False):
        super(RelationalMemory, self).__init__()

        ########## generic parameters for RMC ##########
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.mem_size = self.head_size * self.num_heads 
        print(self.mem_size)
        self.use_topk = use_topk
        self.topk = topk

        # a new fixed params needed for pytorch port of RMC
        # +1 is the concatenated input per time step : we do self-attention with the concatenated memory & input
        # so if the mem_slots = 1, this value is 2
        self.mem_slots_plus_input = self.mem_slots + 1

        if num_blocks < 1:
            raise ValueError('num_blocks must be >=1. Got: {}.'.format(num_blocks))
        self.num_blocks = num_blocks

        print("Using gate style", gate_style)
        if gate_style not in ['unit', 'memory', None]:
            raise ValueError(
                'gate_style must be one of [\'unit\', \'memory\', None]. got: '
                '{}.'.format(gate_style))
        self.gate_style = gate_style

        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(
                attention_mlp_layers))
        self.attention_mlp_layers = attention_mlp_layers

        self.key_size = key_size if key_size else self.head_size
        self.attn_log = None

        ########## parameters for multihead attention ##########
        # value_size is same as head_size
        self.value_size = self.head_size
        # total size for query-key-value
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_qkv_size = self.qkv_size * self.num_heads  # denoted as F

        self.query_proj = nn.Linear(self.mem_size, self.key_size * self.num_heads)
        count_parameters("query", self.query_proj)
        self.key_proj = nn.Linear(self.mem_size, self.key_size * self.num_heads)
        count_parameters("key", self.key_proj)
        self.value_proj = nn.Linear(self.mem_size, self.value_size * self.num_heads)
        count_parameters("value", self.value_proj)


        # used for attend_over_memory function
        self.attention_mlp = nn.ModuleList([nn.Linear(self.mem_size, self.mem_size)] * self.attention_mlp_layers)
        count_parameters("attention_mlp", self.attention_mlp[0])
        self.attended_memory_layernorm = nn.LayerNorm( self.mem_size)
        count_parameters("layernorm1", self.attended_memory_layernorm)
        self.attended_memory_layernorm2 = nn.LayerNorm(self.mem_size)
        count_parameters("layernorm2", self.attended_memory_layernorm2)

        ########## parameters for initial embedded input projection ##########
        self.input_size = input_size
        self.input_projector = nn.Linear(self.input_size, self.mem_size)
        count_parameters("input_projector", self.input_projector)

        #self.output_projector = nn.Linear(self.output_size, self.input_size)

        ########## parameters for gating ##########
        self.num_gates = 2 * self.calculate_gate_size()
        print('mem_size:'+str(self.mem_size))
        
        if gate_style in ['unit', 'memory']:
            self.input_gate_projector = RepeatLinear(self.mem_size, self.num_gates, num_steps)
            count_parameters("input_gate_projector", self.input_gate_projector)
            self.memory_gate_projector = GroupLinearLayer(self.mem_size, self.num_gates, self.mem_slots)
            #self.memory_gate_projector = nn.Linear(self.mem_size, self.num_gates)

            #(self.mem_size, self.num_gates, self.mem_slots)
            count_parameters("memory_gate_projector", self.memory_gate_projector)
        
        # trainable scalar gate bias tensors
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(input_bias, dtype=torch.float32))

        ########## number of outputs returned #####
        self.return_all_outputs = return_all_outputs

        self.null_attention = null_attention

        print("relational volatie!!!") 


    def repackage_hidden(self, h):

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def initial_state(self, batch_size, trainable=False):

        if True:
            init_state = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])

            # pad the matrix with zeros
            if self.mem_size > self.mem_slots:
                difference = self.mem_size - self.mem_slots
                pad = torch.zeros((batch_size, self.mem_slots, difference))
                init_state = torch.cat([init_state, pad], -1)

            # truncation. take the first 'self.mem_size' components
            elif self.mem_size < self.mem_slots:
                init_state = init_state[:, :, :self.mem_size]

            return init_state
        else:
            init_state = torch.randn(batch_size, self.mem_slots, self.mem_size)
            return init_state
    def multihead_attention(self, input, memory, use_topk_ = True, store_log = True, plot=None):

        q = self.query_proj(memory)
        k = self.key_proj(input)
        v = self.value_proj(input)

        q = q.reshape(q.size(0), q.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(2, 3))

        scores = torch.softmax(scores, dim = -1)

        
            


        scores_cpu = scores[0][0].cpu().detach()

        

        if not self.null_attention:
            if self.use_topk and use_topk_:
                topk = torch.topk(scores, dim = -1, k = self.topk)
                mask = torch.zeros(scores.size()).to(scores.device)
                mask.scatter_(3, topk.indices, 1)
                scores = scores * mask
        else:
            memory_flat = memory.reshape(memory.size(0), -1).unsqueeze(1)
            memory_flat = memory_flat.repeat(1, input.shape[1], 1)

            N = torch.cat((input, memory_flat), dim = 2)
            N = self.competition_mlp(N)

            N = torch.nn.functional.gumbel_softmax(N, dim = 2, hard = True, tau = 0.5)

            N = N[:, :, 0]

            scores = scores * N.unsqueeze(1).unsqueeze(1)


        output = torch.matmul(scores, v)

      
        # [B, H, N, V] => [B, N, H, V] => [B, N, H*V]
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))

        return new_memory


    @property
    def state_size(self):
        return [self.mem_slots, self.mem_size]

    @property
    def output_size(self):
        return self.mem_slots * self.mem_size

    def print_log(self):
        print(self.attn_log)

    def calculate_gate_size(self):
        """
        Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self.gate_style == 'unit':
            return self.mem_size
        elif self.gate_style == 'memory':
            return 1
        else:  # self.gate_style == None
            return 0

    def create_gates(self, inputs, memory):

        memory = torch.tanh(memory)

        if len(inputs.shape) == 3:

            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(dim=1)
            gate_memory = self.memory_gate_projector(memory)
        else:
            raise ValueError("input shape of create_gate function is 2, expects 3")


        gates = gate_memory + gate_inputs
        #self.attn_log = gates[0]
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]

        # to be used for equation 7
        self.attn_log = torch.zeros(input_gate.shape[1], input_gate.shape[2], 2)
        self.attn_log[:, :, 0] = input_gate[0].cpu()

        input_gate = torch.sigmoid(input_gate+self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def attend_over_memory(self, inputs, memory, plot=None):
        """
        Perform multiheaded attention over `memory`.
            Args:
              memory: Current relational memory.
            Returns:
              The attended-over memory.
        """
        for _ in range(self.num_blocks):
            attended_memory = self.multihead_attention(inputs, memory, plot=plot)

            memory = self.attended_memory_layernorm(memory + attended_memory)

            attention_mlp = memory
            for i, l in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            memory = self.attended_memory_layernorm2(memory + attention_mlp)

        return memory

    def forward_step(self, inputs, memory, treat_input_as_matrix=False, plot=None):
        """
        Forward step of the relational memory core.
        Args:
          inputs: Tensor input.
          memory: Memory output from the previous time step.
          treat_input_as_matrix: Optional, whether to treat `input` as a sequence
            of matrices. Default to False, in which case the input is flattened
            into a vector.
        Returns:
          output: This time step's output.
          next_memory: The next version of memory to use.
        """

        if treat_input_as_matrix:
            # keep (Batch, Seq, ...) dim (0, 1), flatten starting from dim 2
            inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
            # apply linear layer for dim 2
            inputs_reshape = self.input_projector(inputs)
        else:
            # keep (Batch, ...) dim (0), flatten starting from dim 1
            inputs = inputs.view(inputs.shape[0], -1)
            # apply linear layer for dim 1
            inputs = self.input_projector(inputs)
            # unsqueeze the time step to dim 1
            inputs_reshape = inputs.unsqueeze(dim=1)

        #memory_plus_input = torch.cat([memory, inputs_reshape], dim=1)
        #print(memory_plus_input.size())
        next_memory = self.attend_over_memory(inputs_reshape, memory, plot=plot)

        # cut out the concatenated input vectors from the original memory slots
        #n = inputs_reshape.shape[1]
        #next_memory = next_memory[:, :-n, :]

        if self.gate_style == 'unit' or self.gate_style == 'memory':
            # these gates are sigmoid-applied ones for equation 7
            input_gate, forget_gate = self.create_gates(inputs_reshape, memory)
            # equation 7 calculation
            next_memory = input_gate * torch.tanh(next_memory)
            next_memory += forget_gate * memory
            self.attn_log[:, :, 1] = input_gate[0].cpu()


        output = next_memory.reshape(next_memory.shape[0], -1)
        # if plot is not None:
        #     plot = "Attend_X"
        hx = self.multihead_attention(next_memory, inputs_reshape, use_topk_ = False, store_log = False,plot=plot)
        return output, next_memory, hx

    def forward(self, inputs, memory, parallel = True, plot=None):

        logits = []
        if not parallel:
            for idx_step in range(inputs.shape[1]):
                logit, memory = self.forward_step(inputs[:, idx_step], memory, plot=plot)
                logits.append(logit)
            logits = torch.cat(logits)
        else:
            logits, memory, hx = self.forward_step(inputs, memory, treat_input_as_matrix = True, plot=plot)
        
        memory_out = None #self.output_projector(memory.view(memory.shape[0], -1))
        if self.return_all_outputs:
            return logits, memory_out, memory, hx
        else:
            return logits, memory_out, memory, hx


