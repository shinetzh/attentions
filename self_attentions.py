import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


#### valina self attention
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        qk = einsum("bnd, bmd -> bnm", q, k)
        qk_softmax = self.softmax(qk / (self.hidden_dim ** (1/2)))
        out = einsum("bnm, bmd ->bnd", qk_softmax, v)
        return out


### fast self attention
class SelfAttention2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention2, self).__init__()
        self.hidden_dim = hidden_dim
        self.qkv_forward = nn.Linear(input_dim, hidden_dim*3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        qkv = self.qkv_forward(x)
        q, k, v = torch.chunk(qkv, 3, -2)
        qk_softmax = self.softmax(einsum("bmd, bnd -> bmn", q, k) / self.hidden_dim ** (1/2))
        out = einsum("bmn, bnd -> bmd", qk_softmax, v)
        return out

#### mix for self attention and cross attention
class Attention3(nn.Module):
    def __init__(self, q_input_dim, hidden_dim, kv_input_dim=None):
        super(Attention3, self).__init__()
        if kv_input_dim is None:
            kv_input_dim = q_input_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(q_input_dim, hidden_dim, bias=False)
        self.kv_forward = nn.Linear(kv_input_dim, hidden_dim*2, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, cond=None):
        if cond is None:
            cond = x
        q = self.query(x) # bmd
        kv = self.kv_forward(cond)
        k, v = torch.chunk(kv, chunks=2, dim=-2) # bnd, bnd
        qk = einsum("bmd, bnd -> bmn", q, k)
        qk_softmax = self.softmax(qk / (self.hidden_dim ** (1 / 2)))
        out = einsum("bmn, bnd -> bmd", qk_softmax, v)
        return out

### causel Attention
class CauselAttention(nn.Module):
    def __init__(self, q_input_dim, hidden_dim, kv_input_dim):
        super(CauselAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(q_input_dim, hidden_dim, bias=False)
        self.kv_forward = nn.Linear(kv_input_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, cond=None, mask=None):
        # x: [b, m, d]
        # cond: [b, n, c]
        # mask: [b, m, m]
        q = self.query(x)
        k, v = torch.chunk(self.kv_forward(cond), chunks=2, dim=-2)
        qk = einsum("bmd, bnd -> bmn", q, k)
        qk_softmax = self.softmax(qk / (self.hidden_dim ** (1/2)))

        if mask is not None:
            if mask.dtype == torch.bool:
                mask[~mask] = float("-inf")
            qk_softmax = qk_softmax + mask

        out = einsum("bmn, bnd -> bmd", qk_softmax, v)
        return out
