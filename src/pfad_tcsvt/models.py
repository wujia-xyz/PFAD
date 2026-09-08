"""Set Transformer support classifier with the same PFAD input information.

Masked MAB follows juho-lee/set_transformer/modules.py, commit
73432c640ac78140496d6738416c54d32c686d65 (MIT; license in third_party).
The paper's SAB-SAB-PMA construction is used because there are at most five
source records. The adaptation adds padding masks and concatenates the same
19 query-context features to the output classifier. It is a record-evidence
Set Transformer comparator, not a reproduction of raw point-cloud completion.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class MaskedMAB(nn.Module):
    def __init__(self, dim_q: int, dim_k: int, dim_v: int, heads: int) -> None:
        super().__init__()
        self.dim_V = dim_v
        self.num_heads = heads
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        self.ln0 = nn.LayerNorm(dim_v)
        self.ln1 = nn.LayerNorm(dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)

    def forward(self, query, key, key_mask):
        q = self.fc_q(query)
        k, v = self.fc_k(key), self.fc_v(key)
        width = self.dim_V // self.num_heads
        qh = torch.cat(q.split(width, 2), 0)
        kh = torch.cat(k.split(width, 2), 0)
        vh = torch.cat(v.split(width, 2), 0)
        logits = qh.bmm(kh.transpose(1, 2)) / math.sqrt(self.dim_V)
        logits = logits.masked_fill(~key_mask.bool().repeat(self.num_heads, 1)[:, None, :], -torch.inf)
        attention = torch.softmax(logits, 2)
        output = torch.cat((qh + attention.bmm(vh)).split(q.size(0), 0), 2)
        output = self.ln0(output)
        return self.ln1(output + F.relu(self.fc_o(output)))


class SetTransformerSelector(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.sab0 = MaskedMAB(8, 8, hidden, heads)
        self.sab1 = MaskedMAB(hidden, hidden, hidden, heads)
        self.seed = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.xavier_uniform_(self.seed)
        self.pma = MaskedMAB(hidden, hidden, hidden, heads)
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 19, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, source, mask, query_context):
        source = self.sab0(source, source, mask)
        source = self.sab1(source, source, mask)
        pooled = self.pma(self.seed.expand(source.size(0), -1, -1), source, mask)[:, 0]
        return self.fusion(torch.cat([pooled, query_context], dim=1))
