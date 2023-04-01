import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from einops import rearrange
from model.block.graph_frames import Graph
from model.block.mlp_gcn import Mlp_gcn

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.graph = Graph('hm36_gt', 'spatial', pad=1)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False)
        
        self.embedding = nn.Linear(2*args.frames, args.channel)
        self.mlp_gcn = Mlp_gcn(args.layers, args.channel, args.d_hid, args.token_dim, self.A, length=args.n_joints, frames=args.frames)
        self.head = nn.Linear(args.channel, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> b j (c f)').contiguous() # B 17 (2f)

        x = self.embedding(x)       # B 17 512
        x = self.mlp_gcn(x)         # B 17 512
        x = self.head(x)            # B 17 3

        x = rearrange(x, 'b j c -> b 1 j c').contiguous() # B, 1, 17, 3

        return x

