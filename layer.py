import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np
from dgl.nn import GATConv, GraphConv, SAGEConv, GCN2Conv
from torch.nn.parameter import Parameter
import math

class new_GATConv(GATConv):
    def only_mlp(self, feat):
        h_src = self.feat_drop(feat)
        h = self.fc(h_src)
        return h
    
class new_GraphConv(GraphConv):
    def only_mlp(self, feat):
        feat_src = torch.matmul(feat, self.weight)
        return feat_src
    
class new_SAGEConv(SAGEConv):
    def only_mlp(self, feat):
        h = self.fc_neigh(feat)
        return h
    
class new_GCN2Conv(GCN2Conv):
    def only_mlp(self, feat, feat_0):
        feat = feat * (1 - self.alpha)
        feat_0 = feat_0[: feat.size(0)] * self.alpha
        rst = feat.add_(feat_0)
        rst = torch.addmm(feat, feat, self.weight1, beta=(1 - self.beta), alpha=self.beta)
        return rst
    
class GC2(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GC2, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, in_feat, g , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        g.ndata['h'] = in_feat
        g.ndata['h'] = g.ndata['h'] * g.ndata['d'].unsqueeze(-1)
        g.update_all(fn.copy_u('h', '_'), fn.sum('_', 'h'))
        g.ndata['h'] = g.ndata['h'] * g.ndata['d'].unsqueeze(-1)
        hi = g.ndata.pop('h')     
        
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+in_feat
        return output
    
    def only_mlp(self,in_feat, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = in_feat    
        
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+in_feat
        return output
    
    
class MixHopLayer(nn.Module):
    """ From https://github.com/CUAI/Non-Homophily-Large-Scale. """
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, g, x):
        degs = g.in_degrees().float().clamp(min=1)
        self.norm = torch.pow(degs, -0.5).to(x.device).unsqueeze(1)
        
        xs = [self.lins[0](x)]
        for j in range(1,self.hops+1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            g.ndata['h'] = x_j
            for hop in range(j):
                g.ndata['h'] = g.ndata['h'] * self.norm
                g.update_all(fn.copy_u('h', '_'), fn.sum('_', 'h'))
                g.ndata['h'] = g.ndata['h'] * self.norm
            x_j = g.ndata.pop('h')    
            
            xs += [x_j]
        return torch.cat(xs, dim=1)
    
    def only_mlp(self, x):
        xs = [self.lins[0](x)]
        for j in range(1,self.hops+1):
            x_j = self.lins[j](x)
            xs += [x_j]
        return torch.cat(xs, dim=1)
    
class FALayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']
        