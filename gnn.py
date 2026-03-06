import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np
from layer import *
from utils import *

class FAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return h#F.log_softmax(h, 1)
    
    def only_mlp(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.t2(h)
        return h#F.log_softmax(logits, 1)
    
    def cls_forward(self, h):
        h = self.t2(h)
        return h
    
    
class GAT(nn.Module):
    def __init__(self,
                 graph,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = graph
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.heads = heads
        # input projection (no residual)
        self.gat_layers.append(new_GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(new_GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(new_GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
            h = F.relu(h)
        # output projection
        h = self.gat_layers[-1](self.g, h).mean(1)
        return h#F.log_softmax(logits, 1)
    
    def only_mlp(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l].only_mlp(h)
            h = self.activation(h)
        # output projection
        h = self.gat_layers[-1].only_mlp(h)
        return h#F.log_softmax(logits, 1)
    
    def cls_forward(self, h):
        h = self.gat_layers[-1].only_mlp(h)
        return h
    
    
class SAGE(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(new_SAGEConv(input_dim, output_dim, "gcn"))
        else:
            self.layers.append(new_SAGEConv(input_dim, hidden_dim, "gcn"))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(new_SAGEConv(hidden_dim, hidden_dim, "gcn"))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(new_SAGEConv(hidden_dim, output_dim, "gcn"))

    def forward(self, blocks, feats):
        h = feats
        h_list = []
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h
    
    def only_mlp(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer.only_mlp(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h
    
    def cls_forward(self, feats):
        h = feats
        h = self.layers[-1].only_mlp(h)
        return h

    def inference(self, g, x, args, mlp, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.NodeDataLoader(
                g.to(device),
                torch.arange(g.number_of_nodes()).to(device),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0)
        feats = x
        device = feats.device
        for l, layer in enumerate(self.layers):
            y = torch.zeros(
                feats.shape[0],
                self.hidden_dim if l != self.num_layers - 1 else self.output_dim,
            ).to(device)
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)

                h = feats[input_nodes]
                h_dst = h[: block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != self.num_layers - 1:
                    if self.norm_type != "none":
                        h = self.norms[l](h)
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            feats = y
        return y

class GCN(nn.Module):
    def __init__(self,
                 graph,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 feat_drop):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.g = graph
        # input layer
        self.layers.append(new_GraphConv(in_dim, num_hidden, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(new_GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(new_GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=feat_drop)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h#F.log_softmax(h, 1)
    
    def only_mlp(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer.only_mlp(h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
        return h#F.log_softmax(h, 1)
    
    def cls_forward(self, h):
        layer = self.layers[-1]
        h = layer.only_mlp(h)
        return h
    
class MixHop(nn.Module):
    def __init__(self,
                 graph,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 feat_drop,
                 hop = 2):
        super(MixHop, self).__init__()
        self.layers = nn.ModuleList()
        self.g = graph
        self.hop = hop
        # input layer
        self.layers.append(MixHopLayer(in_dim, num_hidden, hops=self.hop))
        for i in range(num_layers - 1):
            self.layers.append(MixHopLayer(num_hidden*(self.hop+1), num_hidden))
        self.layers.append(MixHopLayer(num_hidden*(self.hop+1), num_classes))
        self.dropout = nn.Dropout(p=feat_drop)
        self.activation = activation
        
    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
        return h
    
    def cls_forward(self, h):
        layer = self.layers[-1]
        h = layer.only_mlp(h)
        return h
    
    def only_mlp(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer.only_mlp(h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
        return h
    
class GCNII(nn.Module):
    def __init__(self, g, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GC2(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.g = g

    def forward(self, x):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, self.g, _layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner
    
    def only_mlp(self, x):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con.only_mlp(layer_inner, _layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner
    
    def cls_forward(self, x):
        layer_inner = self.fcs[-1](x)
        return layer_inner

        
class GraphSAGE(nn.Module):
    def __init__(self,
                 graph,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 feat_drop,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.g = graph
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(feat_drop)
        self.activation = activation

        # input layer
        self.layers.append(new_SAGEConv(in_dim, num_hidden, aggregator_type))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(new_SAGEConv(num_hidden, num_hidden, aggregator_type))
        # output layer
        self.layers.append(new_SAGEConv(num_hidden, num_classes, aggregator_type)) # activation None

    def forward(self, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(self.g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h#F.log_softmax(h, 1)
    
    def only_mlp(self, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer.only_mlp(h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h#F.log_softmax(h, 1)
    
    def cls_forward(self, h):
        layer = self.layers[-1]
        h = layer.only_mlp(h)
        return h
    
    



class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        feat_drop,
        norm_type="batch",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(feat_drop)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, num_classes))
        else:
            self.layers.append(nn.Linear(in_dim, num_hidden))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(num_hidden))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(num_hidden))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(num_hidden, num_hidden))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(num_hidden))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(num_hidden))

            self.layers.append(nn.Linear(num_hidden, num_classes))

    def forward(self, feats):
        h = feats
        self.h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            
            if l != self.num_layers - 1:
                #h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                self.h_list.append(h)
                h = self.dropout(h)
        return h#F.log_softmax(h, 1)
    
    def cls_forward(self, h):
        layer = self.layers[-1]
        h = layer[h]
        return h
    
class APPNP(nn.Module):
    def __init__(self,
                 graph,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 feat_drop,
                 K=10, alpha=0.1, ppr_dropout=0):
        super(APPNP, self).__init__()
        self.feat_transform = MLP(num_layers,
                                  in_dim,
                                  num_hidden,
                                  num_classes,
                                  feat_drop,
                                  norm_type="batch")
        self.propagate = APPNPConv(K, alpha, ppr_dropout)
        self.g = g
    
    def forward(self, features):
        h = features
        h = self.feat_transform(h)
        h = self.propagate(self.g, h)
        return h
    
    def only_mlp(self, features):
        h = features
        h = self.feat_transform(h)
        return h
    
    

    
    
class Generator(nn.Module):
    def __init__(self, hidden_dim, out_dim, nclasses, noise_dim, num_layers=1, embedding=False, extra_noise=False):
        super(Generator, self).__init__()
        self.num_layers = num_layers
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_class = nclasses
        self.noise_dim = noise_dim
        in_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.div_loss = DiversityLoss(metric='l1')
        self.extra_noise = extra_noise
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))
        self.out_layer = nn.Linear(self.hidden_dim, self.out_dim)
        
    def forward(self, labels, verbose=True, device='cuda'):
        result = {}
        noise = torch.rand((labels.size()[0], self.noise_dim)).to(device)
        if verbose == True:
            result['noise'] = noise
        if self.embedding:
            y_input = self.embedding_layer(labels)
        else:
            y_input = F.one_hot(labels, num_classes = self.n_class)
        z = torch.cat((noise, y_input), dim=1)
        for (layer, norm) in zip(self.layers, self.norms):
            h = layer(z)
            h = F.relu(norm(h))
        h = self.out_layer(h)

        if self.extra_noise == True:
            h += torch.randn((labels.size()[0], self.out_dim)).to(device)

        result['h'] = h

        return result
