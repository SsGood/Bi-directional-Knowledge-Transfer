import gc
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from utils import *
from gnn import *
from training_agent import *
import copy


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)
    dgl.random.seed(seed)
    
class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        

        
def main(arg_str=None):        
# torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    "General"
    parser.add_argument('--dataset', default='new_squirrel')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--feat_drop', type=float, default=0.6, help='Feature dropout rate (1 - keep probability).')
    parser.add_argument('--layer_num', type=int, default=1, help='Number of layers')
    parser.add_argument('--train_ratio', type=float, default=0.025, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.025, help='Ratio of training set')
    parser.add_argument('--patience', type=int, default=50, help='Patience')
    parser.add_argument('--net', type=str, default='FAGCN')
    parser.add_argument('--MLP_train', action='store_true', help='directly train the feature transforma part in GNNs.')
    parser.add_argument('--RPMAX', type=int, default=10, help='seed')
    parser.add_argument('--result_path', type=str, default='results_asyn')
    parser.add_argument('--norm_type', type=str, default='none')
    
    "GAT"
    parser.add_argument('--attn_drop', type=float, default=0.6, help='Attention rate (1 - keep probability).')
    parser.add_argument('--slope', type=float, default=0.2, help='Negative slope of elu')
    
    "FAGCN"
    parser.add_argument('--eps', type=float, default=0.2, help='Fixed scalar or learnable weight.')
    
    "MLP"
    parser.add_argument('--turn', type=str, default='APPNP')
    parser.add_argument('--norm', type=str, default='none')
    
    'GCNII'
    parser.add_argument('--alpha', type=float, default=0.1, help=' ')
    parser.add_argument('--lamda', type=float, default=0.5, help=' ')
    parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
    
    "Iterative Training"
    parser.add_argument('--dis_weight', type=float, default=100, help='loss weight')
    parser.add_argument('--gen_weight', type=float, default=1, help='gen loss weight')
    parser.add_argument('--diversity_weight', type=float, default=10, help='diversity loss weight')
    parser.add_argument('--iter_num', type=int, default=1, help='Odd numbers are suspended by GNN and even numbers by MLP')
    parser.add_argument('--with_distill', type=int, default=1, help='using distillation')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch_size')
    parser.add_argument('--gen_size', type=int, default=1024, help='the num of generated samples')
    parser.add_argument('--with_Gen_for_mlp', action='store_true')
    parser.add_argument('--with_Gen_for_gnn', action='store_true')
    parser.add_argument('--extra_noise', action='store_true')
    parser.add_argument('--embed', action='store_true')
    parser.add_argument('--masked', action='store_true')
    parser.add_argument('--start', type=str, default='GNN')

    args = parser.parse_args()

    device = 'cuda'
    print(args.__dict__)


    best_mlp_list = []
    best_gcn_list = []
    val_gcn_list = []
    
    init_mlp_list = []
    init_gcn_list = []
    init_val_gcn_list = []
    
    logits_dict = {}
    index_dict = {}
    logits_turn_dict = {}

    for seed in range(args.RPMAX):
        set_rng_seed(seed)
        start = time.time()
        g, nclass, features, labels, train, val, test = preprocess_data_3(args.dataset, args.train_ratio, args.val_ratio, seed)
#         print('+++++++++++++++++++++++++++ data loading: ',time.time()-start,'+++++++++++++++++++++++++++')
        features = features.to(device)
        labels = labels.to(device)
        train = train.to(device)
        test = test.to(device)
        val = val.to(device)
        index_temp = test

        g = g.to(device)
        deg = g.in_degrees().cuda().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        
        data = [g, features, labels]
        idx_list = [train, val, test]

        gnn = args.net
        if gnn == 'GCN':
            net_gnn = GCN(graph = g, 
                                            num_layers = args.layer_num, 
                                            in_dim = features.size()[1], 
                                            num_hidden = args.hidden,
                                            num_classes = nclass,
                                            activation = F.relu, 
                                            feat_drop = args.feat_drop).cuda()
        elif gnn == 'MixHop':
            net_gnn = MixHop(graph = g, 
                                            num_layers = args.layer_num, 
                                            in_dim = features.size()[1], 
                                            num_hidden = args.hidden,
                                            num_classes = nclass,
                                            activation = F.relu, 
                                            feat_drop = args.feat_drop).cuda()
            
        elif gnn == 'GCNII':
            net_gnn = GCNII(g = g,
                                            nlayers = args.layer_num,
                                            nfeat = features.size()[1],
                                            nhidden = args.hidden,
                                            nclass = int(labels.max()) + 1,
                                            dropout = args.feat_drop,
                                            lamda = args.lamda, 
                                            alpha = args.alpha,
                                            variant = args.variant).to(device)
        elif gnn == 'GraphSAGE':
            net_gnn = GraphSAGE(graph = g, 
                                                  num_layers = args.layer_num, 
                                                  in_dim = features.size()[1], 
                                                  num_hidden = args.hidden,
                                                  num_classes = nclass, 
                                                  activation = F.relu, 
                                                  feat_drop = args.feat_drop, 
                                                  aggregator_type = 'mean').cuda()
        elif gnn == 'FAGCN':
            net_gnn = FAGCN(g, 
                        features.size()[1], 
                        args.hidden, 
                        nclass, 
                        args.feat_drop, 
                        args.eps, 
                        args.layer_num).cuda()
            
        elif gnn == 'GAT':
            heads = [8 for i in range(args.layer_num-1)]
            heads.extend([1,1])
            net_gnn = GAT(graph = g, 
                      num_layers = args.layer_num, 
                      in_dim = features.size()[1], 
                      num_hidden = args.hidden, 
                      num_classes = nclass, 
                      heads = heads, 
                      activation = F.elu, 
                      feat_drop = args.feat_drop, 
                      attn_drop = args.attn_drop, 
                      negative_slope = args.slope, 
                      residual = False).cuda()
        net_gnn_copy = copy.deepcopy(net_gnn)
        
        if hasattr(net_gnn, 'heads'):
            head = net_gnn.heads[-2]
        elif hasattr(net_gnn, 'hop'):
            head = net_gnn.hop+1
        else:
            head=1

        net_gen_mlp = Generator(hidden_dim = 64, 
                            out_dim = args.hidden*head, 
                            nclasses = nclass, 
                            noise_dim = 32, extra_noise=args.extra_noise, embedding = args.embed).cuda()
        
        net_gen_gnn = Generator(hidden_dim = 64, 
                            out_dim = args.hidden*head, 
                            nclasses = nclass, 
                            noise_dim = 32, extra_noise=args.extra_noise, embedding = args.embed).cuda()
        gen_func = train_generator_masked if args.masked else train_generator
            


        dur = []
        los = []
        loc = []
        initial_state = None
        if args.start == 'GNN':
            init_i = 0
        elif args.start == 'MLP':
            init_i = 1
        
        for i in range(init_i, args.iter_num):

            if i%2 == 0:
                generator = None if i == init_i else net_gen_mlp
                state_mlp = None if i == init_i else state_mlp
                
                print(f'-----{i//2}_GNN_Updating-----')
                optimizer_gnn = torch.optim.Adam(net_gnn.parameters(), 
                                                 lr=args.lr, 
                                                 weight_decay=args.weight_decay)
                
                state_gnn, los_temp, dur_temp, best_epoch = training_Gen(data, 
                                                                 idx_list, 
                                                                 net_gnn,
                                                                 generator,
                                                                 optimizer_gnn, 
                                                                 'GNN', args, device)
                net_gnn.load_state_dict(state_gnn)
                los.extend(los_temp)
                dur.extend(dur_temp)
                logits_gnn, logits_mlp, acc_list = evaluating_self_iter(data, idx_list, net_gnn, args, device)
                
                if i == init_i:
                    initial_state = state_gnn
                    
                if args.with_Gen_for_mlp:
                    optimizer_gen_gnn = torch.optim.Adam(net_gen_gnn.parameters(), 
                                         lr=args.lr, 
                                         weight_decay=args.weight_decay)
                    net_gen_gnn, min_loss, avg_time = gen_func(data, 
                                                                net_gnn_copy,
                                                                state_gnn,
                                                                state_mlp,
                                                                net_gen_gnn, 
                                                                optimizer_gen_gnn, 
                                                                args, device)
                
            elif i%2 == 1:
                generator = None if i == init_i else net_gen_gnn
                state_gnn = None if i == init_i else state_gnn
                logit_gnn = None if i == init_i else logits_gnn
                print(f'-----{i//2}_MLP_Updating-----')
                
                optimizer_mlp = torch.optim.Adam(net_gnn.parameters(), 
                                                 lr=args.lr, 
                                                 weight_decay=args.weight_decay)
                state_mlp, los_temp, dur_temp, best_epoch = training_mlp_Gen(data, 
                                                                     idx_list, 
                                                                     net_gnn,
                                                                     generator,
                                                                     optimizer_mlp, 
                                                                     logit_gnn, 
                                                                     'GNN', args, device)
                net_gnn.load_state_dict(state_mlp)
                los.extend(los_temp)
                dur.extend(dur_temp)
                logits_temp, logits_turn_temp, acc_list = evaluating_self_iter(data, idx_list, net_gnn, args, device)
                if i == init_i:
                    initial_state = state_mlp
                
                if args.with_Gen_for_gnn:
                    optimizer_gen_mlp = torch.optim.Adam(net_gen_mlp.parameters(), 
                                         lr=args.lr, 
                                         weight_decay=args.weight_decay)
                    net_gen_mlp, min_loss, avg_time = gen_func(data, 
                                                                net_gnn_copy,
                                                                state_mlp,
                                                                state_gnn,
                                                                net_gen_mlp, 
                                                                optimizer_gen_mlp, 
                                                                args, device)

        logits_gnn, logits_mlp, acc_list = evaluating_final(data, idx_list, net_gnn, args, 'Final', device)    
        best_gcn_list.append(acc_list[2])
        best_mlp_list.append(acc_list[4])
        val_gcn_list.append(acc_list[1])
        
        net_gnn.load_state_dict(initial_state)
        logits_gnn, logits_mlp, acc_list = evaluating_final(data, idx_list, net_gnn, args, 'Initial', device)
        init_gcn_list.append(acc_list[2])
        init_mlp_list.append(acc_list[4])
        init_val_gcn_list.append(acc_list[1])
        
    print(' \n')
    print('gcn_acc',np.mean(best_gcn_list), 'gcn_acc_std',np.std(best_gcn_list),
          'mlp_acc', np.mean(best_mlp_list), 'mlp_acc_std',np.std(best_mlp_list))
    
    print('init_gcn_acc',np.mean(init_gcn_list), 'init_gcn_acc_std',np.std(init_gcn_list),
          'init_mlp_acc', np.mean(init_mlp_list), 'init_mlp_acc_std',np.std(init_mlp_list))
    
    return np.mean(best_gcn_list), np.mean(best_mlp_list), np.std(best_gcn_list), np.std(best_mlp_list), np.mean(init_gcn_list), np.mean(init_mlp_list), np.std(init_gcn_list), np.std(init_mlp_list)
    
if __name__ == '__main__':
    with RedirectStdStreams(stdout=sys.stderr):
        tst_acc_mean, mlp_mean, tst_acc_std, mlp_std, init_tst_acc_mean, init_mlp_mean, init_tst_acc_std, init_mlp_std = main()
    print('(%.4f,%.4f, %.4f,%.4f, %.4f,%.4f)' % (tst_acc_mean, tst_acc_std, 
                                                 mlp_mean, mlp_std,
                                                tst_acc_mean - init_tst_acc_mean, 
                                                 mlp_mean - init_mlp_mean, ))
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
