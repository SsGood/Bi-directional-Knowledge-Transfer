import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def training_Gen(data, idx_list, net, net_gen, optimizer, train_part, args, device):
    net.train()
    min_loss = 100.0
    max_acc = 0.0
    counter = 0
    best_epoch = 0
    los = []
    dur = []
    [g, features, labels] = data
    [train, val, test] = idx_list
    gen_loss_list = []
    cls_loss_list = []
    
    for epoch in range(args.epochs):
        if epoch >= 3:
            t0 = time.time()

        net.train()
        if train_part == 'GNN':
            logp_ = net(features)
        elif train_part == 'MLP':
            logp_ = net.only_mlp(features)

        logp = F.log_softmax(logp_, 1)
        cls_loss = F.nll_loss(logp[train], labels[train])
        train_acc = accuracy(logp[train], labels[train])
        loss = cls_loss
        
        if net_gen != None and args.with_Gen_for_gnn:
            net_gen.eval()
            gen_labels = torch.randint(0, labels.max()+1, (args.batch_size,)).to(device)
            gen_result = net_gen(gen_labels)
            gen_h, gen_noise = gen_result['h'], gen_result['noise']
            
            gen_logp_ = net.cls_forward(gen_h)
            gen_logp = F.log_softmax(gen_logp_, 1)
            gen_loss = F.nll_loss(gen_logp, gen_labels)
            loss += args.gen_weight * gen_loss
            gen_loss_list.append(gen_loss.item())
            cls_loss_list.append(cls_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        logp_ = net(features)
        logp = F.log_softmax(logp_, 1)

        test_acc = accuracy(logp[test], labels[test])
        loss_val = F.nll_loss(logp[val], labels[val]).item()
        val_acc = accuracy(logp[val], labels[val])
        
        val_acc_mlp = 0
        test_acc_mlp = 0

        los.append([epoch, loss_val, val_acc, test_acc, val_acc_mlp, test_acc_mlp])

        if max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
            state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
        else:
            counter += 1
            
        if counter >= args.patience:
            break

        if epoch >= 3:
            dur.append(time.time() - t0)
            
#         print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Mlp_Val {:.4f} | Mlp_Test {:.4f} | Time(s) {:.4f}".format(epoch, loss_val, train_acc, val_acc, test_acc, val_acc_mlp, test_acc_mlp, np.mean(dur)))

    if gen_loss_list != [] and cls_loss_list != []:
        print('gen_loss_list_average: %.4f, cls_loss_list_average: %.4f'% (np.mean(gen_loss_list), np.mean(cls_loss_list)))      
    return state, los, dur, best_epoch


def training_mlp_Gen(data, idx_list, net, net_gen, optimizer, scores, training_part, args, device):
    net.train()
    min_loss = 100.0
    max_acc = 0.0
    counter = 0
    best_epoch = 0
    los = []
    dur = []
    [g, features, labels] = data
    [train, val, test] = idx_list
    gen_loss_list = []
    cls_loss_list = []
    for epoch in range(args.epochs):
        if epoch >= 3:
            t0 = time.time()

        net.train()
        if training_part == 'MLP':
            logp_ = net(features)
        elif training_part == 'GNN':
            logp_ = net.only_mlp(features)
        logp = F.log_softmax(logp_, 1)
        train_acc = accuracy(logp[train], labels[train])
        
        cls_loss = F.nll_loss(logp[train], labels[train])

        if scores != None:
            distill_loss = F.kl_div(logp, scores.detach(), reduction='batchmean', log_target=True)
        elif scores == None:
            distill_loss = 0
        loss = cls_loss + distill_loss*args.dis_weight
#         print(f'cls: {cls_loss}   distill: {distill_loss}')  
        if net_gen != None and args.with_Gen_for_mlp:
            net_gen.eval()
            gen_labels = torch.randint(0, labels.max()+1, (args.batch_size,)).to(device)
            gen_result = net_gen(gen_labels)
            gen_h, gen_noise = gen_result['h'], gen_result['noise']
                
            gen_logp_ = net.cls_forward(gen_h)
            gen_logp = F.log_softmax(gen_logp_, 1)
            gen_loss = F.nll_loss(gen_logp, gen_labels)
            loss += args.gen_weight * gen_loss
            gen_loss_list.append(gen_loss.item())
            cls_loss_list.append(cls_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        if training_part == 'MLP':
            logp_ = net(features)
        elif training_part == 'GNN':
            logp_ = net.only_mlp(features)
        logp = F.log_softmax(logp_, 1)

        test_acc = accuracy(logp[test], labels[test])
        loss_val = F.nll_loss(logp[val], labels[val]).item()
        val_acc = accuracy(logp[val], labels[val])
        
        val_acc_mlp = 0
        test_acc_mlp = 0

        los.append([epoch, loss_val, val_acc, test_acc, val_acc_mlp, test_acc_mlp])

        if max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
            state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
        else:
            counter += 1
            
        if counter >= args.patience:
#             print('early stop')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)
        
#         print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Mlp_Val {:.4f} | Mlp_Test {:.4f} | Time(s) {:.4f}".format(epoch, loss_val, train_acc, val_acc, test_acc, val_acc_mlp, test_acc_mlp, np.mean(dur)))
    if gen_loss_list != [] and cls_loss_list != []:
        print('gen_loss_list_average: %.4f, cls_loss_list_average: %.4f'% (np.mean(gen_loss_list), np.mean(cls_loss_list)))
    return state, los, dur, best_epoch




def evaluating_self_iter(data, idx_list, net_gnn, args, device):
    net_gnn.eval()
    
    [g, features, labels] = data
    [train, val, test] = idx_list
    
    logp_ = net_gnn(features)
    logp = F.log_softmax(logp_, 1)
    test_acc = accuracy(logp[test], labels[test])
    val_loss = F.nll_loss(logp[val], labels[val]).item()
    val_acc = accuracy(logp[val], labels[val])
    
    mlp_logp_ = net_gnn.only_mlp(features)
    mlp_logp = F.log_softmax(mlp_logp_, 1)
    mlp_test_acc = accuracy(mlp_logp[test], labels[test])
    mlp_val_loss = F.nll_loss(mlp_logp[val], labels[val]).item()
    mlp_val_acc = accuracy(mlp_logp[val], labels[val])
    
    distill_loss = F.kl_div(logp, mlp_logp, reduction='batchmean', log_target=True)
    
#     label_smooth = smoothness(g, F.one_hot(labels))
#     logp_smmoth = smoothness(g, oh_encoding_logit(logp))
#     MLP_logp_smmoth = smoothness(g, oh_encoding_logit(mlp_logp))
#     print(f'| label_smooth {label_smooth} | logp_smmoth {logp_smmoth} | MLP_logp_smmoth {MLP_logp_smmoth} |')
    
    distill_loss = F.kl_div(logp, mlp_logp, reduction='batchmean', log_target=True)
    print(f'val_acc : {val_acc} test_acc : {test_acc} mlp_val_acc : {mlp_val_acc} mlp_test_acc : {mlp_test_acc}')
    return logp, mlp_logp, [distill_loss, val_acc, test_acc, mlp_val_acc, mlp_test_acc]

def evaluating_final(data, idx_list, net_gnn, args, moment, device):
    net_gnn.eval()
    
    [g, features, labels] = data
    [train, val, test] = idx_list
    
    logp_ = net_gnn(features)
    logp = F.log_softmax(logp_, 1)
    test_acc = accuracy(logp[test], labels[test])
    val_loss = F.nll_loss(logp[val], labels[val]).item()
    val_acc = accuracy(logp[val], labels[val])
    
    mlp_logp_ = net_gnn.only_mlp(features)
    mlp_logp = F.log_softmax(mlp_logp_, 1)
    mlp_test_acc = accuracy(mlp_logp[test], labels[test])
    mlp_val_loss = F.nll_loss(mlp_logp[val], labels[val]).item()
    mlp_val_acc = accuracy(mlp_logp[val], labels[val])
    
    distill_loss = F.kl_div(logp, mlp_logp, reduction='batchmean', log_target=True)
    
#     label_smooth = smoothness(g, F.one_hot(labels))
#     logp_smmoth = smoothness(g, oh_encoding_logit(logp))
#     MLP_logp_smmoth = smoothness(g, oh_encoding_logit(mlp_logp))
#     print(f'| label_smooth {label_smooth} | logp_smmoth {logp_smmoth} | MLP_logp_smmoth {MLP_logp_smmoth} |')
    
    distill_loss = F.kl_div(logp, mlp_logp, reduction='batchmean', log_target=True)
    print(f'{moment}   val_acc : {val_acc} test_acc : {test_acc} mlp_val_acc : {mlp_val_acc} mlp_test_acc : {mlp_test_acc}')
    return logp, mlp_logp, [distill_loss, val_acc, test_acc, mlp_val_acc, mlp_test_acc]

def compare(data, idx_list, net_gnn, state_gnn, state_mlp, args, device):
    [g, features, labels] = data
    [train, val, test] = idx_list
    
    net_gnn.load_state_dict(state_gnn)
    logp_ = net_gnn(features).detach().cpu()
    logp = F.log_softmax(logp_, 1)
    _, indices = torch.max(logp, dim=1)
    correct_gnn = np.array(indices[test] == labels[test].cpu())
    
    net_gnn.load_state_dict(state_mlp)
    logp_ = net_gnn.only_mlp(features).detach().cpu()
    logp = F.log_softmax(logp_, 1)
    _, indices = torch.max(logp, dim=1)
    correct_mlp = np.array(indices[test] == labels[test].cpu())
    
    union_acc = np.logical_or(correct_gnn, correct_mlp).sum()/len(test)
    xor_acc = np.logical_and(correct_gnn, correct_mlp).sum()/len(test)
    gnn_acc = correct_gnn.sum()/len(test)
    mlp_acc = correct_mlp.sum()/len(test)
    return union_acc, xor_acc, gnn_acc, mlp_acc


def compare_degree(data, idx_list, net_gnn, state_gnn, state_mlp, args, device):
    [g, features, labels] = data
    [train, val, test] = idx_list
    degree = g.in_degrees().cpu().numpy()
    ind_list = []
    for i in range(0, 10, 2):
        tmp_ind = list(set(test.cpu().numpy()).intersection(set(np.where((degree>= i) & (degree<=i+1))[0])))
        ind_list.append(tmp_ind)
    tmp_ind = list(set(test.cpu().numpy()).intersection(set(np.where(degree>= 10)[0])))
    ind_list.append(tmp_ind)
    
    
    net_gnn.load_state_dict(state_gnn)
    logp_ = net_gnn(features).detach().cpu()
    logp = F.log_softmax(logp_, 1)
    _, indices = torch.max(logp, dim=1)
    
    gnn_degree_acc = []
    for ind in ind_list:
        tmp_acc = np.array(indices[ind] == labels[ind].cpu()).sum()/len(ind)
        gnn_degree_acc.append(tmp_acc)
        
        
    
    net_gnn.load_state_dict(state_mlp)
    logp_ = net_gnn.only_mlp(features).detach().cpu()
    logp = F.log_softmax(logp_, 1)
    _, indices = torch.max(logp, dim=1)
    
    mlp_degree_acc = []
    for ind in ind_list:
        tmp_acc = np.array(indices[ind] == labels[ind].cpu()).sum()/len(ind)
        mlp_degree_acc.append(tmp_acc)
    
    return gnn_degree_acc, mlp_degree_acc


def loss_sort(los, args):
    los.sort(key=lambda x: -x[2])
    print('final_acc', los[0][-3], 'epoch', los[0][0])
    return los[0][-3], los[0][0]

def train_generator(data, net, state_pos, state_neg, generator, optimizer, args, device):
    generator.train()
    net_pos = copy.deepcopy(net)
    net_neg = copy.deepcopy(net)
    
    min_loss = 100.0
    max_acc = 0.0
    counter = 0
    best_epoch = 0
    los = []
    dur = []
    [g, features, labels] = data
    loss_pos_list = []
    los_neg_list = []

    for epoch in range(args.epochs):
        if epoch >= 3:
            t0 = time.time()
            
        gen_labels = torch.randint(0, labels.max()+1, (args.batch_size,)).to(device)
        gen_result = generator(gen_labels)
        gen_h, gen_noise = gen_result['h'], gen_result['noise']
        net_pos.load_state_dict(state_pos)
        logp_ = net_pos.cls_forward(gen_h)
        logp = F.log_softmax(logp_, 1)
        cls_loss_pos = F.nll_loss(logp, gen_labels)
        diversity_loss = generator.div_loss(gen_noise, gen_h)
        loss = 10 * diversity_loss + cls_loss_pos
        
        if state_neg != None:
            net_neg.load_state_dict(state_neg)
            logp_ = net_neg.cls_forward(gen_h)
            logp = F.log_softmax(logp_, 1)
            cls_loss_neg = F.nll_loss(logp, gen_labels)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if min_loss > loss.item():
            min_loss = loss.item()
            counter = 0
            state = copy.deepcopy(generator.state_dict())
        else:
            counter += 1
            
        if counter >= args.patience:
#             print('early stop')
            break
        if epoch >= 3:
            dur.append(time.time() - t0)
            
    generator.load_state_dict(state)
    
    gen_labels = torch.randint(0, labels.max()+1, (args.batch_size,)).to(device)
    gen_result = generator(gen_labels)
    gen_h, gen_noise = gen_result['h'], gen_result['noise']
    logp_ = net.cls_forward(gen_h)
    logp = F.log_softmax(logp_, 1)
    diversity_loss = generator.div_loss(gen_noise, gen_h)
    cls_loss = F.nll_loss(logp, gen_labels)
#     print(f'diversity_loss {diversity_loss}, cls_loss_gen {cls_loss}')
    
    return generator, min_loss, np.array(dur).mean()


def train_generator_masked(data, net, state_pos, state_neg, generator, optimizer, args, device):
    generator.train()
    net_pos = copy.deepcopy(net)
    net_neg = copy.deepcopy(net)
    
    min_loss = 100.0
    max_acc = 0.0
    counter = 0
    best_epoch = 0
    los = []
    dur = []
    [g, features, labels] = data
    loss_pos_list = []
    los_neg_list = []
    state = copy.deepcopy(generator.state_dict())
    
    for epoch in range(args.epochs):
        if epoch >= 3:
            t0 = time.time()
            
        gen_labels = torch.randint(0, labels.max()+1, (args.batch_size,)).to(device)
        gen_result = generator(gen_labels)
        gen_h, gen_noise = gen_result['h'], gen_result['noise']
        net_pos.load_state_dict(state_pos)
        pos_logp_ = net_pos.cls_forward(gen_h)
        pos_logp = F.log_softmax(pos_logp_, 1)
        _, pos_pred = torch.max(pos_logp, dim=1)
        
        ind = torch.ones(gen_labels.size()[0]).bool()
#         cls_loss_pos = F.nll_loss(pos_logp, gen_labels)
#         diversity_loss = generator.div_loss(gen_noise, gen_h)
#         loss = args.diversity_weight * diversity_loss + cls_loss_pos
        
        if state_neg != None:
            net_neg.load_state_dict(state_neg)
            neg_logp_ = net_neg.cls_forward(gen_h)
            neg_logp = F.log_softmax(neg_logp_, 1)
            _, neg_pred = torch.max(neg_logp, dim=1)
#             cls_loss_neg = F.nll_loss(neg_logp, gen_labels)
            
            pos = gen_labels == pos_pred
            neg = gen_labels == neg_pred
            ind = torch.logical_and(torch.logical_xor(pos, neg), pos)
        cls_loss = F.nll_loss(pos_logp[ind], gen_labels[ind])
        diversity_loss = generator.div_loss(gen_noise, gen_h)
        loss = args.diversity_weight * diversity_loss + cls_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if min_loss > loss.item():
            min_loss = loss.item()
            counter = 0
            state = copy.deepcopy(generator.state_dict())
        else:
            counter += 1
            
        if counter >= args.patience:
#             print('early stop')
            break
        if epoch >= 3:
            dur.append(time.time() - t0)
            
    generator.load_state_dict(state)
    
#     gen_labels = torch.randint(0, labels.max()+1, (args.batch_size,)).to(device)
#     gen_result = generator(gen_labels)
#     gen_h, gen_noise = gen_result['h'], gen_result['noise']
#     logp_ = net.cls_forward(gen_h)
#     logp = F.log_softmax(logp_, 1)
#     diversity_loss = generator.div_loss(gen_noise, gen_h)
#     cls_loss = F.nll_loss(logp, gen_labels)
#     print(f'diversity_loss {diversity_loss}, cls_loss_gen {cls_loss}')
    
    return generator, min_loss, np.array(dur).mean()