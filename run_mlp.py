import subprocess
import numpy as np
import argparse
import gc
import torch
import yaml

cmds = []
cmds.append('python compare_gnn_mlp.py --net GCN --RPMAX 10 --dataset cora --feat_drop 0.8 --hidden 128 --layer_num 2 --lr 0.05 --train_ratio 0.025 --val_ratio 0.025 --weight_decay 0.001 --iter_num 2')
cmds.append('python compare_gnn_mlp.py --net GCN --RPMAX 10 --dataset citeseer --feat_drop 0.9 --hidden 128 --layer_num 2 --lr 0.001 --train_ratio 0.025 --val_ratio 0.025 --weight_decay 0 --iter_num 2')
cmds.append('python compare_gnn_mlp.py --net GCN --RPMAX 10 --dataset pubmed --feat_drop 0.6 --hidden 32 --layer_num 2 --lr 0.05 --train_ratio 0.025 --val_ratio 0.025 --weight_decay 0.0025 --iter_num 2')
cmds.append('python compare_gnn_mlp.py --net GCN --RPMAX 10 --dataset photo --feat_drop 0.7 --hidden 128 --layer_num 2 --lr 0.05 --train_ratio 0.025 --val_ratio 0.025 --weight_decay 0.0001 --iter_num 2')
cmds.append('python compare_gnn_mlp.py --net GCN --RPMAX 10 --dataset computers --feat_drop 0.8 --hidden 128 --layer_num 2 --lr 0.05 --train_ratio 0.025 --val_ratio 0.025 --weight_decay 0.0001 --iter_num 2')

for cmd in cmds:
    output = subprocess.check_output(cmd, shell=True)
    (union_acc, union_acc_std, xor_acc, xor_acc_std, gnn_acc, gnn_acc_std, mlp_acc, mlp_acc_std) = eval(output)
    filename = f'results/mlp_results.csv'
    with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(f"{gnn_acc} +- {gnn_acc_std}," + f"{mlp_acc} +- {mlp_acc_std}\n")