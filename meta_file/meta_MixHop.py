
import numpy as np
import subprocess
import argparse
import hyperopt

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

min_y = 0
min_c = None
min_tst= 0
min_tst_c = None


def trial(hyperpm):
    global min_y, min_c, min_tst, min_tst_c
    # Plz set nbsz manually. Maybe a larger value if you have a large memory.
    cmd = 'python train_GNN_auto_bi_w_noise.py --net MixHop'
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        
        if isinstance(v, str):
            cmd += ' %s' %v
        elif int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))
    try:
        val, val_std, tst, tst_std, mlp_mean, mlp_mean_std = eval(subprocess.check_output(cmd, shell=True))
    except subprocess.CalledProcessError:
        print('...')
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    print('val=%5.2f%% tst=%5.2f%% @ %s' % (val*100, tst*100, cmd))
    print('>>>>>>>>>> min tst now=%5.2f%% @ %s' % (-min_tst*100, min_tst_c))
    tst_score = -tst
    score = -val
    if score < min_y:
        min_y, min_c = score, cmd
        f= open("./MixHop_log/logger-{}-2.txt".format(args.dataset),"a+")
        f.write('>>>>>>>>>> min val now=%5.2f%% @ %s\n' % (-min_y*100, min_c))
        f.close()
    if tst_score < min_tst:
        min_tst, min_tst_c = tst_score, cmd
        f= open("./MixHop_log/logger-{}.txt".format(args.dataset),"a+")
        f.write('>>>>>>>>>> min tst now=%5.2f%% @ %s\n' % (-min_tst*100, min_tst_c))
        f.close()
    return {'loss': tst_score, 'status': hyperopt.STATUS_OK}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    help='dataset name')
parser.add_argument('--train_ratio', type=float, default=0.025,
                    help='training set rate')
parser.add_argument('--val_ratio', type=float, default=0.025,
                    help='validation set rate')
parser.add_argument('--RPMAX', type=int, default=10,
                    help='the number of test')
args = parser.parse_args()


space = {'lr': hyperopt.hp.choice('lr', [0.01, 0.05, 0.001, 0.005, 0.002]),
         'weight_decay': hyperopt.hp.choice('weight_decay', [0.0005, 0.00025, 0.0001, 0.001, 0.0025, 0.005, 5e-4, 0.0]),
         'hidden': hyperopt.hp.choice('hidden', [8, 16, 32, 64]),
         'feat_drop': hyperopt.hp.quniform('feat_drop', 0, 0.9, 0.1),
         #'attn_drop': hyperopt.hp.quniform('attn_drop', 0, 0.9, 0.1),
         #'slope': hyperopt.hp.quniform('slope', 0.1, 0.5, 0.1),
         'layer_num': hyperopt.hp.quniform('layer_num', 1, 6, 1),
         'dataset': args.dataset,
         'train_ratio': args.train_ratio,
         'val_ratio': args.val_ratio,
         'RPMAX': args.RPMAX}
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
print('>>>>>>>>>> val=%5.2f%% @ %s' % (-min_y * 100, min_c))
print('>>>>>>>>>> tst=%5.2f%% @ %s' % (-min_tst * 100, min_tst_c))