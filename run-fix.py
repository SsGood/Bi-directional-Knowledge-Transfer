import subprocess
import numpy as np
import argparse
import gc
import torch
import yaml

def get_training_config(config_path, model_name, dataset):
    with open(config_path, "r") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    dataset_specific_config = full_config["global"]
    model_specific_config = full_config[dataset][model_name]

    if model_specific_config is not None:
        specific_config = dict(dataset_specific_config, **model_specific_config)
    else:
        specific_config = dataset_specific_config

    specific_config["net"] = model_name
    return specific_config

def cmd_run(cmd, hyperpm, pm, args):
    [gnn, dataset, iter_num, dis_weight, gen_weight] = pm
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        if isinstance(v, str):
            cmd += ' %s' %v
        elif int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))

    print(cmd)
    output = subprocess.check_output(cmd, shell=True)
    (tst_acc_mean, mlp_mean, tst_acc_std, mlp_std, init_tst_acc_mean, init_mlp_mean, init_tst_acc_std, init_mlp_std) = eval(output)
    
    filename = f'results/re_initial.csv'
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{gnn}," + f"{dataset}," + f"{tst_acc_mean} +- {tst_acc_std}," + f"{mlp_mean} +- {mlp_std}," + f"{init_tst_acc_mean} +- {init_tst_acc_std}," + f"{init_mlp_mean} +- {init_mlp_std}\n")
    
def main(arg_str=None):        
# torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--net', default='GCN')
    
    parser.add_argument('--train_ratio', type=float, default=0.025, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.025, help='Ratio of training set')
    parser.add_argument('--RPMAX', type=int, default=10, help='seed')
    parser.add_argument('--dis_weight', type=float, default=100, help='loss weight')
    parser.add_argument('--gen_weight', type=float, default=1, help='gen loss weight')
    parser.add_argument('--iter_num', type=int, default=2, help='Odd numbers are suspended by GNN and even numbers by MLP')
    
    args = parser.parse_args()
    
#     for dataset in ['cora', 'citeseer', 'pubmed', 'computers', 'photo']:
    for dataset in ['cora', 'citeseer', 'pubmed', 'computers', 'photo']:
        args.dataset = dataset
        hyperpm = get_training_config('./train_config.yaml', args.net, args.dataset)
        args_pm = get_training_config('./hyper_config.yaml', args.net, args.dataset)
        pm = [args.net, args.dataset, args_pm['iter_num'], args_pm['dis_weight'], args_pm['gen_weight']]

        cmd = f'python train_GNN_auto_bi_w_noise.py --result_path results --iter_num {args_pm['iter_num']} --dis_weight {args_pm['dis_weight']} --gen_weight {args_pm['gen_weight']} --dataset {args.dataset}'
        cmd_run(cmd, hyperpm, pm, args)
    
if __name__ == '__main__':
    main()
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
