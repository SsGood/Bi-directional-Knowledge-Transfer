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

def main(arg_str=None):        
# torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ns', default=False)
    parser.add_argument('--data_type', default='transductive')
    args = parser.parse_args()
    
    
    val_list = []
    tst_list = []
    mlp_mean_list = []
    trans_dataset_list = ['computers']
    induc_dataset_list = ['arxiv', 'product']
    heter_dataset_list = ['pokec', 'chameleon', 'snap-patents']
    
    #GNN_list = ['GCN', 'GAT', 'GraphSAGE', 'FAGCN']
    #GNN_list = ['GAT', 'FAGCN']
    GNN_list = ['GCN', 'GraphSAGE', 'GAT','FAGCN']
    ratio_list = [0.05]
    
    if args.data_type == 'transductive':
        data_list = trans_dataset_list
    elif args.data_type == 'inductive':
        data_list = induc_dataset_list
    elif args.data_type == 'heterophily':
        data_list = heter_dataset_list
        
    for gnn in GNN_list:
        for dataset in data_list:
            for ratio in ratio_list:
                hyperpm = get_training_config('./train_config.yaml', gnn, dataset)
                cmd = 'python train_GCN.py --train_ratio ' + str(ratio) + ' --val_ratio ' + str(ratio) + ' --dataset ' + str(dataset)

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
                (val, val_std, tst, tst_std, mlp_mean, mlp_mean_std) = eval(output)
    #             val_list.append(val)
    #             tst_list.append(tst)
    #             mlp_mean_list.append(mlp_mean)

                filename = f'results/{args.data_type}_mlp.csv'
                with open(f"{filename}", 'a+') as write_obj:
                    write_obj.write(f"{gnn}," + f"{dataset}," + f"{ratio}," + f"{mlp_mean} +- {mlp_mean_std}," + f"{tst} +- {tst_std}\n")

#             f = open('{}_train_ratio_{}.txt'.format(args.data_type, args.train_ratio), 'a+')
#             #f.write(cmd)
#             f.write('>>> val = %5.2f%% +- %.3f tst = %5.2f%% +- %.3f mlp_mean = %5.2f%% +- %.3f --dataset %s --GNN %s \n' % (val * 100, val_std * 100, tst * 100, tst_std * 100, mlp_mean * 100, mlp_mean_std * 100, dataset, gnn))
#             f.close()
            
        
if __name__ == '__main__':
    main()
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()