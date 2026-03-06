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


def cmd_run(cmd, hyperpm, pm, mode, args):
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
    (tst, tst_std, mlp_mean, mlp_mean_std, gnn_diff, mlp_diff) = eval(output)

    filename = f'results/{args.data_type}_bi_semi_new_dataset.csv'
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{gnn}," + f"{dataset}," + f"{iter_num}," + f"{dis_weight}," + f"{gen_weight}," + f"{mlp_mean} +- {mlp_mean_std}," + f"{tst} +- {tst_std}," + f"{gnn_diff} +- {mlp_diff}\n")


def main(arg_str=None):        
# torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ns', default=False)
    parser.add_argument('--data_type', default='transductive')
    args = parser.parse_args()
    
    
    val_list = []
    tst_list = []
    mlp_mean_list = []
    trans_dataset_list = ['cs','physics']
    induc_dataset_list = ['arxiv', 'product']
    heter_dataset_list = ['pokec', 'chameleon', 'snap-patents']

#     GNN_list = ['GCN']
    GNN_list = ['GCN', 'GAT', 'FAGCN', 'MixHop', 'GCNII']
    ratio_list = [0.025]
    iter_list = [3,4,5,6,7]
    dis_weight_list = [1,10,100,1000]
    gen_weight_list = [0.01, 0.1, 1, 10, 100]
    
    if args.data_type == 'transductive':
        data_list = trans_dataset_list
    elif args.data_type == 'inductive':
        data_list = induc_dataset_list
    elif args.data_type == 'heterophily':
        data_list = heter_dataset_list
        
    for gnn in GNN_list:
        for dataset in data_list:
            for iter_num in iter_list:
                for dis_weight in dis_weight_list:
                    for gen_weight in gen_weight_list:
                        hyperpm = get_training_config('./train_config.yaml', gnn, dataset)
                        pm = [gnn, dataset, iter_num, dis_weight, gen_weight]
                        cmd = 'python train_GNN_auto_bi_w_noise.py --result_path results --iter_num ' + str(iter_num) + ' --dis_weight ' + str(dis_weight) + ' --gen_weight ' + str(gen_weight) + ' --dataset ' + str(dataset) + ' --with_Gen_for_gnn' + ' --with_Gen_for_mlp'
                        cmd_run(cmd, hyperpm, pm, 'GNN', args)

if __name__ == '__main__':
    main()
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
