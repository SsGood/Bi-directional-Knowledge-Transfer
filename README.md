
# 🚀 BiKT: Unleashing the Potential of GNNs via Bi-Directional Knowledge Transfer

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

This repository contains the official implementation of BiKT. BiKT is a novel approach for Graph Neural Networks leveraging bi-directional knowledge transfer mechanisms to enhance model performance and generalization capabilities. The associated paper is titled "BiKT: Unleashing the Potential of GNNs via Bi-Directional Knowledge Transfer" and will appear in IEEE Transactions on Pattern Analysis and Machine Intelligence 2026.
<img width="1742" height="756" alt="image" src="https://github.com/user-attachments/assets/5b73a91f-1f49-4184-bde5-f6c0775b5125" />


## ✨ Key Features

The framework introduces a novel mechanism for transferring knowledge in both directions within graph networks. It achieves competitive results on multiple benchmark datasets. The provided API and examples simplify execution and evaluation. Complete code is available for reproducing paper results.

## 🛠️ Requirements

Execution requires Python 3.7 or higher alongside PyTorch 1.9.0 or higher. The environment must also include PyTorch Geometric 2.0, NumPy, and SciPy.

## 💻 Installation

Clone the repository and install the required dependencies using the package manager.

```bash
git clone [https://github.com/SsGood/Bi-directional-Knowledge-Transfer.git](https://github.com/SsGood/Bi-directional-Knowledge-Transfer.git)
cd Bi-directional-Knowledge-Transfer

```

## 🚀 Usage

The core training pipeline is controlled by the primary execution script. Initiate a single training task by specifying the target dataset and the foundational graph neural network architecture.

```bash
python src_final/train_GNN_auto_bi_w_noise.py --dataset cora --net GraphSAGE

```

Supported datasets encompass citation networks, e-commerce purchase networks, and heterophilic graphs. Valid dataset arguments include `cora`, `citeseer`, `pubmed`, `computers`, `photo`, `arxiv`, and `chameleon`. Supported backbone architectures include `GCN`, `GraphSAGE`, `GAT`, `FAGCN`, `MixHop`, and `GCNII`.

Evaluate the models using the optimal hyperparameters already discovered and saved in the configuration file by executing the run script. This script automatically loads the predefined configurations from the yaml file and executes the evaluation across multiple standard datasets.

```bash
python src_final/run.py --net GCN

```

The framework allows fine-grained control over the alternating optimization process between the multi-layer perceptron and the graph network modules. Command line arguments override default configurations to adjust iteration rounds, loss function weights, and batch sizes.

```bash
python src_final/train_GNN_auto_bi_w_noise.py \
    --dataset computers \
    --net GAT \
    --iter_num 3 \
    --batch_size 1024 \
    --dis_weight 100 \
    --gen_weight 1 \
    --diversity_weight 10 \
    --with_Gen_for_gnn \
    --with_Gen_for_mlp

```

The `--iter_num` parameter sets the total number of alternating optimization iterations between the multi-layer perceptron and graph neural network modules. The `--dis_weight` controls the divergence loss weight during the knowledge distillation process. The `--gen_weight` defines the generator loss weight within the adversarial training phase. The `--diversity_weight` applies a penalty weight for the diversity of generated samples. The `--with_Gen_for_gnn` and `--with_Gen_for_mlp` flags act as independent switches to enable generator noise injection during the respective network training phases.

Comprehensive benchmarking across distinct dataset categories relies on automated scheduling scripts. These scripts parse optimal hyperparameter matrices stored in the configuration file and sequentially dispatch training tasks across specified dataset lists.

```bash
python src_final/search_file/mprf_bi.py --data_type transductive
python src_final/search_file/mprf_bi.py --data_type inductive

```

Upon completion, test accuracies and standard deviations for all models are automatically aggregated and appended to CSV record files located in the results directory.

Bayesian parameter search scripts facilitate the exploration of performance boundaries for newly integrated datasets. The search program executes multiple exploratory training runs within a predefined parameter space.

```bash
python src_final/meta_file/meta_GCN.py --dataset cora --RPMAX 15

```

The optimization algorithm identifies the configuration combination that minimizes validation set loss and persistently stores the search trajectory in dedicated log files.

## 🏗️ Model Architecture

The forward direction passes node features to neighborhood aggregation. The backward direction passes neighborhood information back to node representations. A transfer module learns to move knowledge between these directions using an adaptive weighting mechanism. The graph neural backbone supports various architectures like GCN and GraphSAGE.

## 📊 Results

Our method achieves competitive performance compared to strong baselines. Results may vary slightly depending on the random seed and hyperparameters.

<img width="1246" height="932" alt="image" src="https://github.com/user-attachments/assets/16ce8b98-c364-4d12-8c1e-eeeebca88d61" />

## ⚙️ Configuration

Key hyperparameters can be configured in the `src_final/train_config.yaml` file.

## 📝 Citation

If you use this code in your research, please cite our paper.

```bibtex
@article{bikt2026,
  title={BiKT: Unleashing the Potential of GNNs via Bi-Directional Knowledge Transfer},
  author={Zheng, Shuai and Liu, Zhizhe and Zhu, Zhenfeng and Zhang, Xingxing and Li, Jianxin and Zhao, Yao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  volume={48},
  pages={3304 - 3318}
}

```

**Last Updated**: 2026-03-06



```
