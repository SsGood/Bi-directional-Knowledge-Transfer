
# BiKT: Unleashing the Potential of GNNs via Bi-Directional Knowledge Transfer

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This repository contains the official implementation of **BiKT**, a novel approach for Graph Neural Networks that leverages bi-directional knowledge transfer mechanisms to enhance model performance and generalization capabilities.

**Paper**: "BiKT: Unleashing the Potential of GNNs via Bi-Directional Knowledge Transfer"  
**Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence 2026

## Key Features

* **Bi-directional Knowledge Transfer**: Novel mechanism for transferring knowledge in both directions within graph networks.
* **State-of-the-art Performance**: Achieves superior results on multiple benchmark datasets.
* **Easy to Use**: Simple API and comprehensive examples.
* **Well-documented**: Detailed documentation and usage instructions.
* **Reproducible**: Complete code for reproducing paper results.

## Requirements

* Python >= 3.7
* PyTorch >= 1.9.0
* PyTorch Geometric >= 2.0
* NumPy
* SciPy

## Installation

### Clone the Repository
```bash
git clone [https://github.com/SsGood/Bi-directional-Knowledge-Transfer.git](https://github.com/SsGood/Bi-directional-Knowledge-Transfer.git)
cd Bi-directional-Knowledge-Transfer

```

### Install Dependencies

```bash
pip install -r requirements.txt

```

## Usage

### Running with Pre-searched Hyperparameters

To evaluate the models using the optimal hyperparameters already discovered and saved in the configuration file, execute the `run.py` script. This script automatically loads the predefined configurations from `train_config.yaml` and executes the evaluation across multiple standard datasets.

```bash
python src_final/run.py --net {GNNs} --dataset {dataset}

```


### Basic Model Training

The core training pipeline is controlled by the primary execution script. You can initiate a single training task by specifying the target dataset and the foundational graph neural network architecture.

```bash
python src_final/train_GNN_auto_bi_w_noise.py --dataset cora --net GraphSAGE

```

Supported datasets encompass citation networks, e-commerce purchase networks, and heterophilic graphs. Valid dataset arguments include `cora`, `citeseer`, `pubmed`, `computers`, `photo`, `arxiv`, and `chameleon`. Supported backbone architectures include `GCN`, `GraphSAGE`, `GAT`, `FAGCN`, `MixHop`, and `GCNII`.



### Customized Adversarial Training

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

Key configuration parameters dictate the training behavior:

* `--iter_num` sets the total number of alternating optimization iterations between the multi-layer perceptron and graph neural network modules.
* `--dis_weight` controls the divergence loss weight during the knowledge distillation process.
* `--gen_weight` defines the generator loss weight within the adversarial training phase.
* `--diversity_weight` applies a penalty weight for the diversity of generated samples.
* `--with_Gen_for_gnn` and `--with_Gen_for_mlp` act as independent switches to enable generator noise injection during the respective network training phases.

### Batch Testing and Automated Evaluation

Comprehensive benchmarking across distinct dataset categories relies on automated scheduling scripts. These scripts parse optimal hyperparameter matrices stored in the configuration file and sequentially dispatch training tasks across specified dataset lists.

```bash
# Execute batch testing on transductive learning datasets
python src_final/search_file/mprf_bi.py --data_type transductive

# Execute batch testing on inductive learning datasets
python src_final/search_file/mprf_bi.py --data_type inductive

```

Upon completion, test accuracies and standard deviations for all models are automatically aggregated and appended to CSV record files located in the results directory.

### Global Hyperparameter Optimization

Bayesian parameter search scripts facilitate the exploration of performance boundaries for newly integrated datasets. The search program executes multiple exploratory training runs within a predefined parameter space.

```bash
python src_final/meta_file/meta_GCN.py --dataset cora --RPMAX 15

```

The optimization algorithm identifies the configuration combination that minimizes validation set loss and persistently stores the search trajectory in dedicated log files.

## Model Architecture

### Core Components

1. **Bi-directional Information Flow**:
* Forward direction: Node features to neighborhood aggregation.
* Backward direction: Neighborhood information back to node representations.


2. **Knowledge Transfer Module**:
* Learns to transfer knowledge between directions.
* Adaptive weighting mechanism.


3. **Graph Neural Backbone**:
* Supports various graph network architectures.



## Results

Our method achieves competitive or superior performance compared to strong baselines:

<img width="1246" height="932" alt="image" src="https://github.com/user-attachments/assets/16ce8b98-c364-4d12-8c1e-eeeebca88d61" />

*Note: Results may vary slightly depending on the random seed and hyperparameters.*

## Configuration

### Hyperparameters

Key hyperparameters can be configured in `src_final/train_config.yaml`.

## Citation

If you use this code in your research, please cite our paper:

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

```
