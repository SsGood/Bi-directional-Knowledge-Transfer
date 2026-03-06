# BiKT: Unleashing the Potential of GNNs via Bi-Directional Knowledge Transfer

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This repository contains the official implementation of **BiKT** (Bi-directional Knowledge Transfer), a novel approach for Graph Neural Networks (GNNs) that leverages bi-directional knowledge transfer mechanisms to enhance model performance and generalization capabilities.

**Paper**: "BiKT: Unleashing the Potential of GNNs via Bi-Directional Knowledge Transfer"  
**Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 2026

## Key Features

- 🚀 **Bi-directional Knowledge Transfer**: Novel mechanism for transferring knowledge in both directions within GNNs
- 📊 **State-of-the-art Performance**: Achieves superior results on multiple benchmark datasets
- 🔧 **Easy to Use**: Simple API and comprehensive examples
- 📖 **Well-documented**: Detailed documentation and usage instructions
- 🧪 **Reproducible**: Complete code for reproducing paper results

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0
- NumPy
- SciPy

## Installation

### Clone the Repository
```bash
git clone https://github.com/SsGood/Bi-directional-Knowledge-Transfer.git
cd Bi-directional-Knowledge-Transfer
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install required packages manually:
```bash
pip install torch
pip install torch-geometric
pip install numpy scipy scikit-learn
```

## Usage

### Running Experiments

```bash
python run-fix.py --dataset {dataset} --net {GNN}
```

## Model Architecture

### Core Components

1. **Bi-directional Information Flow**: 
   - Forward direction: Node features to neighborhood aggregation
   - Backward direction: Neighborhood information back to node representations

2. **Knowledge Transfer Module**:
   - Learns to transfer knowledge between directions
   - Adaptive weighting mechanism

3. **Graph Neural Backbone**:
   - Supports various GNN architectures (GCN, GraphSAGE, GAT, etc.)


## Results

Our method achieves competitive or superior performance compared to strong baselines:

<img width="1246" height="932" alt="image" src="https://github.com/user-attachments/assets/16ce8b98-c364-4d12-8c1e-eeeebca88d61" />


*Note: Results may vary slightly depending on the random seed and hyperparameters.*

## Configuration

### Hyperparameters

Key hyperparameters can be configured in `train-config.yaml` and `hyper_config.yaml`:

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
