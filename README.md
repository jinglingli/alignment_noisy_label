# How Does a Neural Network’s Architecture Impact Its Robustness to Noisy Labels?

This repository is the PyTorch implementation of the experiments in the following paper: 

Jingling Li, Mozhi Zhang, Keyulu Xu, John Dickerson, Jimmy Ba. How Does a Neural Network’s Architecture Impact Its Robustness to Noisy Labels? NeurIPS 2021. 

[arXiv](https://arxiv.org/abs/2012.12896) 

If you make use of the relevant code/experiment/idea in your work, please cite our paper (Bibtex below).
```
@article{li2021does,
  title={How does a Neural Network's Architecture Impact its Robustness to Noisy Labels?},
  author={Li, Jingling and Zhang, Mozhi and Xu, Keyulu and Dickerson, John and Ba, Jimmy},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```


## Requirements
- This codebase has been tested for `python3.7` and `pytorch 1.4.0` (with `CUDA VERSION 10.0`).
- The packages [networkx](https://networkx.org/documentation/stable/install.html) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) need to be installed separately. networkx and geometric versions can be decided based on pytorch and CUDA version.

## Instructions
Refer to each folder for instructions to reproduce the experiments. 
- Experiments related to graph algorithms are in the [`gnn`](./gnn) folder.
- Experiments on image classification datasets are in the [`vision`](./vision) folder.
