# Experiments on graph algorithms using GNNs 

This folder contains two tasks with different types of noisy labels, in which we show models that are more **algorithmic aligned** with the target function than the noise fucntion learn more predictive representations under noise label training: 
- Additive label noise
- Instance dependent label noise

First of all, create the following two folders via

```
mkdir data
mkdir run
```

## Additive label noise

### Data Generation

We use [`data_generation.py`](./data_generation.py) to generate graphs with varied sizes and different types of noise labels. The raw generated data is stored under the [`data`][./data] folder with an indicator file stored under the [`run`][./run] folder. The indicator file is used in the training stage as the raw data name is too long to be fed into [`main.py`][./main.py]. It is suggested that you name the indicator file yourself using the `--run` flag for better bookkeeping.
To generate additive label noise from the different distributions used in the paper, run 

```
bash additive_label_noise.sh
```

### Training

To train on the generated dataset, just feed the indicator filename to the `--train` flag. For example,

```
python main.py --model=GGNN --n_iter=2 --hidden_dim=128 --mlp_layer=3 --fc_output_layer=1 --graph_pooling_type=max --neighbor_pooling_type=sum --lr=0.001 --batch_size=64 --epochs=200 --loss_fn=reg --save_model --train=additive_gaussian_0.0_40.0_corr0.75  
```

which trains a 2-layer max-sum GNN on the dataset with 75% labels corrupted by adding a zero-mean Gaussian noise (std=40).
The trained models will be stored under the [`models_dir`][./models_dir] folder.

### Evaluating predictive power

Once the model is trained, we can evalute its representation's predictive power with respect to the target function (maximum degree of a graph) using [`calc_pred.py`][./calc_pred.py] by supplying the path to the saved model to the flag `--model_path` and the data file (which indicates the target function by its name) to the flag `--data`. For example, 
```
python calc_pred.py --data=additive_gaussian_0.0_40.0_corr0.75 --model_path='./models_dir/Train_additive_gaussian_0.0_40.0_corr0.75/Test_additive_gaussian_0.0_40.0_corr0.75/GGNN_lossreg_2_lr0.001_decay0_hdim128_fc1_mlp3_max_sum_bs64_epoch200_seed2.log/model_best.pth.tar'
```

## Instance dependent label noise

### Data Generation

We use [`data_generation.py`](./data_generation.py) to generate graphs with varied sizes and different types of noise labels. The raw generated data is stored under the [`data`][./data] folder with an indicator file stored under the [`run`][./run] folder. The indicator file is used in the training stage as the raw data name is too long to be fed into [`main.py`][./main.py]. It is suggested that you name the indicator file yourself using the `--run` flag for better bookkeeping.
To generate the instance dependent label noise used in the paper, run 

```
bash dependent_label_noise.sh
```

### Training

To train on the generated dataset, just feed the indicator filename to the `--train` flag. For example,

```
python main.py --model=GGNN --n_iter=2 --hidden_dim=128 --mlp_layer=3 --fc_output_layer=1 --graph_pooling_type=max --neighbor_pooling_type=max --lr=0.001 --batch_size=64 --epochs=200 --loss_fn=reg --save_model --train=dependent_label_noise_corr0.75  
```

which trains a 2-layer max-max GNN on the dataset with 75% labels corrupted by the instance dependent label noise.
The trained models will be stored under the [`models_dir`][./models_dir] folder.

### Evaluating predictive power

Once the model is trained, we can evalute its representation's predictive power with respect to the target function (maximum degree of a graph) using [`calc_pred.py`][./calc_pred.py] by supplying the path to the saved model to the flag `--model_path` and the data file (which indicates the target function by its name) to the flag `--data`. For example, 
```
python calc_pred.py --data=dependent_label_noise_corr0.75 --model_path='./models_dir/Train_dependent_label_noise_corr0.75/Test_dependent_label_noise_corr0.75/GGNN_lossreg_2_lr0.001_decay0_hdim128_fc1_mlp3_max_max_bs64_epoch200_seed2.log/model_best.pth.tar'
```