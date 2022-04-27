# Experiments on vision datasets

We conduct the following sets of experiments on various image-calssification datasets with different neural network architectures 
- CIFAR-10/100 with uniform and flipped label noise
- CIFAR-Easy with uniform, flipped, and instance dependent label noise
- Clothing1M and WebVision which have non-synthetic noisy labels

The code this folder are adapted from https://github.com/LiJunnan1992/DivideMix.


## CIFAR-10/100

### Training with the DivideMix technique:

First, create the following folders using

```
mkdir results
mkdir models_dir
mkdir checkpoint
```

We use the [`Train_cifar.py`](./Train_cifar.py) to train models (MLPs, CNNs, and ResNets) on CIFAR-10/100 with DivideMix under uniform and flipped label noise. The flag `--dataset` chooses the dataset (cifar10 or cifar100), and the location to the dataset is passed to `--data_path`. The flag `--noise_mode` chooses which type of noisy labels (asym: flipped; sym: uniform), and `--r` selects the noise ratio. Other hyperparameters are the same as in the [DivideMix paper](https://openreview.net/pdf?id=HJgExaVtwr). For example, the command

```
python Train_cifar.py --lambda_u=25 --model=ResNet18 --dataset=cifar10 --noise_mode=asym --r=0.8 --checkpoint --save_model --correct --clean_ratio=0.1 --data_path=$CIFAR_PATH 
```

trains a ResNet18 model on CIFAR-10 under flipped label noise with 20% noise ratio. The flag `--save_model` should be added in order to save the trained model, and the flag `--correct` computes the predictive power in the learned representations after each epoch (with 10% clean labels).


To compute the predictive power with various amount of clean labels, we use [`obtain_embedding.py.py`](./obtain_embedding.py), where we feed the path to the saved model to the flag `--model_path` and selects the number of clean labels with the flag `--num_clean` (the flag `--count` selects the number of original predictions we want to use).
For example, 

```
python obtain_embedding.py --model_path=./models_dir/cifar10/asym_0.8/ResNet18_lr0.02_lambda25.0_clean0.1_epochs300_seed123_fliptype2016_addcleanFalse.log/model_epoch292.pth.tar --count=0 --num_clean=5000
```

### Vanilla training:

First, create the following folders using

```
mkdir standard
mkdir standard/results
mkdir standard/models_dir
mkdir standard/checkpoint
```

Similarly, we use the [`Train_standard.py`](./Train_standard.py) to perform vanilla trianing for models (MLPs, CNNs, and ResNets) on CIFAR-10/100 under uniform and flipped label noise. The flag `--dataset` chooses the dataset (cifar10 or cifar100), and the location to the dataset is passed to `--data_path`. The flag `--noise_mode` chooses which type of noisy labels (asym: flipped; sym: uniform), and `--r` selects the noise ratio. For example, the command

```
python Train_standard.py --model=ResNet18 --dataset=cifar10 --noise_mode=asym --r=0.8 --checkpoint --save_model --correct --clean_ratio=0.1 --data_path=$CIFAR_PATH
```

We also use [`obtain_embedding.py`](./obtain_embedding.py) to compute the predictive power for vanilla trained models. For example,


```
python obtain_embedding.py --model_path=./standard/models_dir/cifar10/asym_0.8/ResNet18_losscls_lr0.1_cleanonlyFalse_clean0.1_epochs300_seed122.log/model_epoch300.pth.tar --count=0 --num_clean=1000
```


## CIFAR-Easy

For the CIFAR-Easy task, first, create the following folder using

```
mkdir dataset
```

Then, generate the dataset via

```
python generate_cifar_easy.py --data_path=/home/jingling/dataset/cifar-10-python
```

To train on CIFAR-Easy with DivideMix, we use the [`Train_cifar_easy.py`](./Train_cifar_easy.py). For example,

```
python Train_cifar_easy.py --model=MLP --dataset=cifar10 --noise_mode=adv --r=0.2 --checkpoint --save_model --correct --clean_ratio=0.1 --data_path=$CIFAR_PATH
```

The noise mode chooses which kind of label noise we want to train on. Sym, asym, and adv coorespond to uniform, flipped and instance dependent label noise respectively.

Simillary, we use the [`Train_cifar_easy_standard.py`](./Train_cifar_easy_standard.py) to perform vanilla trianing on CIFAR-Easy. For example,

```
python Train_cifar_easy_standard.py --model=MLP --dataset=cifar10 --noise_mode=adv --r=0.2 --checkpoint --save_model --correct --clean_ratio=0.1 --data_path=$CIFAR_PATH
```


## Datasets with real world label noise

We use the same hyperparameter as in DivideMix to validate our hypothesis on Clothing1M and Webvision. We use [`Train_clothing1M.py`](./Train_clothing1M.py) and [`Train_webvision.py`](./Train_webvision.py) to train models on these two datasets respectively. 
Similarly, we use [`obtain_embedding_clothing1M.py`](./obtain_embedding_clothing1M.py) and [`obtain_embedding_webvision.py`](./obtain_embedding_webvision.py) to calculate the predictive power for trained models.
