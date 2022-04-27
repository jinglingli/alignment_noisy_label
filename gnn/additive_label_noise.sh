#!/bin/bash 

# gaussian noise with mean=0.0 and std=40.0
python data_generation.py --prefix=additive --sampling=gaussian --mean=0.0 --train_noise=40.0 --train_corr=0.0 --run='additive_gaussian_0.0_40.0_corr0.0'
python data_generation.py --prefix=additive --sampling=gaussian --mean=0.0 --train_noise=40.0 --train_corr=0.25 --run='additive_gaussian_0.0_40.0_corr0.25'
python data_generation.py --prefix=additive --sampling=gaussian --mean=0.0 --train_noise=40.0 --train_corr=0.5 --run='additive_gaussian_0.0_40.0_corr0.5'
python data_generation.py --prefix=additive --sampling=gaussian --mean=0.0 --train_noise=40.0 --train_corr=0.75 --run='additive_gaussian_0.0_40.0_corr0.75'
python data_generation.py --prefix=additive --sampling=gaussian --mean=0.0 --train_noise=40.0 --train_corr=1.0 --run='additive_gaussian_0.0_40.0_corr1.0'

# gaussian noise with mean=10.0 and std=15.0
python data_generation.py --prefix=additive --sampling=gaussian --mean=10.0 --train_noise=15.0 --train_corr=0.0 --run='additive_gaussian_10.0_15.0_corr0.0'
python data_generation.py --prefix=additive --sampling=gaussian --mean=10.0 --train_noise=15.0 --train_corr=0.25 --run='additive_gaussian_10.0_15.0_corr0.25'
python data_generation.py --prefix=additive --sampling=gaussian --mean=10.0 --train_noise=15.0 --train_corr=0.5 --run='additive_gaussian_10.0_15.0_corr0.5'
python data_generation.py --prefix=additive --sampling=gaussian --mean=10.0 --train_noise=15.0 --train_corr=0.75 --run='additive_gaussian_10.0_15.0_corr0.75'
python data_generation.py --prefix=additive --sampling=gaussian --mean=10.0 --train_noise=15.0 --train_corr=1.0 --run='additive_gaussian_10.0_15.0_corr1.0'

#gamma noise with mean=0.0 and scale=15.0 
python data_generation.py --prefix=additive --sampling=gamma --mean=0.0 --train_noise=15.0 --train_corr=0.0 --run='additive_gamma_0.0_15.0_corr0.0'
python data_generation.py --prefix=additive --sampling=gamma --mean=0.0 --train_noise=15.0 --train_corr=0.25 --run='additive_gamma_0.0_15.0_corr0.25'
python data_generation.py --prefix=additive --sampling=gamma --mean=0.0 --train_noise=15.0 --train_corr=0.5 --run='additive_gamma_0.0_15.0_corr0.5'
python data_generation.py --prefix=additive --sampling=gamma --mean=0.0 --train_noise=15.0 --train_corr=0.75 --run='additive_gamma_0.0_15.0_corr0.75'
python data_generation.py --prefix=additive --sampling=gamma --mean=0.0 --train_noise=15.0 --train_corr=1.0 --run='additive_gamma_0.0_15.0_corr1.0'