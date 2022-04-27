#!/bin/bash 

python data_generation.py --prefix=dependent --train_corr=0.0 --run='dependent_label_noise_corr0.0'
python data_generation.py --prefix=dependent --train_corr=0.1 --run='dependent_label_noise_corr0.1'
python data_generation.py --prefix=dependent --train_corr=0.25 --run='dependent_label_noise_corr0.25'
python data_generation.py --prefix=dependent --train_corr=0.4 --run='dependent_label_noise_corr0.4'
python data_generation.py --prefix=dependent --train_corr=0.5 --run='dependent_label_noise_corr0.5'
python data_generation.py --prefix=dependent --train_corr=0.6 --run='dependent_label_noise_corr0.6'
python data_generation.py --prefix=dependent --train_corr=0.75 --run='dependent_label_noise_corr0.75'
python data_generation.py --prefix=dependent --train_corr=0.9 --run='dependent_label_noise_corr0.9'
python data_generation.py --prefix=dependent --train_corr=1.0 --run='dependent_label_noise_corr1.0'