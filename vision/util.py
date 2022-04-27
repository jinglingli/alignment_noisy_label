import os
import re
import copy
import pickle
import torch
import numpy as np
import random
from os import listdir
from os.path import isfile, join

def use_ELR_results_from_log(log):
    test_acc = []
    if 'cifar100' in log:
        start, end = 241, 251
    else:
        start, end = 191, 201
    for epoch in range(start, end):
        model_path = f"{log}/model_info/epoch{epoch}.p"
        model_acc_dict = pickle.load(open(model_path, 'rb'))
        acc = model_acc_dict['test_acc']
        test_acc.append(acc)
    return np.mean(test_acc), np.std(test_acc)
    
def use_results_from_log(log):
    epochs_a, model_test_accs = obtain_acc_from_log(log, keys=['Test Epoch'])
    model_best_test_acc, model_avg_last10_test_acc, model_std_last10_test_acc = avg_acc_from_log(epochs_a, model_test_accs)
    return model_avg_last10_test_acc, model_std_last10_test_acc

def use_correct_results_from_log(log):
    epochs_a, model_test_accs = obtain_acc_from_log(log, keys=['Correct Epoch', 'reg test'])
    model_best_test_acc, model_avg_last10_test_acc, model_std_last10_test_acc = avg_acc_from_log(epochs_a, model_test_accs)
    return model_avg_last10_test_acc, model_std_last10_test_acc

def use_results_from_paper(paper, data, noise_type, paper_dict): 
    if data in paper_dict[paper]['mean'] and noise_type in paper_dict[paper]['mean'][data]:
        mean = paper_dict[paper]['mean'][data][noise_type]
    else:
        return False, False # no reported acc
    if 'std' in paper_dict[paper]:
        if noise_type in paper_dict[paper]['std'][data]:
            std = paper_dict[paper]['std'][data][noise_type]
        else:
            std = 0
    else:
        std = 0
    return mean, std

def use_results_from_info(avg_path, n_true_data):
    plist = avg_path.split('/')
    for p in plist:
        if 'cifar' in p:
            data = p
        if 'sym' in p:
            noise_type = p
    
    with open(avg_path, 'rb') as f:
        total_avg_info = pickle.load(f)
    
    i = total_avg_info['x'].index(n_true_data)
    
    if n_true_data >= 5000:
        count = 0
    else:
        count = 5000

    if 'asym' in noise_type:
        # use {n_true_data} randomly selected samples + {0} training data w/ predicted labels
        # to train the linear classifier
        mean, std = total_avg_info[0]['rand'][0][i], total_avg_info[0]['rand'][1][i] 
    else:
        if '0.9' in noise_type:
            # same as asym
            mean, std = total_avg_info[5000]['least'][0][i], total_avg_info[5000]['least'][1][i] 
        else:
            # use {n_true_data} least selected samples + {5k} training data w/ predicted labels
            # to train the linear classifier
            mean, std = total_avg_info[5000]['least'][0][i], total_avg_info[5000]['least'][1][i] 
            # mean, std = total_avg_info[count]['least'][0][i], total_avg_info[count]['least'][1][i] 
    return mean, std

def use_results_from_info_ridge(avg_path, n_true_data):
    plist = avg_path.split('/')
    for p in plist:
        if 'cifar' in p:
            data = p
        if 'sym' in p:
            noise_type = p
    
    with open(avg_path, 'rb') as f:
        total_avg_info = pickle.load(f)
    
    i = total_avg_info['x'].index(n_true_data)
    
    if 'asym' in noise_type:
        # use {n_true_data} randomly selected samples + {0} training data w/ predicted labels
        # to train the linear classifier
        mean, std = total_avg_info[0]['rand'][0][i], total_avg_info[0]['rand'][1][i] 
    else:
        if '0.9' in noise_type:
            # same as asym
            mean, std = total_avg_info[500]['rand'][0][i], total_avg_info[500]['rand'][1][i] 
        else:
            # use {n_true_data} least selected samples + {5k} training data w/ predicted labels
            # to train the linear classifier
            #mean, std = total_avg_info[5000]['least'][0][i], total_avg_info[5000]['least'][1][i] 
            mean, std = total_avg_info[500]['rand'][0][i], total_avg_info[500]['rand'][1][i] 
    return mean, std

# prepocess avg across last 10 epochs
def save_avg_info_last_10_epochs(onlyfiles):
    counts = [0, 500, 5000]
    for log in onlyfiles:
        total_avg_info = {}
        for count in counts:
            total_avg_info[count] = {}
            avg_info = {'least': [], 'top': [], 'rand': []}
            for epoch in range(291, 301, 1):
                info_path = f"{log.replace('results', 'models_dir')}/total_info/epoch{epoch}_count{count}.p"
                if not os.path.exists(info_path):
                    info_path = f"{log.replace('results', 'models_dir')}/total_info/epoch{epoch}_count{count}_lout5.p"
                with open(info_path, 'rb') as f:
                    info_dic = pickle.load(f)
                for key in avg_info:
                    avg_info[key].append(info_dic[key][0])

            for key in avg_info:
                mean, std, maxv, minv = np.asarray(avg_info[key]).mean(axis=0), np.asarray(avg_info[key]).std(axis=0), np.asarray(avg_info[key]).min(axis=0), np.asarray(avg_info[key]).max(axis=0)
                total_avg_info[count][key] = (mean, std, maxv, minv)
        total_avg_info['x'] = info_dic['x']
        save_path = f"{log.split('.log')[0]}_avg_info.p"
        pickle.dump(total_avg_info, open(save_path, "wb" ))
        print(f"total info saved to {save_path}")

def save_avg_info_last_10_epochs_ridge(onlyfiles):
    counts = [0, 500, 1000, 5000]
    for log in onlyfiles:
        total_avg_info = {}
        for count in counts:
            total_avg_info[count] = {}
            avg_info = {'least': [], 'top': [], 'rand': []}
            for epoch in range(291, 301, 1):
                info_path = f"{log.replace('results', 'models_dir')}/total_info_ridge/epoch{epoch}_count{count}.p"
                with open(info_path, 'rb') as f:
                    info_dic = pickle.load(f)
                for key in avg_info:
                    avg_info[key].append(info_dic[key][0])

            for key in avg_info:
                mean, std, maxv, minv = np.asarray(avg_info[key]).mean(axis=0), np.asarray(avg_info[key]).std(axis=0), np.asarray(avg_info[key]).min(axis=0), np.asarray(avg_info[key]).max(axis=0)
                total_avg_info[count][key] = (mean, std, maxv, minv)
        total_avg_info['x'] = info_dic['x']
        save_path = f"{log.split('.log')[0]}_avg_info_ridge.p"
        pickle.dump(total_avg_info, open(save_path, "wb" ))
        print(f"total info saved to {save_path}")        
        
def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def avg_acc_from_log(epochs, test_accs, total_epochs=300):
    best_test_acc = test_accs.max()
    turn_on = False
    avg_last10_test_accs = []
    for i in range(len(epochs)):
        epoch = epochs[i]
        test_acc = test_accs[i]
        if epoch == total_epochs - 10 + 1:
            turn_on = True
        if turn_on:
            avg_last10_test_accs.append(test_acc)
            if epoch == 300:
                break
    avg_last10_test_acc = np.mean(avg_last10_test_accs)
    std_last10_test_acc = np.std(avg_last10_test_accs)
    return best_test_acc, avg_last10_test_acc, std_last10_test_acc

def obtain_acc_from_log(log_fn, keys):
    with open(log_fn, 'r') as f:
        lines = f.readlines()
   
    for key in keys:
        lines = [line for line in lines if key in line]
    
    epochs, accs = [], []
    for line in lines:
        epoch = int(re.findall("\d+", line)[0])
        acc = float(re.findall("\d+\.\d+", line)[0])
        epochs.append(epoch)
        accs.append(acc)

    return np.asarray(epochs), np.asarray(accs)

def obtain_acc_from_correct_info(log):
    reg_avg_last10_test_accs = []
    model_avg_last10_test_accs = []
    for epoch in range(291, 301, 1):
        info_path = f"{log.replace('results', 'models_dir')}/correct_info/epoch_{epoch}.p"
        with open(info_path, 'rb') as f:
            info_dict = pickle.load(f)
            reg_acc = info_dict['reg_test_acc']
            if reg_acc < 1.0:
                reg_acc = reg_acc*100
            reg_avg_last10_test_accs.append(reg_acc)
            model_avg_last10_test_accs.append(info_dict['model_test_acc'])
        
    return np.max(model_avg_last10_test_accs), np.mean(model_avg_last10_test_accs), np.max(reg_avg_last10_test_accs), np.mean(reg_avg_last10_test_accs)
    
def generate_table_ridge(datas, paper_lists, filename):
    f = open(filename, "w+")
    
    max_mean = {}
    for data in datas:
        noise_types = datas[data]
        for paper in paper_lists: 
            if paper == 'Cross-Entropy':
                latex_line = "%s " %(paper)
            else:
                latex_line = ""
                cite = paper_cites[paper]
                if 'sym' in noise_types[0]:
                    latex_line += "%s~\\cite{%s} " %(paper, cite)
                else:
                    latex_line += "%s~\\cite{%s} " %(paper, cite)
            m = paper_dict[paper]['M'][data]
            latex_line += "& %s " %(m)
            for noise_type in noise_types:
                add_mark = False
                if paper == 'ELR' and noise_type != "sym_0.0":
                    log = dict_to_log_ELR[data][noise_type]
                else:
                    log = dict_to_log[data][noise_type]
                    
                if paper == 'Cross-Entropy':
                    log = CE_dict_to_log[data][noise_type]
                    mean, std = use_results_from_log(log)
                else:
                    mean, std = use_results_from_paper(paper, data, noise_type, paper_dict)
                if not mean:
                    if paper == 'DivideMix':
                        add_mark = True
                        mean, std = use_results_from_log(log)
                    elif paper == 'ELR':
                        add_mark = True
                        mean, std = use_ELR_results_from_log(log)
                        std = 0
                    else:
                        mean, std = '-', 0
                        
                if mean != '-':
                    if data not in max_mean:
                        max_mean[data] = {}
                    if noise_type not in max_mean[data]:
                        max_mean[data][noise_type] = mean
                    if max_mean[data][noise_type] < mean:
                        max_mean[data][noise_type] = mean
                        
                if std != 0:
                    latex_line += f"& {mean:.1f} $\pm$ {std:.1f} "
                else:
                    if mean == '-':
                        latex_line += f"& {mean} "
                    else:
                        if add_mark and noise_type != "sym_0.0":
                            #if data == 'cifar100' and  noise_type == 'asym_0.2':
                            #    latex_line += "& \\textbf{%.1f}$%s$ " %(mean, mark)
                            #else:
                            latex_line += f"& {mean:.1f}${mark}$ "
                        else:
                            latex_line += f"& {mean:.1f} "
                
            latex_line += f"\\\\"
            #print(latex_line)
            f.write(f"{latex_line} \n")
    #print("\midrule")   
    f.write("\midrule \n")
    
    for data in datas:
        if data == 'cifar100':
            n_true_datas = [1000, 5000]
        else:
            n_true_datas = [100, 1000, 5000]

        noise_types = datas[data]
        for n_true_data in n_true_datas:
            latex_line = f"DivideMix{ours} & "
            '''
            if n_true_data % 1000 != 0:
                m = f"{n_true_data/1000}k"
            else:
                m = f"{int(n_true_data/1000)}k"
            '''
            if data == 'cifar100':
                m = f"{int(n_true_data/100)}"
            else:
                m = f"{int(n_true_data/10)}"
            latex_line += "%s " %(m)
            for noise_type in noise_types:
                log = dict_to_log[data][noise_type]
                if noise_type == 'sym_0.0':
                    mean, std = use_results_from_log(log)
                else:
                    avg_path = f"{log.split('.log')[0]}_avg_info_ridge.p"
                    mean, std = use_results_from_info_ridge(avg_path, n_true_data=n_true_data) 
                if std != 0:
                    if 'asym' in noise_type:
                        entry = f"{mean:.2f} $\pm$ {std:.2f}"
                    else:
                        entry = f"{mean:.1f} $\pm$ {std:.1f}"
                else:
                    if 'asym' in noise_type:
                        entry = f"{mean:.2f}"
                    else:
                        entry = f"{mean:.1f}"
                if mean > max_mean[data][noise_type] and noise_type != 'sym_0.0':
                    entry = "& \\textbf {%s} " %entry
                else:
                    entry = f"& {entry}"
                latex_line += entry
            latex_line += f"\\\\"
            print(latex_line)
            f.write(f"{latex_line} \n")
        
        if 'ELR' in paper_lists:    
            for n_true_data in n_true_datas:
                latex_line = f"ELR+{ours} & "
                '''
                if n_true_data % 1000 != 0:
                    m = f"{n_true_data/1000}k"
                else:
                    m = f"{int(n_true_data/1000)}k"
                '''
                if data == 'cifar100':
                    m = f"{int(n_true_data/100)}"
                else:
                    m = f"{int(n_true_data/10)}"
                latex_line += "%s " %(m)
                for noise_type in noise_types:
                    log = dict_to_log_ELR[data][noise_type]
                    if noise_type == 'sym_0.0':
                        mean, std = use_results_from_log(log)
                    else:
                        avg_path = f"{log}/total_info_avg_info_ridge.p"
                        mean, std = use_results_from_info_ridge(avg_path, n_true_data=n_true_data) 
                    if std != 0:
                        if 'asym' in noise_type:
                            entry = f"{mean:.2f} $\pm$ {std:.2f}"
                        else:
                            entry = f"{mean:.1f} $\pm$ {std:.1f}"
                    else:
                        if 'asym' in noise_type:
                            entry = f"{mean:.2f}"
                        else:
                            entry = f"{mean:.1f}"
                    if mean > max_mean[data][noise_type] and noise_type != 'sym_0.0':
                        entry = "& \\textbf {%s} " %entry
                    else:
                        entry = f"& {entry}"
                    latex_line += entry
                latex_line += f"\\\\"
                print(latex_line)
                f.write(f"{latex_line} \n")
    f.close()
    return max_mean    