from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import logging
import pickle
import numpy as np
from PreResNet import *
from model import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import json
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

'''
count: number of points with original prediction as the label
num_clean: number of clean indices
'''
def compute_mix(model_path, num_classes, count, num_clean, use_ridge=False):
    
    def eval_train_acc(epoch,net1,net2,eval_loader):
        net1.eval()
        net2.eval()
        correct = 0
        total = 0

        logits_total = []
        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)           
                logits = outputs1+outputs2    

                logits_total.append(logits.detach())

        logits_total = torch.cat(logits_total, dim=0).cpu()
        return logits_total

    def calc_correct(given_idx, count, idx, use_ridge=False): 
        random_clean = list(set(idx[:count]).union(set(given_idx)))
        embed1 = tr_embed1[random_clean]
        embed2 = tr_embed2[random_clean]

        mix_label = pred_labels.copy()
        mix_label[given_idx] = clean_label[given_idx]
        label = mix_label[random_clean]

        y_onehot = torch.zeros(len(label), args.num_class)
        y_onehot[torch.arange(len(label)), label] = 1

        if use_ridge:
            # use ridge regression
            clf1, clf2 = Ridge(alpha=1.0).fit(embed1, y_onehot),Ridge(alpha=1.0).fit(embed2, y_onehot)
        else:
            # use linear regression
            clf1, clf2 = LinearRegression().fit(embed1, y_onehot), LinearRegression().fit(embed2, y_onehot)

        logit1, logit2 = clf1.predict(embed1), clf2.predict(embed2)
        logit = logit1+logit2
        pred = logit.argmax(axis=1)
        clean_acc = sum(pred == label)/len(label)

        logit_tt1, logit_tt2 = clf1.predict(embed_tt1), clf2.predict(embed_tt2)
        logit_tt = logit_tt1 + logit_tt2
        pred_tt = logit_tt.argmax(axis=1)
        test_acc = sum(pred_tt == label_tt)/len(label_tt)

        return test_acc*100


    print("loading model...")
    
    args, net1, net2, epoch, train_acc, test_acc = load_model_from_path(model_path)

    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, clean_ratio=args.clean_ratio, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log="", noise_file=f'./noise_jason/{args.dataset}_{args.noise_mode}_{args.r}_clean_{args.clean_ratio}.json')
    clean_loader, test_loader = loader.run('clean'), loader.run('test')

    print(f"model original train acc: {train_acc}%, model test acc: {test_acc}%")

    eval_loader = loader.run('eval_train')
    tr_embed1, _ = obtain_embeddings_tr(args, eval_loader, net1)
    tr_embed2, tr_label = obtain_embeddings_tr(args, eval_loader, net2)

    embed_tt1, _ = obtain_embeddings(args, test_loader, net1)
    embed_tt2, label_tt = obtain_embeddings(args, test_loader, net2)    

    clean_embed1, _ = obtain_embeddings(args, clean_loader, net1)
    clean_embed2, small_clean_label = obtain_embeddings(args, clean_loader, net2)
    
    prob_logits = eval_train_acc(epoch,net1,net2,eval_loader)
    probs, pred_labels = torch.max(prob_logits, dim=1)
    probs = np.asarray(probs)
    pred_labels = np.asarray(pred_labels)

    ground_truth = f'./noise_jason/cifar{num_classes}_sym_0.0_clean_0.1.json'
    clean_label, noise_idx, clean_idx = json.load(open(ground_truth,"r"))
    clean_label = np.asarray(clean_label)

    _, _, clean_idx = json.load(open(loader.noise_file,"r"))
    clean_idx = np.asarray(clean_idx)  

    prob_indices = np.argsort(probs)
    prob_wrong_indices = prob_indices[clean_label[prob_indices] != pred_labels[prob_indices]]
    
    least_conf_accss, top_conf_accss, rand_conf_accss = [], [], []
    least_conf_accss_wrong, top_conf_accss_wrong, rand_conf_accss_wrong = [], [], []

    for seed in range(5):    
        idx = list(range(50000))
        random.shuffle(idx)

        rand_idx = list(range(50000))
        random.shuffle(rand_idx)

        least_conf_acc = calc_correct(prob_indices[:num_clean], count=count, idx=idx, use_ridge=use_ridge)
        top_conf_acc = calc_correct(prob_indices[-num_clean:], count=count, idx=idx, use_ridge=use_ridge)
        rand_conf_acc = calc_correct(rand_idx[:num_clean], count=count, idx=idx, use_ridge=use_ridge)

        least_conf_accss.append(top_conf_acc)
        top_conf_accss.append(top_conf_acc)
        rand_conf_accss.append(rand_conf_acc)

    least_y, least_std, least_min, least_max = np.asarray(least_conf_accss).mean(axis=0), np.asarray(least_conf_accss).std(axis=0), np.asarray(least_conf_accss).min(axis=0), np.asarray(least_conf_accss).max(axis=0)
    top_y, top_std, top_min, top_max = np.asarray(top_conf_accss).mean(axis=0), np.asarray(top_conf_accss).std(axis=0), np.asarray(top_conf_accss).min(axis=0), np.asarray(top_conf_accss).max(axis=0)
    rand_y, rand_std, rand_min, rand_max = np.asarray(rand_conf_accss).mean(axis=0), np.asarray(rand_conf_accss).std(axis=0), np.asarray(rand_conf_accss).min(axis=0), np.asarray(rand_conf_accss).max(axis=0)

    total_dict = {}
    total_dict['least'] = (least_y, least_std, least_min, least_max)
    total_dict['top'] = (top_y, top_std, top_min, top_max)
    total_dict['rand'] = (rand_y, rand_std, rand_min, rand_max)
    
    print(total_dict)
    
    pre = model_path.split("/model_epoch")[0]
    save_dir = f"{pre}/total_info"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/epoch{epoch}_count{count}.p"
    pickle.dump(total_dict, open(save_path, "wb" ))
    print(f"Stats on predictive power saved to {save_path}")
    return
    
    
def compute_mix_standard(model_path, num_classes, count, num_clean, use_ridge=False):

    def eval_train_acc(epoch, net):
        net.eval()

        correct = 0
        total = 0

        logits_total = []
        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = net(inputs)  

                logits_total.append(logits.detach())

        logits_total = torch.cat(logits_total, dim=0).cpu()
        return logits_total

    def calc_correct(given_idx, count, idx, use_ridge=False): 
        random_clean = list(set(idx[:count]).union(set(given_idx)))
        embed = tr_embed[random_clean]

        mix_label = pred_labels.copy()
        mix_label[given_idx] = clean_label[given_idx]
        label = mix_label[random_clean]

        y_onehot = torch.zeros(len(label), args.num_class)
        y_onehot[torch.arange(len(label)), label] = 1

        if use_ridge:
            # use ridge regression
            reg = Ridge(alpha=1.0).fit(embed, y_onehot)
        else:
            # use linear regression
            reg = LinearRegression().fit(embed, y_onehot)
            
        logit = reg.predict(embed)

        pred = logit.argmax(axis=1)
        clean_acc = sum(pred == label)/len(label)

        logit_tt = reg.predict(embed_tt)
        pred_tt = logit_tt.argmax(axis=1)
        test_acc = sum(pred_tt == label_tt)/len(label_tt)

        return test_acc*100
    
    print("loading model...")
    args, net, epoch, train_acc, test_acc = load_model_from_path(model_path)

    print("loading data...")
    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, clean_ratio=args.clean_ratio, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log="", noise_file=f'./noise_jason/{args.dataset}_{args.noise_mode}_{args.r}_clean_{args.clean_ratio}.json')
    clean_loader, test_loader = loader.run('clean'), loader.run('test')

    print(f"model train acc: {train_acc}%, model test acc: {test_acc}%")

    eval_loader = loader.run('eval_train')
    tr_embed, tr_label = obtain_embeddings_tr(args, eval_loader, net)
    embed_tt, label_tt = obtain_embeddings(args, test_loader, net)    
    clean_embed, small_clean_label = obtain_embeddings(args, clean_loader, net)
    
    prob_logits = eval_train_acc(epoch,net)
    probs, pred_labels = torch.max(prob_logits, dim=1)
    probs = np.asarray(probs)
    pred_labels = np.asarray(pred_labels)

    ground_truth = f'./noise_jason/cifar{num_classes}_sym_0.0_clean_0.1.json'
    clean_label, noise_idx, clean_idx = json.load(open(ground_truth,"r"))
    clean_label = np.asarray(clean_label)

    _, _, clean_idx = json.load(open(loader.noise_file,"r"))
    clean_idx = np.asarray(clean_idx)  

    prob_indices = np.argsort(probs)
    prob_wrong_indices = prob_indices[clean_label[prob_indices] != pred_labels[prob_indices]]
    
    least_conf_accss, top_conf_accss, rand_conf_accss = [], [], []
    least_conf_accss_wrong, top_conf_accss_wrong, rand_conf_accss_wrong = [], [], []

    for seed in range(5):    
        idx = list(range(50000))
        random.shuffle(idx)

        rand_idx = list(range(50000))
        random.shuffle(rand_idx)
  
        least_conf_acc = calc_correct(prob_indices[:num_clean], count=count, idx=idx, use_ridge=use_ridge)
        top_conf_acc = calc_correct(prob_indices[-num_clean:], count=count, idx=idx, use_ridge=use_ridge)
        rand_conf_acc = calc_correct(rand_idx[:num_clean], count=count, idx=idx, use_ridge=use_ridge)

        least_conf_accss.append(least_conf_acc)
        top_conf_accss.append(top_conf_acc)
        rand_conf_accss.append(rand_conf_acc)

    least_y, least_std, least_min, least_max = np.asarray(least_conf_accss).mean(axis=0), np.asarray(least_conf_accss).std(axis=0), np.asarray(least_conf_accss).min(axis=0), np.asarray(least_conf_accss).max(axis=0)
    top_y, top_std, top_min, top_max = np.asarray(top_conf_accss).mean(axis=0), np.asarray(top_conf_accss).std(axis=0), np.asarray(top_conf_accss).min(axis=0), np.asarray(top_conf_accss).max(axis=0)
    
    rand_y, rand_std, rand_min, rand_max = np.asarray(rand_conf_accss).mean(axis=0), np.asarray(rand_conf_accss).std(axis=0), np.asarray(rand_conf_accss).min(axis=0), np.asarray(rand_conf_accss).max(axis=0)

    total_dict = {}
    total_dict['least'] = (least_y, least_std, least_min, least_max)
    total_dict['top'] = (top_y, top_std, top_min, top_max)
    total_dict['rand'] = (rand_y, rand_std, rand_min, rand_max)
    
    print(total_dict)

    pre = model_path.split("/model_epoch")[0]
    save_dir = f"{pre}/total_info"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/epoch{epoch}_num_clean{num_clean}_count{count}.p"
    pickle.dump(total_dict, open(save_path, "wb" ))
    print(f"Stats on predictive power saved to {save_path}")
    return
    
    
def adjust_params(param):
    keys = ["save_model", "checkpoint", "no_log", "correct", "add_clean", "train_only_clean"]
    
    for k in param.copy():
        if k in keys:
            del param[k]
    return param


def create_args_from_log(log):
    parser = argparse.ArgumentParser(description='Calculate predictive power')
    parser.add_argument('--model', type=str, default='ResNet18', help='choose which model')
    parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--noise_mode',  default='sym')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
    parser.add_argument('--clean_ratio', default=0.1, type=float, help='clean_ratio')
    parser.add_argument('--id', default='')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--add_clean', action='store_true', default=False, help='add clean index to dividemix')
    parser.add_argument('--train_only_clean', action='store_true', default=False, help='train only on clean data')
    
    parser.add_argument('--no_log', action='store_true', default=False, help='disables logging of results')
    parser.add_argument('--save_model', action='store_true', default=False, help='store the training models')
    parser.add_argument('--correct', action='store_true', default=False, help='make correction at each epoch')
    parser.add_argument('--checkpoint', action='store_true', default=False, help='checkpoint the training process')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='directory to save the checkpoint files')
    parser.add_argument('--log_interval', type=int, default=10, help='how many epochs to save model')
    parser.add_argument('--filename', type=str, default='', help='the file which store trained model logs')
    parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models logs')

    with open(log, 'r') as f:
        argline = f.readlines()[0]
    param = eval(argline) 
    param = adjust_params(param)
    command = ' '.join('--%s=%s' % (k, param[k]) for k in param)
    args = parser.parse_args(command.split())
    
    return args


def create_model(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    if 'ResNet' in args.model:
        if args.model == 'ResNet18':
            model = ResNet18(num_classes=args.num_class)
        elif args.model == 'ResNet18_wider':
            model = ResNet18_wider(num_classes=args.num_class)
        elif args.model == 'ResNet18_wider2':
            model = ResNet18_wider2(num_classes=args.num_class)
    elif 'CNN' in args.model:
        if args.model == 'CNN9':
            model = CNN9(num_classes=args.num_class)
        elif args.model == 'CNN9_wider':
            model = CNN9_wider(num_classes=args.num_class)
    elif 'MLP' in args.model:
        model = MLP(num_classes=args.num_class)
    model = model.to(device)
    return model


def load_args_from_path(model_path):
    tmp = model_path.replace("models_dir", "results")
    log = tmp.split("/model_epoch")[0]
    args = create_args_from_log(log)
    
    epoch = int(tmp.split("/model_epoch")[1].split('.pth')[0])
    with open(log, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if f"Train Epoch #{epoch}\t Accuracy" in line:
            train_acc = float(line.split("Accuracy: ")[1].split('%')[0])

        if f"Test Epoch #{epoch}\t Accuracy" in line:
            test_acc = float(line.split("Accuracy: ")[1].split('%')[0])
        
    return args, model_path.split('/')[-1], epoch, train_acc, test_acc


def load_model_from_path(model_path):
    if "standard" not in model_path:
        args, model_name, epoch, train_acc, test_acc = load_args_from_path(model_path)
        net1 = create_model(args)
        net2 = create_model(args)

        checkpoint = torch.load(model_path)            
        net1.load_state_dict(checkpoint['state_dict1'])
        net2.load_state_dict(checkpoint['state_dict2'])
        return args, net1, net2, epoch, train_acc, test_acc
    else:
        args, model_name, epoch, train_acc, test_acc = load_args_from_path(model_path)
        net = create_model(args)
        checkpoint = torch.load(model_path)            
        net.load_state_dict(checkpoint['state_dict'])
        return args, net, epoch, train_acc, test_acc

    
def obtain_embeddings(args, data_loader, model):
    model.eval()
    
    embedding_data = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(data_loader):
            inputs, label = inputs.to(device), label.to(device)

            labels.extend(label.cpu().numpy())
            embedding = model.obtain_embedding(inputs)
            embedding_data.append(embedding.detach())
            torch.cuda.empty_cache()
   
    embbedings = torch.cat(embedding_data, dim=0).cpu().numpy()
    return embbedings, np.asarray(labels)    


def obtain_embeddings_tr(args, data_loader, model):
    model.eval()
    
    embedding_data = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, label, index) in enumerate(data_loader):
            inputs, label = inputs.to(device), label.to(device)

            labels.extend(label.cpu().numpy())
            embedding = model.obtain_embedding(inputs)
            embedding_data.append(embedding.detach())
            torch.cuda.empty_cache()
   
    embbedings = torch.cat(embedding_data, dim=0).cpu().numpy()
    return embbedings, np.asarray(labels)   


def calc_test_acc(num_class, epoch, embed_dict):
    print("obtaining embedding...")
    embed1, embed2 = embed_dict['embed1'], embed_dict['embed2'] 
    embed_tt1, embed_tt2 = embed_dict['embed_tt1'], embed_dict['embed_tt2']
    label, label_tt = embed_dict['label'], embed_dict['label_tt']
    y_onehot = torch.zeros(len(label), num_class)
    y_onehot[torch.arange(len(label)), label] = 1
    
    #using linear regression
    reg1, reg2 = LinearRegression().fit(embed1, y_onehot), LinearRegression().fit(embed2, y_onehot)
    logit1, logit2 = reg1.predict(embed1), reg2.predict(embed2)
    logit = logit1+logit2
    pred = logit.argmax(axis=1)
    clean_acc = sum(pred == label)/len(label)
    print(f"Correct Epoch {epoch}: reg clean loader acc is {clean_acc*100:.3f}%")
    
    logit_tt1, logit_tt2 = reg1.predict(embed_tt1), reg2.predict(embed_tt2)
    logit_tt = logit_tt1 + logit_tt2
    pred_tt = logit_tt.argmax(axis=1)
    test_acc = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Correct Epoch {epoch}: reg test acc is {test_acc*100:.3f}% \n")
    return clean_acc, test_acc
    
    
def correct(args, epoch, net1, net2):
    print("obtaining embedding...")
    embed1, _ = obtain_embeddings(args, clean_loader, net1)
    embed2, label = obtain_embeddings(args, clean_loader, net2)
    embed_tt1, _ = obtain_embeddings(args, test_loader, net1)
    embed_tt2, label_tt = obtain_embeddings(args, test_loader, net2)
    y_onehot = torch.zeros(len(label), args.num_class)
    y_onehot[torch.arange(len(label)), label] = 1

    #using linear regression
    reg1, reg2 = LinearRegression().fit(embed1, y_onehot), LinearRegression().fit(embed2, y_onehot)
    logit1, logit2 = reg1.predict(embed1), reg2.predict(embed2)
    logit = logit1+logit2
    pred = logit.argmax(axis=1)
    clean_acc = sum(pred == label)/len(label)
    print(f"Correct Epoch {epoch}: reg clean loader acc is {clean_acc*100:.3f}%")
    
    logit_tt1, logit_tt2 = reg1.predict(embed_tt1), reg2.predict(embed_tt2)
    logit_tt = logit_tt1 + logit_tt2
    pred_tt = logit_tt.argmax(axis=1)
    test_acc = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Correct Epoch {epoch}: reg test acc is {test_acc*100:.3f}% \n")
    return clean_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='calculate embedding')
    parser.add_argument('--model_path', type=str, default='', help='path to the saved model')
    parser.add_argument('--count', type=int, help='number of predictions to use')
    parser.add_argument('--num_clean', type=int, help='number of clean examples to use')
    parser.add_argument('--use_ridge', action='store_true', default=False, help='use ridge regression')
    
    main_args = parser.parse_args()
    
    count, num_clean, use_ridge = main_args.count, main_args.num_clean, main_args.use_ridge
    
    model_path = main_args.model_path
    if 'cifar100' in model_path:
        num_classes = 100
    else:
        num_classes = 10
    
    if 'standard' in main_args.model_path:
        compute_mix_standard(model_path, num_classes, count, num_clean, use_ridge=use_ridge)
    else:
        compute_mix(model_path, num_classes, count, num_clean, use_ridge=use_ridge)


if __name__ == '__main__':
    main()

