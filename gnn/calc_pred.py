import re
import copy
import torch
import numpy as np
import random
import argparse
import pickle
from util import *
from model import GGNN
from main import tensor_data, cvt_data_axis
from os import listdir
from os.path import isfile, join
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

model_types = {'GGNN': GGNN} 


def obtain_embeddings(args, dataset, model):
    model.eval()
    data_size = len(dataset)
    data = cvt_data_axis(dataset)
    bs = args.batch_size
    
    embedding_data = []
    labels = []
    args.batch_size = 64
    for batch_idx in range(int(np.ceil(data_size / args.batch_size))):
        input_nodes, label = tensor_data(data, batch_idx, args)
        labels.extend(label.cpu().numpy())
        embedding = model.obtain_embedding(input_nodes)
        embedding_data.append(embedding.detach())
        torch.cuda.empty_cache()
    embbedings = torch.cat(embedding_data, dim=0).cpu().numpy()
    return embbedings, np.asarray(labels)


def obtain_pred(args, dataset, model):
    model.eval()
    data_size = len(dataset)
    data = cvt_data_axis(dataset)
    bs = args.batch_size
    
    pred_data = []
    labels = []
    args.batch_size = 64
    for batch_idx in range(int(np.ceil(data_size / args.batch_size))):
        input_nodes, label = tensor_data(data, batch_idx, args)
        labels.extend(label.cpu().numpy())
        pred = model(input_nodes)
        pred_data.append(pred.detach())
        torch.cuda.empty_cache()
    preds = torch.cat(pred_data, dim=0).cpu().numpy().flatten()
    return preds, np.asarray(labels)


def metrics(pred, label):
    mse = (np.square(pred - label)).mean()
    l1 = (abs(pred - label)).mean()
    mape = 100*(abs(pred - label)/label).mean()
    return mse, l1, mape


def calc_predictive_power(data, model_path, n_clean, n_orig=0):
    log_path = (model_path.replace("models_dir", "results")).split("/model_")[0]
    args = obtain_args_from_log(log_path)
    train_datasets, validation_datasets, test_datasets = load_data(data)
    true_label = obtain_true_labels(train_datasets, data)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    model = model_types[args.model](args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']-1  
    
    print("obtain embedding...")
    preds, _ = obtain_pred(args, train_datasets, model)
    embed, labels = obtain_embeddings(args, train_datasets, model)
    embed_tt, labels_tt = obtain_embeddings(args, test_datasets, model)
    
    idx = list(range(len(true_label)))
    random.shuffle(idx)
    selected_idx = idx[:n_clean]
    
    if n_orig == 0:
        selected_pred_idx = []
    else:
        selected_pred_idx = idx[-n_orig:]
    
    embed_tr, true_label_tr = embed[selected_idx], true_label[selected_idx]
    embed_pred_tr,  pred_label_tr = embed[selected_pred_idx], preds[selected_pred_idx]
    embed_new, label_new = np.vstack([embed_tr, embed_pred_tr]), np.hstack([true_label_tr, pred_label_tr])
    
    reg = LinearRegression().fit(embed_new, label_new)
    pred_tt = reg.predict(embed_tt)
    mse, l1, mape = metrics(pred_tt, labels_tt)
    
    return mse, l1, mape
    
    
def load_data(data_name):
    with open("./run/%s.txt" %data_name, 'r') as f:
        for line in f:
            if "./data/" in line:
                line = line[7:]
            elif 'data/' in line:
                line = line[5:]
            
            with open("./data/%s" %line.strip(), 'rb') as f2:
                tr, val, test = pickle.load(f2)
    return tr, val, test    


def obtain_true_labels(dataset, data):
    true_label = []
    if  "additive" in data:
        for graph, ans in dataset:
            true_label.append(max([len(n) for n in graph.neighbors]))
    elif "dependent" in data:
        for graph, ans in dataset:
            true_label.append(graph.node_features.max())
    return np.asarray(true_label)


def obtain_args_from_log(log): 
    parser = argparse.ArgumentParser(description='PyTorch Reasoning2')

    #Model specifications
    parser.add_argument('--model', type=str, choices=['GGNN_flex', 'GGNN_E', 'GGNN', 'MLP'], default='GGNN_E', help='choose which model')
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh','linear','sigmoid'], default='relu', help='activation function')
    parser.add_argument('--n_iter', type=int, default=7, help='number of RN/RRN iterations/layers (default: 1)')
    parser.add_argument('--mlp_layer', type=int, default=3, help='number of layers for MLPs in RN/RRN/MLP (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='feature hidden dimension of MLPs (default: 128)')
    parser.add_argument('--fc_output_layer', type=int, default=4, help='number of layers for output(softmax) MLP in RN/RRN/MLP (default: 3)')
    parser.add_argument('--graph_pooling_type', type=str, default="max", choices=["sum", "mean", "max", "min", "max_sum"],
                            help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="max", choices=["sum", "mean", "max", "min"],
                            help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--add_self_loop', action='store_true', 
                        default=False, help='add self loops in case graph does not contain it')

    # Training settings
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, help='resume from model stored')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0.0)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--loss_fn', type=str, choices=['cls', 'reg', 'mape', 'l1_loss'], default='cls', help='classification or regression loss')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], default='Adam', help='Adam or SGD')


    # Logging and storage settings
    parser.add_argument('--log_file', type=str, default='accuracy.log', help='dataset filename')
    parser.add_argument('--save_model', action='store_true', default=False, help='store the training models')
    parser.add_argument('--no_log', action='store_true', default=False, help='disables logging of results')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--filename', type=str, default='', help='the file which store trained model logs')
    parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models logs')
    
    # Data settings
    parser.add_argument('--train', type=str, default='sanity0', help='train index filename')
    parser.add_argument('--val', type=str, default=None, help='test index filename')
    parser.add_argument('--test', type=str, default=None, help='test index filename')
    parser.add_argument('--weight', type=str, default=None, help='the directory to store trained models')
    parser.add_argument('--edge_feature_size', type=int, default=1, help='size of edge features')
    parser.add_argument('--node_feature_size', type=int, default=4, help='size of node features')
    parser.add_argument('--n_objects', default=50, type=int, help='num of objects for fixed graph size')

    args = parser.parse_args("")
    
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        
    with open(log, 'r') as f:
        args_dict = f.readlines()[0]

    param = eval(args_dict)
    param['save_model'] = False
    param = adjust_params(param)
    command = ' '.join('--%s=%s' % (k, param[k]) for k in param)
    args = parser.parse_args(command.split())
    return args    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--n_clean', type=int, default=1000)
    args = parser.parse_args()
    
    mse, l1, mape = calc_predictive_power(args.data, args.model_path, args.n_clean)
    print(f"{args.n_clean}: test l2 loss: {mse:.8f}, l1 loss: {l1:.8f}, mape: {mape:.6f}%")