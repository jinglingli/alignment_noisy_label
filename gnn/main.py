import os
import argparse
import pickle
import random
import torch
import numpy as np
import logging
from util import *
from model import GGNN

best_prec, best_loss, best_test_loss, best_test_mape, best_model_test_acc, best_model_test_loss, best_model_mape_loss = 0.0, 1e+8*1.0, 1e+8*1.0, 1e+8*1.0, 0.0, 1e+8*1.0, 1e+8*1.0
is_best, is_best_test = False, False
val_acc, best_val_acc = 0.0, 0.0
val_loss, val_mape, best_val_loss, best_val_mape = 1e+8*1.0, 1e+8*1.0, 1e+8*1.0, 1e+8*1.0
best_epoch, best_test_loss_epoch = 0, 0

model_types = {'GGNN': GGNN} 

def save_checkpoint(state, is_best, epoch, args):
    global is_best_test
    
    directory = "models_dir/Train_%s/Test_%s/%s/"%(args.train, args.test, args.filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = directory + 'model_epoch%d.pth.tar' %epoch
    torch.save(state, filename)

    """Saves checkpoint to disk"""
    if is_best:
        filename = directory + 'model_best.pth.tar' 
        torch.save(state, filename)
        
    if is_best_test:
        filename = directory + 'model_best_test.pth.tar' 
        torch.save(state, filename)

def cvt_data_axis(dataset):
    data = []
    label = []   
    for d, ans in dataset:
        data.append(d)
        label.append(ans)    
    return (data, label)

def tensor_data(data, i, args):
    nodes = data[0][args.batch_size*i:args.batch_size*(i+1)]
    if args.loss_fn == 'cls':
        ans = torch.LongTensor(data[1][args.batch_size*i:args.batch_size*(i+1)]).to(args.device)
    else: 
        ans = torch.FloatTensor(data[1][args.batch_size*i:args.batch_size*(i+1)]).to(args.device)
    return nodes, ans

def obtain_embeddings(args, dataset, model):
    model.eval()
    data_size = len(dataset)
    print(data_size)
    data = cvt_data_axis(dataset)
    bs = args.batch_size
    
    embedding_data = []
    labels = []
    for batch_idx in range(int(np.ceil(data_size / bs))):
        input_nodes, label = tensor_data(data, batch_idx, args)
        labels.extend(label)
        embedding = model.obtain_embedding(input_nodes)
        embedding_data.append(embedding.detach())
        torch.cuda.empty_cache()
    embbedings = torch.cat(embedding_data, dim=0)
    new_dataset = [(embbedings[i].unsqueeze(0), labels[i]) for i in range(data_size)]
    return new_dataset

def train(epoch, dataset, args, model):
    model.train()
    train_size = len(dataset)
    bs = args.batch_size
    random.shuffle(dataset)
    
    data = cvt_data_axis(dataset)
    
    running_loss, running_loss_mape = 0.0, 0.0
    accuracys = []
    losses, losses_mape = [], []
    batch_runs = int(np.ceil(train_size / bs))
    for batch_idx in range(batch_runs):
        input_nodes, label = tensor_data(data, batch_idx, args)
        accuracy, loss, mape_loss = model.train_(input_nodes, label)
        running_loss += loss
        running_loss_mape += mape_loss        
        accuracys.append(accuracy)
        losses.append(loss)
        losses_mape.append(mape_loss)
        torch.cuda.empty_cache()
        
        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)] accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(epoch, batch_idx * bs, train_size, 100 * batch_idx * bs / train_size, accuracy, running_loss/(1 * args.log_interval), running_loss_mape/(1 * args.log_interval)))
            logging.info('Train Epoch: {} [{}/{} ({:.2f}%)] accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(epoch, batch_idx * bs, train_size, 100 * batch_idx * bs / train_size, accuracy, running_loss/(1 * args.log_interval), running_loss_mape/(1 * args.log_interval)))
            running_loss, running_loss_mape = 0.0, 0.0            
    
    avg_accuracy = sum(accuracys) *1.0 / len(accuracys)
    avg_losses = sum(losses) *1.0 / len(losses)
    avg_losses_mape = sum(losses_mape) *1.0 / len(losses_mape)
    
    print('\nTrain set: accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(avg_accuracy, avg_losses, avg_losses_mape))
    logging.info('\nTrain set: accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(avg_accuracy, avg_losses, avg_losses_mape))

def validate(epoch, dataset, args, model):
    global is_best, best_prec, best_loss, val_acc, val_loss, val_mape
    
    model.eval()
    data_size = len(dataset)
    bs = args.batch_size
    
    random.shuffle(dataset) 
    data = cvt_data_axis(dataset)

    accuracys = []
    losses, losses_mape = [], []
    for batch_idx in range(int(np.ceil(data_size / bs))):
        input_nodes, label = tensor_data(data, batch_idx, args)
        accuracy, loss, mape_loss = model.test_(input_nodes, label)
        accuracys.append(accuracy)
        losses.append(loss)
        losses_mape.append(mape_loss)
        torch.cuda.empty_cache()

    avg_accuracy = sum(accuracys) *1.0 / len(accuracys)
    avg_losses = sum(losses) *1.0 / len(losses)
    avg_losses_mape = sum(losses_mape) *1.0 / len(losses_mape)

    print('Validation set: accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(avg_accuracy, avg_losses, avg_losses_mape))
    logging.info('Validation set: accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(avg_accuracy, avg_losses, avg_losses_mape))
    
    val_acc, val_loss, val_mape = avg_accuracy, avg_losses, avg_losses_mape
    
    if args.loss_fn == 'cls':
        is_best = avg_accuracy > best_prec
    else: #lif args.loss_fn == 'reg':
        is_best = avg_losses < best_loss
    best_prec = max(avg_accuracy, best_prec)
    best_loss = min(avg_losses, best_loss)

def test(epoch, dataset, args, model):
    global is_best, is_best_test, best_model_test_acc, best_model_test_loss, best_epoch, best_model_mape_loss, val_acc, val_loss, val_mape, best_val_loss, best_val_mape, best_val_acc, best_test_loss, best_test_mape, best_test_loss_epoch
    
    model.eval()
    data_size = len(dataset)
    bs = args.batch_size
    
    random.shuffle(dataset)    
    data = cvt_data_axis(dataset)
    
    accuracys = []
    losses, losses_mape = [], []
    args.batch_size = 128
    for batch_idx in range(int(np.ceil(data_size / args.batch_size))):
        input_nodes, label = tensor_data(data, batch_idx, args)
        accuracy, loss, mape_loss = model.test_(input_nodes, label)
        accuracys.append(accuracy)
        losses.append(loss)
        losses_mape.append(mape_loss)
        torch.cuda.empty_cache()
    
    args.batch_size = bs
    avg_accuracy = sum(accuracys) *1.0 / len(accuracys)
    avg_losses = sum(losses) *1.0 / len(losses)
    avg_losses_mape = sum(losses_mape) *1.0 / len(losses_mape)

    print('Test set: accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f} \n'.format(avg_accuracy, avg_losses, avg_losses_mape))
    logging.info('Test set: accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f} \n'.format(avg_accuracy, avg_losses, avg_losses_mape))
       
    if is_best:
        best_model_test_acc = avg_accuracy
        best_model_test_loss = avg_losses
        best_model_mape_loss = avg_losses_mape
        best_epoch = epoch
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_val_mape = val_mape
    
    if best_test_loss >  avg_losses:
        best_test_loss_epoch = epoch
        best_test_loss = avg_losses
        best_test_mape = avg_losses_mape
        is_best_test = True
    else:
        is_best_test = False
    
    if epoch%10 == 0:
        logtext = '************ Best model\'s test accuracy: {:.2f}%, test loss: {:.7f}, mape: {:.7f} (best model is from epoch {} with val information: [{:.2f}%, {:.7f}, {:.7f}]) ************\n'.format(best_model_test_acc, best_model_test_loss, best_model_mape_loss, best_epoch, best_val_acc, best_val_loss, best_val_mape)
        print(logtext)
        logging.info(logtext)

def load_data(index_filename, mode):
    with open("./run/%s.txt" %index_filename, 'r') as f:
        for line in f:
            with open(line.strip(), 'rb') as f2:
                dataset = pickle.load(f2)[mode]
    return dataset

def setup_logs(args):
    file_dir = "results"
    if not args.no_log:
        files_dir = '%s/Train_%s/Test_%s' %(file_dir, args.train, args.test)
        args.files_dir = files_dir
        args.filename = '%s_loss%s_%s_lr%s_decay%s_hdim%s_fc%s_mlp%s_%s_%s_bs%s_epoch%d_seed%d.log' \
            %(args.model, args.loss_fn, args.n_iter, args.lr, args.decay, args.hidden_dim, args.fc_output_layer, args.mlp_layer, 
              args.graph_pooling_type, args.neighbor_pooling_type, args.batch_size, args.epochs, args.seed)

        if not os.path.exists(files_dir):
            os.makedirs(files_dir)
        mode = 'w+'
        if args.resume:
            mode = 'a+'
        logging.basicConfig(format='%(message)s',
                            level=logging.INFO,
                            datefmt='%m-%d %H:%M',
                            filename="%s/%s" %(args.files_dir, args.filename),
                            filemode='w+')

        print(vars(args))
        logging.info(vars(args))

def resume(args, model):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        logging.info("=> loading checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)

        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        best_model_test_acc = checkpoint['best_model_test_acc']
        best_model_test_loss = checkpoint['best_model_test_loss']
        best_model_mape_loss = checkpoint['best_model_mape_loss']
        model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch Reasoning2')

    #Model specifications
    parser.add_argument('--model', type=str, choices=['GGNN_E', 'GGNN', 'MLP', 'Skip', 'MLP2', 'RegFC'], default='GGNN_E', help='choose which model')
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh','linear','sigmoid'], default='relu', help='activation function')
    parser.add_argument('--n_iter', type=int, default=7, help='number of RN/RRN iterations/layers (default: 1)')
    parser.add_argument('--mlp_layer', type=int, default=3, help='number of layers for MLPs in RN/RRN/MLP (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='feature hidden dimension of MLPs (default: 128)')
    parser.add_argument('--fc_output_layer', type=int, default=4, help='number of layers for output(softmax) MLP in RN/RRN/MLP (default: 3)')
    parser.add_argument('--graph_pooling_type', type=str, default="max", choices=["sum", "mean", "max", "min"],
                            help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="max", choices=["sum", "mean", "max", "min"],
                            help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--add_self_loop', action='store_true', 
                        default=False, help='add self loops in case graph does not contain it')

    # Training settings
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, help='resume from model stored')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0.0)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--loss_fn', type=str, choices=['cls', 'reg', 'mape', 'l1_loss'], default='reg', help='classification or regression loss')
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

    args = parser.parse_args()
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.val == None: 
        args.val = args.train
    if args.test == None: 
        args.test = args.train
    
    train_datasets = load_data(args.train, 0)
    validation_datasets = load_data(args.val, 1)
    test_datasets = load_data(args.test, 2)
    
    args.node_feature_size = len(train_datasets[0][0].node_features[0])

    setup_logs(args)
        
    model = model_types[args.model](args).to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=50, gamma=0.5)
    bs = args.batch_size
    
    models_dir = './models_dir'
    try:
        os.makedirs(models_dir)
    except:
        print('directory {} already exists'.format(models_dir))

    if args.epochs == 0:
        epoch = 0
        validate(epoch, validation_datasets, args, model)
        test(epoch, test_datasets, args, model)
        args.epochs = -1

    if args.save_model:
        save_checkpoint({
                        'epoch': 0,
                        'arch': args.model,
                        'args': args, 
                        'state_dict': model.state_dict(),
                        'optimizer' : model.optimizer.state_dict(),
                    }, is_best, 0, args)    
        
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_datasets, args, model)
        validate(epoch, validation_datasets, args, model)
        test(epoch, test_datasets, args, model)
        scheduler.step()
        if args.save_model and (is_best or is_best_test): 
            save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'args': args, 
                    'state_dict': model.state_dict(),
                    'best_prec': best_prec,
                    'best_model_test_acc': best_model_test_acc,
                    'best_model_test_loss': best_model_test_loss,
                    'best_model_mape_loss': best_model_mape_loss,
                    'optimizer' : model.optimizer.state_dict(),
                }, is_best, epoch, args)
        
    logtext = '************ Best model\'s test accuracy: {:.2f}%, test loss: {:.7f}, mape: {:.7f} (best model is from epoch {} with val information: [{:.2f}%, {:.7f}, {:.7f}]) best_test_loss: {:.7f}, {:.7f} epoch {}************\n'.format(best_model_test_acc, best_model_test_loss, best_model_mape_loss, best_epoch, best_val_acc, best_val_loss, best_val_mape, best_test_loss, best_test_mape, best_test_loss_epoch)
    print(logtext)
    logging.info(logtext)

if __name__ == '__main__':
    main()
