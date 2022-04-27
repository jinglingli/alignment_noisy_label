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
import numpy as np
from PreResNet import *
from model import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

loss_fn = 'cls'

correct_acc = 0.0

parser = argparse.ArgumentParser(description='CIFAR-10/100 vanilla training')
parser.add_argument('--model', type=str, default='ResNet18', help='choose which model')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--clean_ratio', default=0.1, type=float, help='clean_ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=122)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)

parser.add_argument('--no_log', action='store_true', default=False, help='disables logging of results')
parser.add_argument('--save_model', action='store_true', default=False, help='store the training models')
parser.add_argument('--correct', action='store_true', default=False, help='make correction at each epoch')
parser.add_argument('--train_only_clean', action='store_true', default=False, help='train only on the clean subset of data')
parser.add_argument('--checkpoint', action='store_true', default=False, help='checkpoint the training process')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='directory to save the checkpoint files')
parser.add_argument('--log_interval', type=int, default=100, help='how many epochs to save model')
parser.add_argument('--filename', type=str, default='', help='the file which store trained model logs')
parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models logs')
args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(args, epoch, state, file_dir):    
    directory = f"./standard/{file_dir}/{args.dataset}/{args.noise_mode}_{args.r}/{args.filename}/"
    sys.stdout.write(f"save dir {directory}")
    sys.stdout.flush()
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if args.checkpoint and "checkpoint" in file_dir:
        filename = directory + 'checkpoint.pth.tar'
        torch.save(state, filename)
        return
        
    if (epoch%args.log_interval == 0) or (epoch > args.num_epochs - 10):
        filename = directory + 'model_epoch%d.pth.tar' %epoch
        torch.save(state, filename)
        
    sys.stdout.write(f"model saved to {filename}")
    sys.stdout.flush()

def setup_logs(args, mode):
    file_dir = "./standard/results"
    if not args.no_log:
        files_dir = '%s/%s/%s_%s' %(file_dir, args.dataset, args.noise_mode, args.r)
        args.files_dir = files_dir

        if not os.path.exists(files_dir):
            os.makedirs(files_dir)

        logging.basicConfig(format='%(message)s',
                            level=logging.INFO,
                            datefmt='%m-%d %H:%M',
                            filename="%s/%s" %(args.files_dir, args.filename),
                            filemode=mode)

        print(vars(args))
        if mode == 'w+':
            logging.info(vars(args))       


def train(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    correct = 0
    total = 0  
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = net(inputs) 
        
        if loss_fn != 'reg':
            loss = CEloss(outputs, labels)
        else:
            labels_onehot = torch.zeros(len(labels), num_class)
            labels_onehot[torch.arange(len(labels)), labels] = 1
            loss = MSEloss(outputs, labels_onehot)
        
        _, predicted = torch.max(outputs, 1)       
        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()
            
        loss.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.2f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    logging.info("Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))

def train_clean(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    correct = 0
    total = 0  
    for batch_idx, (inputs, labels) in enumerate(dataloader):      
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = net(inputs) 
        
        if loss_fn != 'reg':
            loss = CEloss(outputs, labels)
        else:
            labels_onehot = torch.zeros(len(labels), num_class)
            labels_onehot[torch.arange(len(labels)), labels] = 1
            labels_onehot = labels_onehot.to(device)
            loss = MSEloss(outputs, labels_onehot)
        
        _, predicted = torch.max(outputs, 1)       
        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()
            
        loss.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.2f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    logging.info("Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))

def test(epoch,net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))   
    logging.info("Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))

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
    
def correct(epoch, net):
    global correct_acc 
    
    #using linear regression
    embed, label = obtain_embeddings(args, clean_loader, net)
    embed_tt, label_tt = obtain_embeddings(args, test_loader, net)
    y_onehot = torch.zeros(len(label), args.num_class)
    y_onehot[torch.arange(len(label)), label] = 1

    reg = LinearRegression().fit(embed, y_onehot)
    logit = reg.predict(embed)
    pred = logit.argmax(axis=1)
    acc = sum(pred == label)/len(label)
    #print(f"Correct Epoch {epoch}: reg clean loader acc is {acc*100:.3f}%")
    #logging.info(f"Correct Epoch {epoch}: reg clean loader acc is {acc*100:.3f}%")
    
    logit_tt = reg.predict(embed_tt)
    pred_tt = logit_tt.argmax(axis=1)
    acc = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Epoch {epoch}: predictive power accuracy is {acc*100:.3f}% \n")
    logging.info(f"Epoch {epoch}: predictive power accuracy is {acc*100:.3f}% \n")

    clf = Ridge(alpha=1.0).fit(embed, y_onehot)
    logit_tt = clf.predict(embed_tt)
    pred_tt = logit_tt.argmax(axis=1)
    acc = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Epoch {epoch}: predictive power (ridge) accuracy is {acc*100:.3f}% \n")
    logging.info(f"Epoch {epoch}: predictive power (ridge) accuracy is {acc*100:.3f}% \n")
    
    correct_acc = 100*acc

def create_model():
    if 'ResNet' in args.model:
        model = ResNet18(num_classes=args.num_class)
    elif 'CNN' in args.model:
        model = CNN9(num_classes=args.num_class)
    elif 'MLP' in args.model:
        model = MLP(num_classes=args.num_class)
    model = model.to(device)
    return model

if args.dataset == 'cifar10':
    args.num_class = 10
else:
    args.num_class = 100

num_class = args.num_class

stats_log=open('./standard/results/%s_%.2f_%s_clean%.2f_%s'%(args.dataset,args.r,args.noise_mode,args.clean_ratio, args.model)+'_stats.txt','w') 

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, clean_ratio=args.clean_ratio, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log=stats_log, noise_file=f'./noise_jason/{args.dataset}_{args.noise_mode}_{args.r}_clean_{args.clean_ratio}.json')

print('| Building net')
model = create_model()
cudnn.benchmark = True
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
MSEloss = nn.MSELoss()

all_loss = [[],[]] # save the history of losses from two networks

args.filename = f"{args.model}_loss{loss_fn}_lr{args.lr}_cleanonly{args.train_only_clean}_clean{args.clean_ratio}_epochs{args.num_epochs}_seed{args.seed}.log"
start_epoch = 0
mode = 'w+'
if args.checkpoint:
    checkpth = f"./standard/{args.checkpoint_dir}/{args.dataset}/{args.noise_mode}_{args.r}/{args.filename}/checkpoint.pth.tar"
    # load checkpoint if exisits
    print(checkpth)
    if os.path.exists(checkpth):
        checkpoint = torch.load(checkpth)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])        
        optimizer.load_state_dict(checkpoint['optimizer'])
        mode = 'a+'
        print(f"checkpoint loaded epoch {checkpoint['epoch']}!") 

setup_logs(args, mode)              

print(f"{args.model} number of parameters: {sum([p.data.nelement() for _, p in model.named_parameters()])}")
if mode == 'w+':
    logging.info(f"{args.model} number of parameters: {sum([p.data.nelement() for _, p in model.named_parameters()])}")

if args.save_model and start_epoch == 0:
    save_checkpoint(args, 0, {
                        'epoch': 0,
                        'args': args, 
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
    }, "./models_dir")   

test_loader = loader.run('test')
eval_loader = loader.run('eval_train')
clean_loader = loader.run('clean')

if args.num_epochs == 0:
    epoch = 0
    test(epoch,model)

    if args.correct:
        correct(epoch, model)
        
    start_epoch = 1
        
for epoch in range(start_epoch, args.num_epochs+1):                 
    lr=args.lr
    lr = lr * (0.99)**epoch      
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr       
    
    if args.train_only_clean:
        trainloader = loader.run('clean')
        train_clean(epoch,model,optimizer,trainloader)
    else:
        trainloader = loader.run('warmup')
        train(epoch,model,optimizer,trainloader)
    
    test(epoch,model)
    
    if args.correct:
        correct(epoch, model)
    
    print(epoch, args.num_epochs - 10, args.save_model and epoch > args.num_epochs - 10)
    if args.save_model and epoch > args.num_epochs - 10: 
        save_checkpoint(args, epoch, {
                      'epoch': epoch,
                      'args': args, 
                      'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
         }, "models_dir")            
              
    if args.checkpoint:
        save_checkpoint(args, epoch, {
                      'epoch': epoch,
                      'args': args, 
                      'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
         }, args.checkpoint_dir)
            
