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

correct_acc = 0.0
flip_type = '2016'

parser = argparse.ArgumentParser(description='CIFAR-10/100 with DivideMix technique')
parser.add_argument('--model', type=str, default='ResNet18', choices=['ResNet18', 'CNN9', 'MLP'], help='choose which model')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym', choices=['sym', 'asym'])
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

parser.add_argument('--no_log', action='store_true', default=False, help='disables logging of results')
parser.add_argument('--save_model', action='store_true', default=False, help='store the training models')
parser.add_argument('--correct', action='store_true', default=False, help='make correction at each epoch')
parser.add_argument('--checkpoint', action='store_true', default=False, help='checkpoint the training process')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='directory to save the checkpoint files')
parser.add_argument('--log_interval', type=int, default=10, help='how many epochs to save model')
parser.add_argument('--filename', type=str, default='', help='the file which store trained model logs')
parser.add_argument('--files_dir', type=str, default='./models_dir', help='the directory to store trained models and logs')
args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(args, epoch, state, file_dir):    
    directory = f"{file_dir}/{args.dataset}/{args.noise_mode}_{args.r}/{args.filename}/"
    sys.stdout.write(f"save dir {directory}\n")
    sys.stdout.flush()
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if args.checkpoint and "checkpoint" in file_dir:
        filename = directory + 'checkpoint.pth.tar'
        torch.save(state, filename)
        return
        
    if epoch == 0 or epoch > args.num_epochs - 10:
        filename = directory + 'model_epoch%d.pth.tar' %epoch
        torch.save(state, filename)
        
    sys.stdout.write(f"model saved to {filename}")
    sys.stdout.flush()

def setup_logs(args, mode):
    file_dir = "results"
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

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.2f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.2f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def eval_train_acc(epoch,net1,net2):    
    net1.eval()
    net2.eval()
    correct = 0
    total = 0  
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1) 
            
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    logging.info("Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))
        
def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  
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
    
def correct(epoch, net1, net2):
    global correct_acc 
    
    #using linear regression
    embed1, _ = obtain_embeddings(args, clean_loader, net1)
    embed2, label = obtain_embeddings(args, clean_loader, net2)
    embed_tt1, _ = obtain_embeddings(args, test_loader, net1)
    embed_tt2, label_tt = obtain_embeddings(args, test_loader, net2)
    y_onehot = torch.zeros(len(label), args.num_class)
    y_onehot[torch.arange(len(label)), label] = 1

    reg1, reg2 = LinearRegression().fit(embed1, y_onehot), LinearRegression().fit(embed2, y_onehot)
    logit1, logit2 = reg1.predict(embed1), reg2.predict(embed2)
    logit = logit1+logit2
    pred = logit.argmax(axis=1)
    acc = sum(pred == label)/len(label)
    # print(f"Correct Epoch {epoch}: reg clean loader acc is {acc*100:.3f}%")
    # logging.info(f"Correct Epoch {epoch}: reg clean loader acc is {acc*100:.3f}%")
    
    logit_tt1, logit_tt2 = reg1.predict(embed_tt1), reg2.predict(embed_tt2)
    logit_tt = logit_tt1 + logit_tt2
    pred_tt = logit_tt.argmax(axis=1)
    acc = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Epoch {epoch}: predictive power accuracy is {acc*100:.3f}% \n")
    logging.info(f"Epoch {epoch}: predictive power accuracy is {acc*100:.3f}% \n")

    clf1, clf2 = Ridge(alpha=1.0).fit(embed1, y_onehot), Ridge(alpha=1.0).fit(embed2, y_onehot)
    logit_tt1, logit_tt2 = clf1.predict(embed_tt1), clf2.predict(embed_tt2)
    logit_tt = logit_tt1 + logit_tt2
    pred_tt = logit_tt.argmax(axis=1)
    acc = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Epoch {epoch}: predictive power (ridge) accuracy is {acc*100:.3f}% \n")
    logging.info(f"Epoch {epoch}: predictive power (ridge) accuracy is {acc*100:.3f}% \n")
    
    correct_acc = 100*acc

def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

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

if args.r == 0.9:
    args.p_threshold = 0.6

stats_log=open('./results/%s_%.2f_%s_clean%.2f_%s'%(args.dataset,args.r,args.noise_mode,args.clean_ratio, args.model)+'_stats.txt','w') 
test_log=open('./results/%s_%.2f_%s_clean%.2f_%s'%(args.dataset,args.r,args.noise_mode,args.clean_ratio,args.model)+'_acc.txt','w')     

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, clean_ratio=args.clean_ratio, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log=stats_log, flip_type=flip_type, noise_file=f'./noise_jason/{args.dataset}_{flip_type}_{args.noise_mode}_{args.r}_clean_{args.clean_ratio}.json', add_clean=args.add_clean)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

args.filename = f"{args.model}_lr{args.lr}_lambda{args.lambda_u}_clean{args.clean_ratio}_epochs{args.num_epochs}_seed{args.seed}_fliptype{flip_type}_addclean{args.add_clean}.log"
start_epoch = 0
mode = 'w+'
if args.checkpoint:
    checkpth = f"{args.checkpoint_dir}/{args.dataset}/{args.noise_mode}_{args.r}/{args.filename}/checkpoint.pth.tar"
    # load checkpoint if exisits
    if os.path.exists(checkpth):
        checkpoint = torch.load(checkpth)
        start_epoch = checkpoint['epoch'] + 1
            
        net1.load_state_dict(checkpoint['state_dict1'])
        net2.load_state_dict(checkpoint['state_dict2'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        mode = 'a+'
        print(f"checkpoint loaded epoch {checkpoint['epoch']}!") 

setup_logs(args, mode)              
            
if args.save_model and start_epoch == 0 and False:
    save_checkpoint(args, 0, {
                        'epoch': 0,
                        'args': args, 
                        'state_dict1': net1.state_dict(),
                        'state_dict2': net2.state_dict(),
                        'optimizer1' : optimizer1.state_dict(),
                        'optimizer2' : optimizer2.state_dict(),
    }, "./models_dir")   
    
if args.correct:
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    clean_loader = loader.run('clean')
    correct(start_epoch, net1, net2)
    
for epoch in range(start_epoch, args.num_epochs+1):                 
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    clean_loader = loader.run('clean')
    
    #if args.clean_only: 
    #    eval_loader = loader.run('clean')

    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        #if args.clean_only: 
        #    warmup_trainloader = loader.run('clean')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:         
        prob1,all_loss[0]=eval_train(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(net2,all_loss[1])          
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        stats_log.write("Train Net1\n")
        stats_log.flush() 
        if sum(pred2) > 0:
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide    
            train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        stats_log.write('Train Net2\n')
        stats_log.flush() 
        if sum(pred1) > 0:
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
            train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2         

    eval_train_acc(epoch,net1,net2)
    test(epoch,net1,net2)
    if args.correct:
        correct(epoch, net1, net2)
    
    # save the model from last 10 epochs
    if args.save_model and epoch > args.num_epochs - 10: 
        save_checkpoint(args, epoch, {
                    'epoch': epoch,
                    'args': args, 
                    'state_dict1': net1.state_dict(),
                    'state_dict2': net2.state_dict(),
                    'optimizer1' : optimizer1.state_dict(),
                    'optimizer2' : optimizer2.state_dict(),
         }, "./models_dir")            
              
    if args.checkpoint:
        save_checkpoint(args, epoch, {
                    'epoch': epoch,
                    'args': args, 
                    'state_dict1': net1.state_dict(),
                    'state_dict2': net2.state_dict(),
                    'optimizer1' : optimizer1.state_dict(),
                    'optimizer2' : optimizer2.state_dict(),
         }, args.checkpoint_dir)
            
