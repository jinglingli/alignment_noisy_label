from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
import logging
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--id', default='',type=str)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='/scratch/ssd002/datasets/webvision/', type=str, help='path to dataset')

parser.add_argument('--no_log', action='store_true', default=False, help='disables logging of results')
parser.add_argument('--save_model', action='store_true', default=False, help='store the training models')
parser.add_argument('--checkpoint', action='store_true', default=False, help='checkpoint the training process')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='directory to save the checkpoint files')
parser.add_argument('--log_interval', type=int, default=1, help='how many epochs to save model')
parser.add_argument('--filename', type=str, default='', help='the file which store trained model logs')
parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models logs')

args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(args, epoch, state, file_dir):
    directory = f"{file_dir}/webvision/{args.filename}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if args.checkpoint and "checkpoint" in file_dir:
        filename = directory + 'checkpoint.pth.tar'
        torch.save(state, filename)
        return
        
    if epoch%args.log_interval == 0:
        filename = directory + 'model_epoch%d.pth.tar' %epoch
        torch.save(state, filename)
        
    print(f"model saved to {filename}")

def setup_logs(args, mode):
    file_dir = "results"
    if not args.no_log:
        files_dir = '%s/webvision' %(file_dir)
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

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

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

        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t Labeled loss: %.2f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)   
        
        #penalty = conf_penalty(outputs)
        L = loss #+ penalty      

        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
                
def test(epoch,net1,net2,test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)                 
            acc_meter.add(outputs,targets)
    accs = acc_meter.value()
    return accs

def eval_train(model,all_loss): 
    print("eval_train")
    model.eval()
    num_iter = (len(eval_loader.dataset)//eval_loader.batch_size)+1
    #print("num_iter")
    losses = torch.zeros(len(eval_loader.dataset))    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            #print("batch_idx")
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]       
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' %(batch_idx,num_iter)) 
            sys.stdout.flush()    
                                    
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1,1)
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
    model = InceptionResNetV2(num_classes=args.num_class)
    model = model.cuda()
    return model

def obtain_embedding(args, data_loader, model):
    model.eval()

    embedding_data = []
    labels_all = []
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        #labels = torch.zeros(len(targets), 50)
        #labels[torch.arange(len(targets)), targets] = 1
        #labels = labels.cuda()
            
        if torch.cuda.device_count() > 1:
            embed = model.module.obtain_embedding(inputs)
        else:
            embed = model.obtain_embedding(inputs)
        
        embedding_data.append(embed.detach())
        labels_all.extend(targets.numpy())
        
        torch.cuda.empty_cache() 
   
    embbedings = torch.cat(embedding_data, dim=0).cpu().numpy()
    return embbedings, np.asarray(labels_all)  

stats_log=open('./checkpoint/%s'%(args.id)+'_stats.txt','w') 
test_log=open('./checkpoint/%s'%(args.id)+'_acc.txt','w')     

warm_up=1

loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_workers=5,root_dir=args.data_path,log=stats_log, num_class=args.num_class)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net1 = nn.DataParallel(net1)
    net2 = nn.DataParallel(net2)

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True) 

args.filename = f"InceptionResNetV2_lr{args.lr}_epochs{args.num_epochs}_seed{args.seed}_bs{args.batch_size}.log"
start_epoch = 0
mode = 'w+'
if args.checkpoint:
    checkpth = f"{args.checkpoint_dir}/webvision/{args.filename}/checkpoint.pth.tar"
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
        epoch = checkpoint['epoch']
        eval_loader = loader.run('eval_train')  
        web_valloader = loader.run('test')
        imagenet_valloader = loader.run('imagenet') 

        if start_epoch < args.num_epochs:
            print('\n==== net 1 evaluate training data loss ====') 
            prob1,all_loss[0]=eval_train(net1,all_loss[0])   
            print('\n==== net 2 evaluate training data loss ====') 
            prob2,all_loss[1]=eval_train(net2,all_loss[1])
            torch.save(all_loss,'./checkpoint/%s.pth.tar'%(args.id)) 
        
        web_acc = test(epoch,net1,net2,web_valloader)  
        imagenet_acc = test(epoch,net1,net2,imagenet_valloader)  
  
        print("Checkpoint Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(checkpoint['epoch'],web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))

setup_logs(args, mode)   

print(args.save_model, start_epoch)
if args.save_model and start_epoch == 0:
    save_checkpoint(args, 0, {
                        'epoch': 0,
                        'args': args, 
                        'state_dict1': net1.state_dict(),
                        'state_dict2': net2.state_dict(),
                        'optimizer1' : optimizer1.state_dict(),
                        'optimizer2' : optimizer2.state_dict(),
    }, "./models_dir")   
             
for epoch in range(start_epoch, args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr              
    eval_loader = loader.run('eval_train')  
    web_valloader = loader.run('test')
    imagenet_valloader = loader.run('imagenet')   
              
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:                
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2    
   
    web_acc = test(epoch,net1,net2,web_valloader)  
    imagenet_acc = test(epoch,net1,net2,imagenet_valloader)  
    
    logging.info("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    
    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))  
    test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    test_log.flush()  

    print('\n==== net 1 evaluate training data loss ====') 
    prob1,all_loss[0]=eval_train(net1,all_loss[0])   
    print('\n==== net 2 evaluate training data loss ====') 
    prob2,all_loss[1]=eval_train(net2,all_loss[1])
    torch.save(all_loss,'./checkpoint/%s.pth.tar'%(args.id)) 
    
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

