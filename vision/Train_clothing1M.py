from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
#import torchvision.models as models
from resnet import *
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture
import logging
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=16, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='../dataset/clothing1M/images', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)

parser.add_argument('--trained_log', type=str, default='resnet50_lr0.002_epochs80_seed123.log', help='path to the trained models')
parser.add_argument('--no_log', action='store_true', default=False, help='disables logging of results')
parser.add_argument('--save_model', action='store_true', default=False, help='store the training models')
parser.add_argument('--add_clean', action='store_true', default=False, help='store the training models')
parser.add_argument('--fine_tune', action='store_true', default=False, help='store the training models')
parser.add_argument('--checkpoint', action='store_true', default=False, help='checkpoint the training process')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='directory to save the checkpoint files')
parser.add_argument('--log_interval', type=int, default=80, help='how many epochs to save model')
parser.add_argument('--filename', type=str, default='', help='the file which store trained model logs')
parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models logs')

args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

model_best_epoch = 0

def save_checkpoint(args, epoch, state, file_dir):
    directory = f"{file_dir}/clothing1M/{args.filename}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if args.checkpoint and "checkpoint" in file_dir:
        filename = directory + 'checkpoint.pth.tar'
        torch.save(state, filename)
        return
        
    if epoch%args.log_interval == 0 or epoch > args.num_epochs-10:
        filename = directory + 'model_epoch%d.pth.tar' %epoch
        torch.save(state, filename)
        
    print(f"model saved to {filename}")

def setup_logs(args, mode):
    file_dir = "results"
    if not args.no_log:
        files_dir = '%s/clothing1M' %(file_dir)
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
    losses = []
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
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()
        losses.append(Lx.item())
        
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        sys.stdout.flush()
    
def finetune(net,optimizer,dataloader):
    net.train()
    size = 50000//args.batch_size
    for batch_idx, (inputs, labels) in enumerate(dataloader):      
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = 0 #conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Fine-tuning: Iter[%3d/%3d]\t CE-loss: %.4f '
                %(batch_idx+1, size, L.item()))
        sys.stdout.flush()    
    
def val(net,val_loader,k):
    global model_best_epoch
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()   
            #torch.cuda.empty_cache()
    acc = 100.*correct/total
    print("\n| Epoch #%d: Validation\t Net%d  Acc: %.2f%%" %(epoch, k,acc))  
    logging.info("\n| Epoch #%d: Validation\t Net%d  Acc: %.2f%%" %(epoch, k,acc))  
    if acc > best_acc[k-1]:
        model_best_epoch = epoch
        best_acc[k-1] = acc
        print('| Saving Best Net%d from epoch %d...'%(k, model_best_epoch))
        logging.info('| Saving Best Net%d from epoch %d...'%(k, model_best_epoch))
        save_point = './checkpoint/%s_net%d.pth.tar'%(args.filename,k)
        torch.save(net.state_dict(), save_point)
    return acc

def test(net1,net2,test_loader):
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
            #torch.cuda.empty_cache()
    acc = 100.*correct/total
    print("\n| Epoch #%d:  Test Acc: %.2f%%\n" %(epoch, acc))  
    logging.info("\n| Epoch #%d:  Test Acc: %.2f%%\n" %(epoch, acc))  
    return acc    

def obtain_embeddings(args, data_loader, model):
    model.eval()
    
    embedding_data = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print("batch_idx", batch_idx)
            inputs, label = inputs.to(device), label.to(device)

            labels.extend(label.cpu().numpy())
            if torch.cuda.device_count() > 1:
                embedding = model.module.obtain_embedding(inputs)
            else:
                embedding = model.obtain_embedding(inputs)
            embedding_data.append(embedding.detach())
            #torch.cuda.empty_cache()
   
    embbedings = torch.cat(embedding_data, dim=0).cpu().numpy()
    return embbedings, np.asarray(labels)    

def correct(epoch, net1, net2):   
    #using linear regression
    embed1, _ = obtain_embeddings(args, clean_loader, net1)
    embed2, label = obtain_embeddings(args, clean_loader, net2)
    
    y_onehot = torch.zeros(len(label), args.num_class)
    y_onehot[torch.arange(len(label)), label] = 1
    
    reg1, reg2 = LinearRegression().fit(embed1, y_onehot), LinearRegression().fit(embed2, y_onehot)
    logit1, logit2 = reg1.predict(embed1), reg2.predict(embed2)
    logit = logit1+logit2
    pred = logit.argmax(axis=1)
    acc = sum(pred == label)/len(label)
    print(f"Correct Epoch {epoch}: reg clean loader acc is {acc*100:.3f}%")
    logging.info(f"Correct Epoch {epoch}: reg clean loader acc is {acc*100:.3f}%")
    
    clf1, clf2 = Ridge(alpha=1.0).fit(embed1, y_onehot), LinearRegression().fit(embed2, y_onehot)
    logit1, logit2 = clf1.predict(embed1), clf2.predict(embed2)
    logit = logit1+logit2
    pred = logit.argmax(axis=1)
    clean_acc_rdige = sum(pred == label)/len(label)
    print(f"Correct Epoch {epoch}: ridge reg clean loader acc is {clean_acc_rdige*100:.3f}%")
    
    embed_val1, _ = obtain_embeddings(args, val_loader, net1)
    embed_val2, label_val = obtain_embeddings(args, val_loader, net2)
    
    logit_val1, logit_val2 = reg1.predict(embed_val1), reg2.predict(embed_val2)
    logit_val = logit_val1 + logit_val2
    pred_val = logit_val.argmax(axis=1)
    acc = sum(pred_val == label_val)/len(label_val)
    print(f"Correct Epoch {epoch}: reg val acc is {acc*100:.3f}% \n")
    logging.info(f"Correct Epoch {epoch}: reg val acc is {acc*100:.3f}% \n")
    
    logit_val1, logit_val2 = clf1.predict(embed_val1), clf2.predict(embed_val2)
    logit_val = logit_val1 + logit_val2
    pred_val = logit_val.argmax(axis=1)
    val_acc_rdige = sum(pred_val == label_val)/len(label_val)
    print(f"Correct Epoch {epoch}: ridge val acc is {val_acc_rdige*100:.3f}%")
    logging.info(f"Correct Epoch {epoch}: ridge val acc is {val_acc_rdige*100:.3f}%")
    
    embed_tt1, _ = obtain_embeddings(args, test_loader, net1)
    embed_tt2, label_tt = obtain_embeddings(args, test_loader, net2)
    
    logit_tt1, logit_tt2 = reg1.predict(embed_tt1), reg2.predict(embed_tt2)
    logit_tt = logit_tt1 + logit_tt2
    pred_tt = logit_tt.argmax(axis=1)
    acc = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Correct Epoch {epoch}: reg test acc is {acc*100:.3f}% \n")
    logging.info(f"Correct Epoch {epoch}: reg test acc is {acc*100:.3f}% \n")
    
    logit_tt1, logit_tt2 = clf1.predict(embed_tt1), clf2.predict(embed_tt2)
    logit_tt = logit_tt1 + logit_tt2
    pred_tt = logit_tt.argmax(axis=1)
    test_acc_rdige = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Correct Epoch {epoch}: ridge test acc is {test_acc_rdige*100:.3f}% \n\n")
    logging.info(f"Correct Epoch {epoch}: ridge test acc is {test_acc_rdige*100:.3f}% \n\n")

def eval_train(epoch,model):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    if args.add_clean:
        num_samples += 47570
    losses = torch.zeros(num_samples)
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                paths.append(path[b])
                n+=1
            #torch.cuda.empty_cache()
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
            sys.stdout.flush()
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob,paths  
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
def create_model():
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048,args.num_class)
    model = model.to(device)
    return model     

log=open('./checkpoint/%s.txt'%args.id,'w')     
log.flush()

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches,add_clean=args.add_clean)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True
              
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
  
    net1 = nn.DataParallel(net1)
    net2 = nn.DataParallel(net2)

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                      
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

if args.fine_tune:
    args.filename = args.trained_log
else:
    args.filename = f"resnet50_lr{args.lr}_epochs{args.num_epochs}_seed{args.seed}_bs{args.batch_size}_numb{args.num_batches}.log"

best_acc = [0,0]
start_epoch = 0
mode = 'w+'
if args.checkpoint:
    checkpth = f"{args.checkpoint_dir}/clothing1M/{args.filename}/checkpoint.pth.tar"
    # load checkpoint if exisits
    if os.path.exists(checkpth):
        checkpoint = torch.load(checkpth)
        start_epoch = checkpoint['epoch'] + 1
        try:
            best_acc = checkpoint['best_acc']
        except:
            print("no saved best acc")
        net1.load_state_dict(checkpoint['state_dict1'])
        net2.load_state_dict(checkpoint['state_dict2'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        
        mode = 'a+'
        print(f"checkpoint loaded epoch {checkpoint['epoch']}!") 
        epoch = checkpoint['epoch']
              
        clean_loader = loader.run('clean')
        val_loader = loader.run('val')
        test_loader = loader.run('test')
        acc = test(net1,net2,test_loader)    
        if start_epoch < args.num_epochs:
            print('\nloading prob for net 1') 
            eval_loader = loader.run('eval_train')   
            prob1,paths1 = eval_train(checkpoint['epoch'],net1) 
            print('\nloading prob for net 2') 
            eval_loader = loader.run('eval_train')  
            prob2,paths2 = eval_train(checkpoint['epoch'],net2) 

if args.fine_tune:
    mode = 'w+'
    start_epoch = 1
    args.filename = f"resnet50_lr{args.lr}_epochs{args.num_epochs}_seed{args.seed}_bs{args.batch_size}_ft{args.fine_tune}_addclean{args.add_clean}.log"

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
    
    clean_loader = loader.run('clean')
    val_loader = loader.run('val')
    test_loader = loader.run('test')
             
    if epoch<1:  
        # warm up  
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)     
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2,optimizer2,train_loader)   
    elif args.fine_tune:
        print('Finetune Net1')
        finetune(net1,optimizer1,clean_loader)     
        print('\nFinetune Net2')
        finetune(net2,optimizer2,clean_loader)  
    else:       
        pred1 = (prob1 > args.p_threshold)  # divide dataset  
        pred2 = (prob2 > args.p_threshold)      
        
        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)              # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)              # train net2
    
    correct(epoch, net1, net2)
    acc = test(net1,net2,test_loader) 
    # validation
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    logging.info('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))         
    log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    log.flush() 
    print('\n==== net 1 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
    prob1,paths1 = eval_train(epoch,net1) 
    print('\n==== net 2 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  
    prob2,paths2 = eval_train(epoch,net2) 
       
        
    if args.save_model and epoch > args.num_epochs - 10: 
        save_checkpoint(args, epoch, {
                    'epoch': epoch,
                    'args': args, 
                    'state_dict1': net1.state_dict(),
                    'state_dict2': net2.state_dict(),
                    'optimizer1' : optimizer1.state_dict(),
                    'optimizer2' : optimizer2.state_dict(),
                    'best_acc': best_acc
         }, "./models_dir")
            
    if args.checkpoint:
        save_checkpoint(args, epoch, {
                    'epoch': epoch,
                    'args': args, 
                    'state_dict1': net1.state_dict(),
                    'state_dict2': net2.state_dict(),
                    'optimizer1' : optimizer1.state_dict(),
                    'optimizer2' : optimizer2.state_dict(),
                    'best_acc': best_acc
         }, args.checkpoint_dir)          

correct(epoch, net1, net2)
clean_loader = loader.run('clean')
val_loader = loader.run('val')
test_loader = loader.run('test')
net1.load_state_dict(torch.load('./checkpoint/%s_net1.pth.tar'%args.filename))
net2.load_state_dict(torch.load('./checkpoint/%s_net2.pth.tar'%args.filename))
acc = test(net1,net2,test_loader)      
correct(model_best_epoch, net1, net2)
              
log.write('Test Accuracy:%.2f\n'%(acc))
logging.info('Best test Accuracy:%.2f from epoch %d\n'%(acc, model_best_epoch))              
log.flush() 
