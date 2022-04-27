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
import pickle
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture
import logging
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
  
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
    
def val(net,val_loader, k):
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
            torch.cuda.empty_cache()
    acc = 100.*correct/total
    print("\n| Epoch #%d:  Test Acc: %.2f%%\n" %(epoch, acc))  
    return acc    

def obtain_embeddings(args, data_loader, model):
    model.eval()
    
    embedding_data = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(data_loader):
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
    
    embed_val1, _ = obtain_embeddings(args, val_loader, net1)
    embed_val2, label_val = obtain_embeddings(args, val_loader, net2)
    
    embed_tt1, _ = obtain_embeddings(args, test_loader, net1)
    embed_tt2, label_tt = obtain_embeddings(args, test_loader, net2)
    
    y_onehot = torch.zeros(len(label), args.num_class)
    y_onehot[torch.arange(len(label)), label] = 1

    reg1, reg2 = LinearRegression().fit(embed1, y_onehot), LinearRegression().fit(embed2, y_onehot)
    logit1, logit2 = reg1.predict(embed1), reg2.predict(embed2)
    logit = logit1+logit2
    pred = logit.argmax(axis=1)
    clean_acc = sum(pred == label)/len(label)
    print(f"Correct Epoch {epoch}: reg clean loader acc is {clean_acc*100:.3f}%")
    
    logit_val1, logit_val2 = reg1.predict(embed_val1), reg2.predict(embed_val2)
    logit_val = logit_val1 + logit_val2
    pred_val = logit_val.argmax(axis=1)
    val_acc = sum(pred_val == label_val)/len(label_val)
    print(f"Correct Epoch {epoch}: reg val acc is {val_acc*100:.3f}%")
    
    logit_tt1, logit_tt2 = reg1.predict(embed_tt1), reg2.predict(embed_tt2)
    logit_tt = logit_tt1 + logit_tt2
    pred_tt = logit_tt.argmax(axis=1)
    test_acc = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Correct Epoch {epoch}: reg test acc is {test_acc*100:.3f}%\n")
    
    clf1, clf2 = Ridge(alpha=1.0).fit(embed1, y_onehot), Ridge(alpha=1.0).fit(embed2, y_onehot)
    logit1, logit2 = clf1.predict(embed1), clf2.predict(embed2)
    logit = logit1+logit2
    pred = logit.argmax(axis=1)
    clean_acc_rdige = sum(pred == label)/len(label)
    print(f"Correct Epoch {epoch}: Ridge reg clean loader acc is {clean_acc_rdige*100:.3f}%")
    
    logit_val1, logit_val2 = clf1.predict(embed_val1), clf2.predict(embed_val2)
    logit_val = logit_val1 + logit_val2
    pred_val = logit_val.argmax(axis=1)
    val_acc_rdige = sum(pred_val == label_val)/len(label_val)
    print(f"Correct Epoch {epoch}: reg val acc is {val_acc_rdige*100:.3f}%")
    
    logit_tt1, logit_tt2 = clf1.predict(embed_tt1), clf2.predict(embed_tt2)
    logit_tt = logit_tt1 + logit_tt2
    pred_tt = logit_tt.argmax(axis=1)
    test_acc_rdige = sum(pred_tt == label_tt)/len(label_tt)
    print(f"Correct Epoch {epoch}: reg test acc is {test_acc_rdige*100:.3f}% \n\n")
    return clean_acc, val_acc, test_acc, clean_acc_rdige, val_acc_rdige, test_acc_rdige

def eval_train(epoch,model):
    model.eval()
    num_samples = args.num_batches*args.batch_size
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

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
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

parser.add_argument('--model_path', type=str, default='', help='model path')

args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

model_path = args.model_path

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                      
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

checkpoint = torch.load(model_path)
epoch = checkpoint['epoch']

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
  
    net1 = nn.DataParallel(net1)
    net2 = nn.DataParallel(net2)
else:
    statedict1 = checkpoint['state_dict1']
    for key in list(statedict1): 
        if 'module' in key:
            statedict1[key[7:]] =  statedict1[key]
            del statedict1[key]

    statedict2 = checkpoint['state_dict2']
    for key in list(statedict2): 
        if 'module' in key:
            statedict2[key[7:]] =  statedict2[key]
            del statedict2[key]

net1.load_state_dict(checkpoint['state_dict1'])
net2.load_state_dict(checkpoint['state_dict2'])
optimizer1.load_state_dict(checkpoint['optimizer1'])
optimizer2.load_state_dict(checkpoint['optimizer2'])
print(f"checkpoint loaded epoch {checkpoint['epoch']}!") 

test_loader = loader.run('test')
model_test_acc = test(net1,net2,test_loader)
'''
val_loader = loader.run('val')
model_val_acc1 = val(net1,val_loader, k=1)
val_loader = loader.run('val')
model_val_acc2 = val(net2,val_loader,k=2)
'''

clean_loader = loader.run('clean')
test_loader = loader.run('test')
val_loader = loader.run('val')
clean_acc, val_acc, test_acc, clean_acc_rdige, val_acc_rdige, test_acc_rdige = correct(epoch, net1, net2)

acc_dict = {}

acc_dict['clean_acc'], acc_dict['val_acc'], acc_dict['test_acc'], acc_dict['model_test_acc'], acc_dict['clean_acc_ridge'], acc_dict['val_acc_ridge'], acc_dict['test_acc_ridge'] = clean_acc, val_acc, test_acc, model_test_acc, clean_acc_rdige, val_acc_rdige, test_acc_rdige

print("model_test_acc", model_test_acc)

pre = model_path.split("/model_epoch")[0]
save_dir = f"{pre}/correct_info"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = f"{save_dir}/epoch_{epoch}.p"
pickle.dump(acc_dict, open(save_path, "wb" ))