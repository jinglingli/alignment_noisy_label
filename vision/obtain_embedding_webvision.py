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
import pickle
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

loss = nn.MSELoss()

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=2, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--decay_interval', default=20, type=float, help='how learning rate decay')
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

parser.add_argument('--regression_type', type=str, default='linear', help='regression type')
parser.add_argument('--no_log', action='store_true', default=False, help='disables logging of results')
parser.add_argument('--amplify', default=1, type=int)
parser.add_argument('--clean_num', default=100, type=int)
parser.add_argument('--pred_num', default=5000, type=int)
parser.add_argument('--save_model', action='store_true', default=False, help='store the training models')
parser.add_argument('--correct', action='store_true', default=False, help='make correction')
parser.add_argument('--model_path', type=str, default='./checkpoint', help='directory to save the checkpoint files')
parser.add_argument('--log_interval', type=int, default=1, help='how many epochs to save model')
parser.add_argument('--filename', type=str, default='', help='the file which store trained model logs')
parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models logs')

args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#torch.cuda.set_device(args.gpuid)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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

def test_linear(epoch,net1,net2,linear1,linear2,test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if torch.cuda.device_count() > 1:
                embed1 = net1.module.obtain_embedding(inputs)
                embed2 = net2.module.obtain_embedding(inputs)
            else:
                embed1 = net1.obtain_embedding(inputs)
                embed2 = net2.obtain_embedding(inputs)
            
            outputs1 = linear1(embed1)
            outputs2 = linear2(embed2)
            
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1) 
            acc_meter.add(outputs,targets)

    accs = acc_meter.value()
    return accs

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

class nnlinearRegression(nn.Module):
    def __init__(self):
        super(nnlinearRegression, self).__init__()
        self.linear = nn.Linear(1536, 50)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out

def obtain_embedding_linear(epoch,model,linear_model,test_loader):
    acc_meter.reset()
    model.eval()
    linear_model.train()

    mseloss = nn.MSELoss()
    linear_optim = torch.optim.SGD(linear_model.parameters(), lr=args.lr)
    lr = args.lr
    
    for epoch in range(100):
        print("epoch", epoch)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            labels = torch.zeros(len(targets), 50)
            labels[torch.arange(len(targets)), targets] = 1
            labels = labels.cuda()
            
            if torch.cuda.device_count() > 1:
                embed = model.module.obtain_embedding(inputs)
            else:
                embed = model.obtain_embedding(inputs)

            outputs = linear_model(embed)
            loss = mseloss(outputs, labels)   
            
            #penalty = conf_penalty(outputs)
            #L += loss #+ penalty      
            if (batch_idx+1) % 200 == 0:
                print(f"epoch {epoch} batch {batch_idx}: loss {loss}")
                logging.info(f"epoch {epoch} batch {batch_idx}: loss {loss}")
            if epoch == 99:
                acc_meter.add(outputs.detach(),targets)
            torch.cuda.empty_cache()
            
            loss.backward()
            linear_optim.step() 
        
        if (epoch+1) % args.decay_interval == 0:
            lr = lr*0.95
            for param_group in linear_optim.param_groups:
                param_group['lr'] = lr
    accs = acc_meter.value()
    return accs

def obtain_embeddings(args, data_loader, model, using_Train=False, amplifying=args.amplify):
    model.eval()

    embedding_data = []
    labels_all = []
    for _ in range(amplifying):
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.cuda()
            #targets = targets.cuda()
            #labels = torch.zeros(len(targets), 50)
            #labels[torch.arange(len(targets)), targets] = 1   
            if torch.cuda.device_count() > 1:
                embed = model.module.obtain_embedding(inputs)
            else:
                embed = model.obtain_embedding(inputs)
            embedding_data.append(embed.detach().cpu())
            labels_all.extend(targets.numpy())
            torch.cuda.empty_cache() 
    
    if using_Train:
        for batch_idx, (inputs, targets, index) in enumerate(train_loader):
            inputs = inputs.cuda()
            if torch.cuda.device_count() > 1:
                embed = model.module.obtain_embedding(inputs)
            else:
                embed = model.obtain_embedding(inputs)

            embedding_data.append(embed.detach().cpu())
            labels_all.extend(targets.numpy())
            torch.cuda.empty_cache() 
        
    embbedings = torch.cat(embedding_data, dim=0).numpy()
    return embbedings, np.asarray(labels_all) 

def correct(epoch, net1, net2, clean_loader, webvision_loader, imagenet_loader):   
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True) 
    
    #using linear regression
    print("Calculating embedding")
    embed1, _ = obtain_embeddings(args, clean_loader, net1, using_Train=True)
    embed2, label = obtain_embeddings(args, clean_loader, net2, using_Train=True)
    
    y_onehot = torch.zeros(len(label), args.num_class)
    y_onehot[torch.arange(len(label)), label] = 1
    print(len(label))
    
    print("Compute linear regression")
    if 'linear' in args.regression_type:
        reg1, reg2 = LinearRegression().fit(embed1, y_onehot), LinearRegression().fit(embed2, y_onehot)
    else:
        reg1, reg2 = Ridge(alpha=1.0).fit(embed1, y_onehot), Ridge(alpha=1.0).fit(embed2, y_onehot)
    logit1, logit2 = reg1.predict(embed1), reg2.predict(embed2)
    logit = logit1+logit2
    acc_meter.add(logit,label)
    #pred = logit.argmax(axis=1)
    #clean_acc = sum(pred == label)/len(label)
    #print(f"Correct Epoch {epoch}: reg clean loader acc is {clean_acc*100:.3f}%")    
    train_accs = acc_meter.value()
    print("Clean Acc: %.2f%% (%.2f%%)"%(train_accs[0],train_accs[1]))  
    logging.info("Clean Acc: %.2f%% (%.2f%%)"%(train_accs[0],train_accs[1]))
    
    acc_meter.reset()
    embed_val1, _ = obtain_embeddings(args, webvision_loader, net1)
    embed_val2, label_val = obtain_embeddings(args, webvision_loader, net2)
    logit_val1, logit_val2 = reg1.predict(embed_val1), reg2.predict(embed_val2)
    logit_val = logit_val1 + logit_val2
    #pred_val = logit_val.argmax(axis=1)
    #val_acc = sum(pred_val == label_val)/len(label_val)
    #print(f"Correct Epoch {epoch}: reg val loader acc is {val_acc*100:.3f}%")
    acc_meter.add(logit_val,label_val)
    webvision_accs = acc_meter.value()
    print("Webvision Acc: %.2f%% (%.2f%%)"%(webvision_accs[0],webvision_accs[1]))
    logging.info("Webvision Acc: %.2f%% (%.2f%%)"%(webvision_accs[0],webvision_accs[1]))
    
    acc_meter.reset()
    embed_tt1, _ = obtain_embeddings(args, imagenet_loader, net1)
    embed_tt2, label_tt = obtain_embeddings(args, imagenet_loader, net2)
    logit_tt1, logit_tt2 = reg1.predict(embed_tt1), reg2.predict(embed_tt2)
    logit_tt = logit_tt1 + logit_tt2
    #pred_tt = logit_tt.argmax(axis=1)
    #test_acc = sum(pred_tt == label_tt)/len(label_tt)
    #print(f"Correct Epoch {epoch}: reg test acc is {test_acc*100:.3f}% \n")
    acc_meter.add(logit_tt,label_tt)
    imagenet_accs = acc_meter.value()
    print("Imagenet Acc: %.2f%% (%.2f%%)"%(imagenet_accs[0],imagenet_accs[1]))
    logging.info("Imagenet Acc: %.2f%% (%.2f%%)"%(imagenet_accs[0],imagenet_accs[1]))
    
    return train_accs, webvision_accs, imagenet_accs

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

def create_model():
    model = InceptionResNetV2(num_classes=args.num_class)
    model = model.cuda()
    return model

stats_log=open('./checkpoint/correct_%s'%(args.id)+'_stats.txt','w') 
test_log=open('./checkpoint/correct_%s'%(args.id)+'_acc.txt','w')     

warm_up=1

print('| Building net')
net1 = create_model()
net2 = create_model()
linear1 = nnlinearRegression().cuda()
linear2 = nnlinearRegression().cuda()

cudnn.benchmark = True
device_count = torch.cuda.device_count()

if device_count > 1:
    print("Let's use", device_count, "GPUs!")
    net1 = nn.DataParallel(net1)
    net2 = nn.DataParallel(net2)

acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True) 
#args.filename = f"InceptionResNetV2_lr{args.lr}_epochs{args.num_epochs}_seed{args.seed}_bs{args.batch_size}.log"
start_epoch = 0
mode = 'w+'
if args.correct:
    checkpth = args.model_path
    # load checkpoint if exisits
    if os.path.exists(checkpth):
        checkpoint = torch.load(checkpth, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        
        state_dict1 = checkpoint['state_dict1']
        for key in list(state_dict1):
            if "module." in key and device_count <= 1:
                state_dict1[key[7:]] = state_dict1[key]
                del state_dict1[key]  
            elif device_count > 1 and "module." not in key:
                state_dict1["module.%s" %key] = state_dict1[key]
                del state_dict1[key] 
                    
        state_dict2 = checkpoint['state_dict2']
        for key in list(state_dict2):
            if "module." in key and device_count <= 1:
                state_dict2[key[7:]] = state_dict2[key]
                del state_dict2[key] 
            elif device_count > 1 and "module." not in key:
                state_dict2["module.%s" %key] = state_dict2[key]
                del state_dict2[key] 
        
        net1.load_state_dict(checkpoint['state_dict1'])
        net2.load_state_dict(checkpoint['state_dict2'])
        
        print(f"checkpoint loaded epoch {checkpoint['epoch']}!") 
        epoch = checkpoint['epoch']

        loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_workers=5,root_dir=args.data_path,log=stats_log, num_class=args.num_class)
        web_valloader = loader.run('test')
        imagenet_valloader = loader.run('imagenet') 
        
        web_acc = test(epoch,net1,net2,web_valloader)  
        imagenet_acc = test(epoch,net1,net2,imagenet_valloader)  
  
        print("Checkpoint Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(checkpoint['epoch'],web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
        
args.filename = f"{args.regression_type}_correct_cleanNum{args.clean_num}_predNum{args.pred_num}_amplify{args.amplify}.log"  
setup_logs(args, mode)
              
save_path = f"./results/correct_info/webvision/ELR_{args.regression_type}_cleanNum{args.clean_num}_predNum{args.pred_num}_amplify{args.amplify}.p"
              
avg_tr_acc, avg_webvision_acc, avg_imagenet_acc = [[],[]], [[],[]], [[],[]]
for seed in range(5): 
    seed += 100
    random.seed(seed)
              
    loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_workers=5,root_dir=args.data_path,log=stats_log, num_class=args.num_class, clean_file=f'./noise_jason/webvision_clean{args.clean_num}_{seed}.p', pred_file=f'./noise_jason/webvision_pred{args.pred_num}_{seed}.p', clean_num=args.clean_num, pred_num=args.pred_num)          
              
              
    imagenet_clean_tr = loader.run('imagenet_clean') #using transform_clean
    #imagenet_clean_tt = loader.run('imagenet_clean_test') #using transform_test 
              
    web_valloader = loader.run('test')
    imagenet_valloader = loader.run('imagenet')   
    train_loader = loader.run('part')   
              
    print(len(imagenet_clean_tr.dataset))
    print(len(train_loader.dataset))   
    train_accs, webvision_accs, imagenet_accs = correct(epoch, net1, net2, imagenet_clean_tr, web_valloader, imagenet_valloader)           
    avg_tr_acc[0].append(train_accs[0])          
    avg_tr_acc[1].append(train_accs[1]) 
              
    avg_webvision_acc[0].append(webvision_accs[0])
    avg_webvision_acc[1].append(webvision_accs[1])
    
    avg_imagenet_acc[0].append(imagenet_accs[0])        
    avg_imagenet_acc[1].append(imagenet_accs[1])          

print("Average clean Acc: %.2f%% (%.2f%%)"%(np.mean(avg_tr_acc[0]),np.mean(avg_tr_acc[1])))
logging.info("Average clean Acc: %.2f%% (%.2f%%)"%(np.mean(avg_tr_acc[0]),np.mean(avg_tr_acc[1])))

print("Average webvision Acc: %.2f%% (%.2f%%)"%(np.mean(avg_webvision_acc[0]),np.mean(avg_webvision_acc[1])))
logging.info("Average webvision Acc: %.2f%% (%.2f%%)"%(np.mean(avg_webvision_acc[0]),np.mean(avg_webvision_acc[1])))                            
print("Average imagenet Acc: %.2f%% (%.2f%%)"%(np.mean(avg_imagenet_acc[0]),np.mean(avg_imagenet_acc[1])))
logging.info("Average imagenet Acc: %.2f%% (%.2f%%)"%(np.mean(avg_imagenet_acc[0]),np.mean(avg_imagenet_acc[1])))

pickle.dump([avg_tr_acc, avg_webvision_acc, avg_imagenet_acc], open(save_path, "wb" ))
              
'''             
for epoch in range(1):   
    imagenet_clean = loader.run('imagenet_clean')   
    web_valloader = loader.run('test')
    imagenet_valloader = loader.run('imagenet')   
     
    print(len(imagenet_clean.dataset))         
    clean_acc1 = obtain_embedding_linear(epoch,net1,linear1,imagenet_clean)  
    clean_acc2 = obtain_embedding_linear(epoch,net2,linear2,imagenet_clean) 
    web_acc = test(epoch,net1,net2,linear1,linear2,web_valloader)         
    imagenet_acc = test(epoch,net1,net2,linear1,linear2,imagenet_valloader)    
    
    print("\n| Clean net 1 Acc: %.2f%% (%.2f%%)\n" %(clean_acc1[0],clean_acc1[1]))    
    logging.info("\n| Clean net 1 Acc: %.2f%% (%.2f%%)\n" %(clean_acc1[0],clean_acc1[1]))          
    print("\n| Clean net 2 Acc: %.2f%% (%.2f%%)\n" %(clean_acc2[0],clean_acc2[1]))       
    logging.info("\n| Clean net 2 Acc: %.2f%% (%.2f%%)\n" %(clean_acc2[0],clean_acc2[1]))           
    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))  
    logging.info("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))           
''' 