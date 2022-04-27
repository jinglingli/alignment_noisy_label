
import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

def call_bn(bn, x, update_batch_stats=True):
    if update_batch_stats:# not implemented 
        #return F.batch_norm(x, torch.mean(x), torch.std(x), training=True)
        return bn(x)
    else:
        return bn(x)

class CNN9(nn.Module):
    def __init__(self, input_channel=3, num_classes=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN9, self).__init__()
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)        
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)        
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)        
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)        
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)        
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)        
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)        
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)        
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)        
        self.l_c1=nn.Linear(128,num_classes)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)
        
    def forward(self, x, update_batch_stats=True):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        if self.top_bn:
            logit=call_bn(self.bn_c1, logit, update_batch_stats=update_batch_stats)
        return logit
    
    def obtain_embedding(self, x, update_batch_stats=True, lout=8):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        if lout==0:
            out = h.reshape(h.shape[0], h.shape[1], -1)
            out = torch.mean(out,2) 
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        if lout==1:
            out = h.reshape(h.shape[0], h.shape[1], -1)
            out = torch.mean(out,2) 
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        if lout==2:
            out = h.reshape(h.shape[0], h.shape[1], -1)
            out = torch.mean(out,2)         
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        if lout==3:
            out = h.reshape(h.shape[0], h.shape[1], -1)
            out = torch.mean(out,2)         
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        if lout==4:
            out = h.reshape(h.shape[0], h.shape[1], -1)
            out = torch.mean(out,2)         
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        if lout==5:
            out = h.reshape(h.shape[0], h.shape[1], -1)
            out = torch.mean(out,2) 
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        if lout==6:
            out = h.reshape(h.shape[0], h.shape[1], -1)
            out = torch.mean(out,2)         
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        if lout==7:
            out = h.reshape(h.shape[0], h.shape[1], -1)
            out = torch.mean(out,2)         
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h,update_batch_stats=update_batch_stats), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        if lout==8:
            out = h
        return out

    
class MLP(nn.Module):
    def __init__(self, input_size=32*32*3, hidden_size=512, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False) 
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False) 
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc4 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = x.view(-1, 32*32*3)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
    
    def obtain_embedding(self, x, lout=2):
        x = x.view(-1, 32*32*3)
        out = self.fc1(x)
        if lout==0:
            res = out
            
        out = self.fc2(out)
        if lout==1:
            res = out
            
        out = self.fc3(out)
        if lout==2:
            res = out
        return res
   