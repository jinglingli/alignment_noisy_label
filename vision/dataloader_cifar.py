from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from numpy.testing import assert_array_almost_equal
            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, flip_type='2016', noise_file='', clean_ratio=0.1, pred=[], probability=[], log='', add_clean=False): 
        
        self.r = r # noise ratio
        self.clean_ratio = clean_ratio # ratio of clean train data
        self.transform = transform
        self.mode = mode
        # class transition for asymmetric noise
        if dataset in ['cifar10', 'binary_cifar10']: 
            self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} 
        
        elif dataset=='cifar100':
            self.transition = {}
            if flip_type == '2016':
                for i in range(20): # 20 super classes
                    for j in range(i*5, (i+1)*5): # each has 5 classes
                        self.transition[j] = j+1
                    self.transition[j] = i*5 # circular corruption
            elif flip_type == '2019':
                for i in range(100):
                    self.transition[i] = i
                # flipping between two randomly selected sub-classes within each super-class
                for i in range(20): # 20 super classes
                    cls1, cls2 = np.random.choice(range(5), size=2, replace=False)
                    cls1, cls2 = int(cls1), int(cls2)
                    self.transition[i*5 + cls1] = i*5 + cls2
                    self.transition[i*5 + cls2] = i*5 + cls1
                    
        if self.mode in ['test', 'binary_test']:
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']    
            elif dataset=='binary_cifar10':
                test_dic = unpickle(f'./dataset/{root_dir}_cifar10_test.p')
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
                self.orig_label = test_dic['orig_labels']
        else:    
            train_data=[]
            train_label=[]
            if 'binary' in dataset: 
                train_dic = unpickle(f'./dataset/{root_dir}_cifar10_train.p')
                train_data = train_dic['data']
                train_label = train_dic['labels']
                self.orig_label = train_dic['orig_labels']
            elif dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise_label, noise_idx, clean_idx = json.load(open(noise_file,"r"))
            else:    #inject noise   
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]
                num_clean = int(self.clean_ratio*50000)
                clean_idx = idx[num_noise:num_noise+num_clean]
                if len(clean_idx) < num_clean:
                    clean_idx = idx[-num_clean:]
                    
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            if dataset in ['cifar10', 'binary_cifar10']: 
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                        elif noise_mode=='adv':
                            assert dataset == 'binary_cifar10'
                            noiselabel = self.orig_label[i]
                            noise_label.append(noiselabel)
                    else:    
                        noise_label.append(train_label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump((noise_label, noise_idx, clean_idx) ,open((noise_file),"w"))       
            
            if len(clean_idx) + len(noise_idx) > len(train_label):
                assert dataset == 'binary_cifar10' and noise_mode=='adv'
                clean_data = train_data[clean_idx]
                clean_label = np.asarray(train_label)[clean_idx]
            else:        
                clean_data = train_data[clean_idx]
                clean_label = np.asarray(noise_label)[clean_idx]
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            elif 'clean' in self.mode: # self.mode == 'clean':
                self.train_data = clean_data
                self.noise_label = clean_label
            elif self.mode == 'binary':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    print("labeled before", add_clean, len(pred_idx))
                    if add_clean:
                        pred_idx = np.union1d(pred_idx, clean_idx)
                    print("labeled after", len(pred_idx))    
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]   
                    print("unlabeled before", add_clean, len(pred_idx))
                    if add_clean:
                        pred_idx = np.setdiff1d(pred_idx, clean_idx, assume_unique=True)
                    print("unlabeled after", len(pred_idx))  
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))
                log.write("%s data has a size of %d\n"%(self.mode,len(self.noise_label)))
                log.flush()   
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        elif 'clean' in self.mode:  #self.mode=='clean':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        elif self.mode=='binary':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index       
        elif self.mode=='binary_test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        
    def __len__(self):
        if self.mode not in ['test', 'binary_test']:
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, clean_ratio, noise_mode, batch_size, num_workers, root_dir, log, flip_type='2019', noise_file='', seed=123, clean_size=5000, add_clean=False):
        self.dataset = dataset
        self.r = r
        self.clean_ratio = clean_ratio
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.flip_type = flip_type
        self.seed = seed
        self.clean_size = clean_size
        self.add_clean = add_clean
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_none = transforms.Compose([
                    transforms.ToTensor(),
                ]) 
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all", flip_type=self.flip_type, noise_file=self.noise_file, clean_ratio=self.clean_ratio)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        elif mode=='standard':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode="all", flip_type=self.flip_type, noise_file=self.noise_file, clean_ratio=self.clean_ratio) 
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader                             

        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled",flip_type=self.flip_type, noise_file=self.noise_file, pred=pred, probability=prob,log=self.log, clean_ratio=self.clean_ratio, add_clean=self.add_clean)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", flip_type=self.flip_type, noise_file=self.noise_file, pred=pred,log=self.log, clean_ratio=self.clean_ratio, add_clean=self.add_clean)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test', flip_type=self.flip_type)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', flip_type=self.flip_type, noise_file=self.noise_file, clean_ratio=self.clean_ratio)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader      
        
        elif mode=='clean':
            clean_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='clean', flip_type=self.flip_type, noise_file=self.noise_file, clean_ratio=self.clean_ratio)      
            clean_loader = DataLoader(
                dataset=clean_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return clean_loader
    
        elif mode=='fine_tune':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', flip_type=self.flip_type, noise_file=self.noise_file, clean_ratio=self.clean_ratio) 
            
            np.random.seed(self.seed)
            random.seed(self.seed)
            
            num_train = len(eval_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            clean_idx = indices[:self.clean_size]
            clean_sampler = SubsetRandomSampler(clean_idx)
            
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                sampler=clean_sampler,
                shuffle=False,
                num_workers=self.num_workers)  
            return eval_loader    
        
        elif mode=='binary':
            binary_dataset = cifar_dataset(dataset='binary_cifar10', noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_none, mode='binary', flip_type=self.flip_type, noise_file=self.noise_file, clean_ratio=self.clean_ratio)      
            binary_loader = DataLoader(
                dataset=binary_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)          
            return binary_loader    
        
        elif mode=='binary_train':
            labeled_dataset = cifar_dataset(dataset='binary_cifar10', noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_none, mode="labeled",flip_type=self.flip_type, noise_file=self.noise_file, pred=pred, probability=prob,log=self.log, clean_ratio=self.clean_ratio, add_clean=self.add_clean)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset='binary_cifar10', noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_none, mode="unlabeled", flip_type=self.flip_type, noise_file=self.noise_file, pred=pred,log=self.log, clean_ratio=self.clean_ratio, add_clean=self.add_clean)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='binary_test':
            binary_dataset = cifar_dataset(dataset='binary_cifar10', noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_none, mode='binary_test', flip_type=self.flip_type, noise_file=self.noise_file, clean_ratio=self.clean_ratio)      
            binary_loader = DataLoader(
                dataset=binary_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return binary_loader 
        
        elif mode=='binary_clean':
            clean_dataset = cifar_dataset(dataset='binary_cifar10', noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_none, mode='binary_clean', flip_type=self.flip_type, noise_file=self.noise_file, clean_ratio=self.clean_ratio)      
            clean_loader = DataLoader(
                dataset=clean_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return clean_loader