import argparse
import pickle
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

pos_dict = {0: [3, 3], 1: [3, 4], 2: [3, 5], 3: [4, 3], 4: [4, 4], 5: [4, 5], 
            6: [5, 3], 7: [5, 4], 8: [5, 5], 9: [28, 28]}


def create_data(train_data, train_label): 
    random.seed(123)
    train_data = train_data.copy()
    train_label = train_label.copy()
    
    new_idx = list(range(len(train_label)))
    random.shuffle(new_idx)
    
    for i in range(len(train_label)):
        index = new_idx[i]
        img, target = train_data[index], train_label[index]
        img = img.reshape(3, 32, 32)
                
        color_idx = i % 10
        pos = pos_dict[color_idx]
        img[:, pos[0], pos[1]] = np.array([255, 0, 255])
        train_label[index] = color_idx
        
        train_data[index] = img.flatten()
    return train_data, train_label   


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

    
def main():
    parser = argparse.ArgumentParser(description='Generate CIFAR-Easy')
    parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
    args = parser.parse_args()
    
    ind = 2
    transform = transforms.Compose([transforms.ToTensor(),])         
        
    root_dir = args.data_path
    train_data=[]
    train_label=[]

    for n in range(1,6):
        dpath = '%s/data_batch_%d'%(root_dir,n)
        data_dic = unpickle(dpath)
        train_data.append(data_dic['data'])
        train_label = train_label+data_dic['labels']
    train_data = np.concatenate(train_data)

    new_train_data, new_train_label = create_data(train_data, train_label)  

    train_loc = './dataset/cifar_easy_train.p'
    train_dic = {}
    train_dic['data'] = new_train_data
    train_dic['labels'] = new_train_label
    train_dic['orig_labels'] = train_label
    pickle.dump(train_dic, open(train_loc, 'wb'))
    
    test_dic = unpickle('%s/test_batch'%root_dir)
    test_data = test_dic['data']
    test_label = test_dic['labels']

    new_test_data, new_test_label = create_data(test_data, test_label)    

    test_loc = './dataset/cifar_easy_test.p'
    test_dic = {}
    test_dic['data'] = new_test_data
    test_dic['labels'] = new_test_label
    test_dic['orig_labels'] = test_label
    pickle.dump(test_dic, open(test_loc, 'wb'))
    
    print(f"dataset stored to {train_loc} and {test_loc}")

if __name__ == '__main__':
    main()
