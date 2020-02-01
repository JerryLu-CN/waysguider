#coding:utf-8
import os
import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from vis import *
from collections import Counter

class GuiderDataset(Dataset):

    def __init__(self, dir_path, test_size, max_len, min_len=20, target_inter=0.8):
        self.dir = dir_path
        self.max_len = max_len # use to padding all sequence to a fixed length
        self.min_len = min_len # use to delete sequence with too many points
        self.target_inter = target_inter
        self.pic_dir = os.path.join(self.dir,'map/')
        self.seq_dir = os.path.join(self.dir,'seq/')
        self.pic_name = []
        self.data = [] # use to store all seq
        self.trans = torchvision.transforms.ToTensor()

        for image in os.listdir(self.pic_dir):
            image_name = image.rsplit('.')[0]
            self.pic_name.append(image_name)

        # data preprocess:
        # 1. normalize all coordinate according to the first element(x,y,w,h) in each npy file
        # 2. append the image file name to each coordinate sequence
        # 3. concate all sequence in npy file into one
        for image_name in self.pic_name:
            data = np.load(os.path.join(self.seq_dir,image_name+'.npy'),allow_pickle=True)
            anchor = data[0]
            x,y = anchor[0]
            w,h = anchor[1]
            data = np.delete(data,0)  # !!! here delete the anchor point
            for seq in data:
                if len(seq) > self.min_len:
                    # omit too long
                    continue
                if cal_dis(seq) < 0.2:
                    continue # delete seq too short
                if intervals_avg(seq) > 1:
                    continue
                if seq[-1] == [2,2] or seq[0] == [2,2]:
                    seq = seq[:-1] # !!! 暂不考虑
                for i in range(len(seq)):
                    if isinstance(seq[i],tuple):
                        seq[i] = list(seq[i])
                    #code for data that hasn't been normalized
                    #seq[i][0] = 2. * (seq[i][0] - x) / w - 1. # rescale to (-1,1)
                    #seq[i][1] = 2. * (seq[i][1] - y) / h - 1.
                    # it seems data has some error.
                    #if seq[i][0] < -1. or seq[i][0] > 1.:
                    #    print('Error')
                    #    seq[i] = [0., 0.]
                    #if seq[i][1] < -1. or seq[i][1] > 1.:
                    #    print('Error')
                    #    seq[i] = [0., 0.]
                    seq[i][0] = 2 * seq[i][0] - 1
                    seq[i][1] = 2 * seq[i][1] - 1
                seq.append(image_name) # append cooresponding map image name to each sequence
                self.data.append(seq) # seq is a list of list

        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size,random_state=0)
        print("="*50)
        print("Data Preprocess Done!")
        print("Dataset size:{}, train:{}, val:{}".
              format(len(self.data),len(self.train_data),len(self.test_data)))
        print("="*50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = copy.deepcopy(self.data[item]) # a list
        seq_len = len(seq) - 1 # except the last filename element

        trans = transforms.Compose(
            [transforms.Resize((512,512)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        image_name = seq[-1]
        seq = seq[:-1]
        image_path = os.path.join(self.pic_dir,image_name+'.png')
        image = Image.open(image_path)
        tensor = trans(image) # (C,W,H)

        enter_point = torch.tensor(seq[0],dtype=torch.float) # dim = 2
        esc_d = torch.tensor(cal_direction(seq[-1]),dtype=torch.float) # dim = 4 for 4 direction
        
        # <----- data preprocess ------->
        if seq_len < self.max_len:
            seq += [[0., 0.] for _ in range(self.max_len - seq_len)]
        elif seq_len > self.max_len:
            # systematically sample a subset for long sequence
            dis = int(seq_len / self.max_len)
            ind = [dis*i for i in range(self.max_len)]
            seq = [seq[i] for i in ind]
            seq_len = self.max_len # be careful!
        # <----------------------------->
        
        seq = torch.tensor(seq ,dtype=torch.float) # (max_len, 2)
        seq_len = torch.tensor(seq_len,dtype=torch.long).unsqueeze(0)

        return {'name':image_name, 'image':tensor, 'seq':seq, 'enter':enter_point, 'esc':esc_d, 'len':seq_len}

    def train_set(self):
        '''call this method to switch to train mode'''
        self.data = self.train_data
        return copy.deepcopy(self)

    def test_set(self):
        '''call this method to switch to test mode'''
        self.data = self.test_data
        return copy.deepcopy(self)

def cal_direction(point):
    '''
    give out the closest direction of the sequnce final point
    :param: point should be a list
    :output: direction dim = 4
        x+y-1 | x-y | direction
        +     | +   | 3 
        +     | -   | 2
        -     | +   | 0
        -     | -   | 1
    '''
    a = point[0] + point[1] - 1
    b = point[0] - point[1]
    
    if a > 0 and b > 0:
        di = 3
    elif a > 0 and b <= 0:
        di = 2
    elif a <= 0 and b > 0:
        di = 0
    else:
        di = 1
    direction = [0,0,0,0]
    direction[di] = 1
    return direction

def intervals_avg(seq):
    '''used to calculate the average intervals of a certain sequence'''
    #seq = seq.tolist()
    seq_ = np.array(seq[1:])
    seq = np.array(seq[:-1])
    intervals = np.sqrt(np.power(seq-seq_,2).sum(axis=1)).mean()
    return intervals
    
    
def cal_dis(seq):
    '''calculate the distance between seq[0] and seq[-1]'''
    x1 = seq[0][0]
    y1 = seq[0][1]
    x2 = seq[-1][0]
    y2 = seq[-1][1]
    dis = np.sqrt((y2 - y1)**2 + (x2 - x1)**2).item()
    return dis
    
    
# debug
if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    data_path = '/data/lzt/project/waysguider/dataset'
    dataset = GuiderDataset(data_path,0.2,max_len=8)
    data = dataset.data[10]
    cache = []
    for i in range(len(dataset.data)):
        cache.append(len(dataset.data[i]))
    summary = Counter(cache)
    print(summary)
