import os
import cv2 as cv
import os
import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

class GuiderDataset(Dataset):

    def __init__(self, dir_path, test_size, max_len):
        self.dir = dir_path
        self.max_len = max_len # use to padding all sequence to a fixed length
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
                if seq[-1] == [2,2]:
                    continue
                    print('True')
                for i in range(len(seq)):
                    if isinstance(seq[i],tuple):
                        seq[i] = list(seq[i])
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
        )

        image_path = os.path.join(self.pic_dir,seq[-1]+'.png')
        image = Image.open(image_path)
        tensor = trans(image) # (C,W,H)

        enter_point = torch.tensor(seq[0],dtype=torch.float)
        esc_point = torch.tensor(seq[-2],dtype=torch.float)

        seq = seq[:-1]
        if seq_len < self.max_len:
            seq += [[0., 0.] for _ in range(self.max_len - seq_len)]
        elif seq_len > self.max_len:
            seq = seq[:self.max_len]
            seq_len = self.max_len # be careful!
        seq = torch.tensor(seq ,dtype=torch.float) # (max_len, 2)
        seq_len = torch.tensor(seq_len,dtype=torch.long).unsqueeze(0)

        return {'image':tensor, 'seq':seq, 'enter':enter_point, 'esc':esc_point, 'len':seq_len}

    def train_set(self):
        '''call this method to switch to train mode'''
        self.data = self.train_data
        return copy.deepcopy(self)

    def test_set(self):
        '''call this method to switch to test mode'''
        self.data = self.test_data
        return copy.deepcopy(self)

# debug
if __name__ == '__main__':
    output_path = '/data/lzt/project/waysguider/vis/'
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    data_path = '/data/lzt/project/waysguider/dataset/'
    dataset = GuiderDataset(data_path,0.2,max_len=60)
    train_loader = DataLoader(dataset.train_set(), batch_size=32, shuffle=False)
    for i,data in enumerate(train_loader):
        save_dir = os.path.join(output_path,str(i))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        imgs = data['image'] # (b,c,w,h)
        seq = data['seq'] # (b,max_len,2)
        enter = data['enter'] # (b,2)
        esc = data['esc'] # (b,2)
        length = data['len'] # (b,1) it seem to be a 1D CPU int64 tensor when use pack_padded_sequence below
        #print(imgs)
        for k in range(len(data['image'])):
            img = imgs[k].numpy()
            c,w,h = img.shape
            img[0] = img[0]*0.229+0.485
            img[1] = img[1]*0.224+0.456
            img[2] = img[2]*0.225+0.406
            img = img * 255
            img.astype(np.int)
            img = img.transpose(1,2,0)
            print(enter[k],esc[k])
            img = cv.copyMakeBorder(img, 5, 5, 5, 5, cv.BORDER_CONSTANT,value=[225,225,225])
            for j in range(length[k]):
                m, n = int(h/2+seq[k][j,1]*(h/2-1)),int(h/2+seq[k][j,0]*(h/2-1))
                if seq[k][j,1] == 3:
                    break
                img[-m-3:-m+3,n-3:n+3,:] = np.zeros_like(img[-m-3:-m+3,n-3:n+3,:])
                #print(img[:,int(seq[k][j,1]*h)+256,256+int(seq[k][j,0]*h)])

            img[-int(h/2+enter[k][1]*(h/2-1))-3:-int(h/2+enter[k][1]*(h/2-1))+3,int(h/2+enter[k][0]*(h/2-1))-3:int(h/2+enter[k][0]*(h/2-1))+3,:] = np.full_like(img[-int(h/2+enter[k][1]*(h/2-1))-3:-int(h/2+enter[k][1]*(h/2-1))+3,int(h/2+enter[k][0]*(h/2-1))-3:int(h/2+enter[k][0]*(h/2-1))+3,:],100)
            if esc[k][1] != 3:
                img[-int(h/2+esc[k][1]*(h/2-1))-3:-int(h/2+esc[k][1]*(h/2-1))+3,int(h/2+esc[k][0]*(h/2-1))-3:int(h/2+esc[k][0]*(h/2-1))+3,:] = np.full_like(img[-int(h/2+esc[k][1]*(h/2-1))-3:-int(h/2+esc[k][1]*(h/2-1))+3,int(h/2+esc[k][0]*(h/2-1))-3:int(h/2+esc[k][0]*(h/2-1))+3,:],200)

            cv.imwrite(os.path.join(save_dir,str(k)+'.png'),img)
        break



