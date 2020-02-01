#coding:utf-8
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


def visualize(output_dir,imgs, seq, enter, esc, length, epoch):
    """
    visualize a output of validation epoch
    
    :param output_dir: a path, which will be created when not exists
    :param imgs: torch GPU Tensor of (b,c,w,h)
    :param seq: torch GPU Tensor of (b,max_len,2)
    :param enter: (b,2)
    :param esc: (b,4)
    :param length: (b,1)
    """

    imgs = imgs.cpu()
    seq = seq.cpu()
    enter = enter.cpu()
    esc = esc.cpu().nonzero().tolist() # (b,2)
    
    direction = {0:'left', 1:'up', 2:'right', 3:'below'}
    output_path = output_dir
    save_dir = output_path+'epoch_'+str(epoch)+'/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    for k in range(len(seq)):
        img = imgs[k].numpy()
        c,w,h = img.shape
        #img[0] = img[0]*0.229+0.485
        #img[1] = img[1]*0.224+0.456
        #img[2] = img[2]*0.225+0.406
        img = img * 255
        img.astype(np.int)
        img = img.transpose(1,2,0)
        img = cv.copyMakeBorder(img, 5, 5, 5, 5, cv.BORDER_CONSTANT,value=[225,225,225])
        for j in range(length[k]):
            m, n = int(h/2+seq[k][j,1]*(h/2-1)),int(h/2+seq[k][j,0]*(h/2-1))
            if seq[k][j,1] == 3:
                break
            img[-m-3:-m+3,n-3:n+3,:] = np.zeros_like(img[-m-3:-m+3,n-3:n+3,:])
            #print(img[:,int(seq[k][j,1]*h)+256,256+int(seq[k][j,0]*h)])

        img[-int(h/2+enter[k][1]*(h/2-1))-3:-int(h/2+enter[k][1]*(h/2-1))+3,int(h/2+enter[k][0]*(h/2-1))-3:int(h/2+enter[k][0]*(h/2-1))+3,:] = np.full_like(img[-int(h/2+enter[k][1]*(h/2-1))-3:-int(h/2+enter[k][1]*(h/2-1))+3,int(h/2+enter[k][0]*(h/2-1))-3:int(h/2+enter[k][0]*(h/2-1))+3,:],100)
        di = direction[esc[k][1]]
        
        cv.imwrite(os.path.join(save_dir,str(k)+'di-{}'.format(di)+'.png'),img)