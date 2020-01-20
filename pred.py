#coding:utf-8
import os
import time
import torch.utils.data as Data
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, Decoder
from dataset import *
from utils import *
from vis import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
torch.backends.cudnn.benchmark = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)

checkpoint = '/data/lzt/project/waysguider/checkpoint/checkpoint_best.pth'
data_path = '/data/lzt/project/waysguider/dataset/'
output_path = '/data/lzt/project/waysguider/pred_vis/'

def predict(checkpoint ,data_path, output_path):
    """
    prediction
    """
    max_len = 24
    batch_size = 32

    checkpoint = torch.load(checkpoint)
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']
    
    decoder.eval()
    encoder.eval()

    dataset = GuiderDataset(data_path,0.2,max_len=max_len)
    pred_loader = Data.DataLoader(dataset.test_set(), batch_size=batch_size, shuffle=False) # pred 1 seq each time

    with torch.no_grad():
        for i, data in tqdm(enumerate(pred_loader)):
            # Move to device, if available
            imgs = data['image'].to(device)  # (b,c,w,h)
            enter = data['enter'].to(device)  # (b,2)
            esc = data['esc'].to(device) # (b,4)
            length = torch.full((batch_size,1),max_len,dtype=torch.long)

            encoder_out = encoder(imgs)
            
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)

            # Flatten image
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (b, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1) # for attention. not useful at the moment

            # Initialize LSTM state
            h, c = decoder.init_hidden_state(encoder_out, enter, esc)  # (batch_size, decoder_dim)

            # Create tensors to hold two coordination predictions
            predictions = torch.zeros((batch_size,max_len,2)).to(device)  # (b,max_len,2)
            predictions[:,0,:] = enter

            for t in range(max_len):
                h, c = decoder.decoder(
                    predictions[:,t,:],
                    (h, c))  # (batch_size_t, decoder_dim)
                preds = decoder.fc(decoder.dropout(h))  # (batch_size_t, 2)
                if t < max_len - 1:
                    predictions[:, t + 1, :] = preds # (b,max_len,2)
            
            output_dir = output_path + 'batch-{}/'.format(i)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            visualize(output_dir,imgs, predictions, enter, esc, length, 'pred')

if __name__ == '__main__':
    predict(checkpoint ,data_path, output_path)