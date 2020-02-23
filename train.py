#coding:utf-8
import time
import torch.utils.data as Data
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, Decoder
from dataset import *
from utils import *
from vis import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
torch.backends.cudnn.benchmark = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)

data_path = '/data/lzt/project/waysguider/dataset'
decoder_dim = 1024
dropout = 0.5

start_epoch = 1
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_loss = 1.  # best loss score right now
print_freq = 1  # print training/validation stats every __ batches
fine_tune_encoder = False # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
save_path = './checkpoint/' # checkpoint save path
vis_dir = './vis/' # store visualized result

max_len = 12 # the longest sequence

# calculate 
lambd = 1.
convsize = 7
std = 5

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, lambd, convsize, std):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    encoder.train()
    decoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    wayslosses = AverageMeter()

    start = time.time()

    for i,data in enumerate(train_loader):
        img_name = data['name']
        imgs = data['image'].to(device) # (b,c,w,h)
        seq = data['seq'].to(device) # (b,max_len,2)
        seq_inv = data['seq_inv'].to(device)
        enter = data['enter'].to(device) # (b,2)
        esc = data['esc'].to(device) # (b,4) one-hot indicate four direction
        length = data['len'] # (b) it seem to be a 1D CPU int64 tensor when use pack_padded_sequence below

        #skip = [10,30,31,59,65,89]
        #if i in skip:
        #if i in skip:
        #    print(img_name)
        #    continue
        if i == 117:
            continue

        data_time.update(time.time() - start)

        # Forward prop.
        imgs = encoder(imgs) # encoder_out
        pred, pred_inv,_ , sort_ind = decoder(imgs, enter, esc, seq[:,:-1,:], seq_inv[:,:-1,:], length-1)
        # pred (b,max_len,2)

        targets = seq[sort_ind,1:,:] # to the sorted version
        targets_inv = seq_inv[sort_ind,1:,:]
        # Remove timesteps that we didn't decode at, or are pads
        #pred = pack_padded_sequence(pred, length.squeeze(1), batch_first=True)
        #targets = pack_padded_sequence(targets, length.squeeze(1), batch_first=True)
        
        # used to calculate the loss of coordinates away from ways
        reference = imgs.detach().permute(0,3,1,2) # (b, 1, encoded_image_size, encoded_image_size)
        waysloss = cal_waysloss(reference, pred, pred_inv, convsize, std, device)
        # Calculate loss
        loss = criterion(pred, targets) + criterion(pred_inv, targets_inv) + lambd * waysloss
        #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()

        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), length.sum().item())
        wayslosses.update(waysloss.item(), length.sum().item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}] [{1}/{2}]\n'
                  'Batch Time {batch_time.val:.3f}s (Average:{batch_time.avg:.3f}s)\n'
                  'Data Load Time {data_time.val:.3f}s (Average:{data_time.avg:.3f}s)\n'
                  'Loss {loss.val:.4f} (Average:{loss.avg:.4f})\n'
                  'waysloss {waysloss.val:.4f} (Average:{waysloss.avg:.4f})\n'
                  .format(epoch, i, len(train_loader),batch_time=batch_time,
                          data_time=data_time, loss=losses, waysloss=wayslosses))


def validate(val_loader, encoder, decoder, criterion, lambd, convsize, std, device):
    
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    wayslosses = AverageMeter()

    start = time.time()

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        for i, data in enumerate(val_loader):
            # Move to device, if available
            if i == 23:
                continue
            imgs = data['image'].to(device)  # (b,c,w,h)
            seq = data['seq'].to(device)  # (b,max_len,2)
            seq_inv = data['seq_inv'].to(device)
            enter = data['enter'].to(device)  # (b,2)
            esc = data['esc'].to(device)  # (b,4)
            length = data['len']  # (b)  it seem to be a 1D CPU int64 tensor

            # Forward prop.
            if encoder is not None:
                imgs_encode = encoder(imgs)
            pred, pred_inv,predictions_assemble, sort_ind = decoder(imgs_encode, enter, esc, seq[:,:-1,:], seq_inv[:,:-1,:], length - 1)

            targets = seq[sort_ind,1:,:]
            targets_inv = seq_inv[sort_ind,1:,:]

            #pred_cal = pred.clone()
            #pred_cal = pack_padded_sequence(pred_cal, length.squeeze(1), batch_first=True)
            #targets = pack_padded_sequence(targets, length.squeeze(1), batch_first=True)
            
            reference = imgs_encode.detach().permute(0,3,1,2) # (b, 1,encoded_image_size, encoded_image_size)
            waysloss = cal_waysloss(reference, pred, pred_inv, convsize, std, device)
            # Calculate loss
            loss = criterion(pred, targets) + criterion(pred_inv, targets_inv) + lambd * waysloss

            # Keep track of metrics
            losses.update(loss.item(),length.sum().item())
            wayslosses.update(waysloss.item(),length.sum().item())
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\n'
                      'Batch Time {batch_time.val:.3f}s (Average:{batch_time.avg:.3f}s)\n'
                      'Loss {loss.val:.4f} (Average:{loss.avg:.4f})\n'
                      'waysloss {waysloss.val:.4f} (Average:{waysloss.avg:.4f})\n'
                      .format(i, len(val_loader), batch_time=batch_time,loss=losses, waysloss=wayslosses))
                
    return losses.avg, imgs[sort_ind,:,:,:], pred, predictions_assemble, enter[sort_ind,:], esc[sort_ind,:], length[sort_ind,:]

def main():
    global epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, best_loss, save_path, vis_dir, decoder_dim, lambd, convsize, std

    if checkpoint is None:
        decoder = Decoder(decoder_dim)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['testLoss']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr)

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.MSELoss().to(device)
    #criterion = traj_loss().to(device)

    dataset = GuiderDataset(data_path,0.2,max_len=max_len)
    train_loader = Data.DataLoader(dataset.train_set(), batch_size=batch_size, shuffle=False)
    val_loader = Data.DataLoader(dataset.test_set(), batch_size=batch_size, shuffle=False)
    
    for epoch in range(start_epoch, start_epoch + epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch, lambd=lambd, convsize=convsize, std=std)

        # One epoch's validation, return the average loss of each batch in this epoch
        loss, imgs, pred, pred_vis, enter, esc, length = validate(val_loader=val_loader,
                                    encoder=encoder, decoder=decoder, criterion=criterion,
                                    lambd=lambd, convsize=convsize, std=std, device=device)
        # visualize the last batch of validate epoch
        visualize(vis_dir, imgs, pred_vis, None, None, None, enter, esc, length, epoch)

        # Check if there was an improvement
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(save_path, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, loss, is_best)
        
if __name__ == '__main__':
    main()
