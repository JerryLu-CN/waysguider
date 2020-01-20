#coding:utf-8
import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=16, fine_tune=True):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet50(pretrained=True)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        if fine_tune:
            self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Decoder(nn.Module):
    """Decoder"""
    def __init__(self,decoder_dim, condition_dim=128, encoder_dim=2048, dropout=0.5):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.condition_dim = condition_dim # enter point coordination / esc direction
        self.decoder_dim = decoder_dim

        self.dropout = nn.Dropout(p=dropout)
        self.decoder = nn.LSTMCell(2, decoder_dim,bias=True) # 2 indicates (X,Y)
        self.init_condition = nn.Linear(6,self.condition_dim)
        self.init_h = nn.Linear(encoder_dim + condition_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim + condition_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, 2) # regression question
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out, enter, esc):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param enter: enter point
        :param esc: escape point
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1) # (batch_size, encoder_dim)
        condition = torch.cat([enter,esc],dim=1) # (batch_size,4)
        condition_embedding = self.init_condition(condition)  # (batch_size,condition_dim)
        # 这里或许可以通过增加注意机制来平衡来自图片和出入点的信息权重
        h = self.init_h(torch.cat([mean_encoder_out,condition_embedding],dim=1)) # (batch_size, decoder_dim)
        c = self.init_c(torch.cat([mean_encoder_out,condition_embedding],dim=1))
        return h, c

    def forward(self, encoder_out, enter, esc, sequence, seq_len):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param enter: enter point (b,2)
        :param esc: escape point (b,2)
        :param sequence: coodination sequence (batch_size, max_seq_len, 2)
        :param seq_len: sequence length (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1) # for attention. not useful at the moment

        # Sort input data by decreasing lengths; why? apparent below
        seq_len, sort_ind = seq_len.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        sequence = sequence[sort_ind]

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out, enter, esc)  # (batch_size, decoder_dim)

        # Create tensors to hold two coordination predictions
        predictions = torch.zeros(sequence.shape).to(device)  # (b,max_len,2)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(sequence.shape[1]):
            batch_size_t = sum([l > t for l in seq_len])
            h, c = self.decoder(
                sequence[:batch_size_t,t,:],
                (h[:batch_size_t,:], c[:batch_size_t,:]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, 2)
            predictions[:batch_size_t, t, :] = preds # (b,max_len,2)

        return predictions, sort_ind
