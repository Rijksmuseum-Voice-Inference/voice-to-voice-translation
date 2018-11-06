import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def concat_x_c(x, c):
    # Replicate spatially and concatenate domain information.
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, x.size(2), x.size(3))
    x = torch.cat([x, c], dim=1)
    return x

class GatedBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding=0, deconv=False):
        super(GatedBlock, self).__init__()
        conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.conv = conv(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_conv = nn.BatchNorm2d(dim_out)
        self.gate = conv(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_gate = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x1 = self.bn_conv(self.conv(x))
        x2 = F.sigmoid(self.bn_gate(self.gate(x)))
        out = x1 * x2
        return out

class NoCompressGenerator16(nn.Module):
    """Generator network. Latent size (8, 64, 16)"""
    def __init__(self, c_dim):
        super(NoCompressGenerator16, self).__init__()
        encoder = []

        # my dims (reduces length dim by 2, keeps features dim until last conv, like in StarGAN-VC)
        encoder.append(GatedBlock(1, 32, kernel_size=(9,3), stride=(1,1), padding=(4,1)))
        encoder.append(GatedBlock(32, 64, kernel_size=(8,4), stride=(2,2), padding=(3,1)))
        encoder.append(GatedBlock(64, 128, kernel_size=(8,4), stride=(2,2), padding=(3,1)))
        encoder.append(GatedBlock(128, 8, kernel_size=(5,3), stride=(1,1), padding=(2,1)))

        self.l1 = GatedBlock(8 + c_dim, 128, kernel_size=(5,3), stride=(1,1), padding=(2,1), deconv=True)
        self.l2 = GatedBlock(128, 64, kernel_size=(8,4), stride=(2,2), padding=(3,1), deconv=True)
        self.l3 = GatedBlock(64, 32, kernel_size=(8,4), stride=(2,2), padding=(3,1), deconv=True)
        self.l4 = nn.ConvTranspose2d(32, 1, kernel_size=(9,3), stride=(1,1), padding=(4,1))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x, c):
        x = self.encoder(x)

        x = concat_x_c(x, c)
        x = self.l1(x)
        x = concat_x_c(x, c)
        x = self.l2(x)
        x = concat_x_c(x, c)
        x = self.l3(x)
        x = concat_x_c(x, c)
        x = self.l4(x)
        return x

class NoCompressGenerator32(nn.Module):
    """Generator network. Latent size (8, 64, 32)"""
    def __init__(self, c_dim):
        super(NoCompressGenerator32, self).__init__()
        encoder = []

        # my dims (reduces length dim by 2, keeps features dim until last conv, like in StarGAN-VC)
        encoder.append(GatedBlock(1, 32, kernel_size=(9,3), stride=(1,1), padding=(4,1)))
        encoder.append(GatedBlock(32, 64, kernel_size=(8,4), stride=(2,2), padding=(3,1)))
        encoder.append(GatedBlock(64, 128, kernel_size=(8,3), stride=(2,1), padding=(3,1)))
        encoder.append(GatedBlock(128, 8, kernel_size=(5,3), stride=(1,1), padding=(2,1)))

        self.l1 = GatedBlock(8 + c_dim, 128, kernel_size=(5,3), stride=(1,1), padding=(2,1), deconv=True)
        self.l2 = GatedBlock(128 + c_dim, 64, kernel_size=(8,3), stride=(2,1), padding=(3,1), deconv=True)
        self.l3 = GatedBlock(64 + c_dim, 32, kernel_size=(8,4), stride=(2,2), padding=(3,1), deconv=True)
        self.l4 = nn.ConvTranspose2d(32 + c_dim, 1, kernel_size=(9,3), stride=(1,1), padding=(4,1))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x, c):
        x = self.encoder(x)

        x = concat_x_c(x, c)
        x = self.l1(x)
        x = concat_x_c(x, c)
        x = self.l2(x)
        x = concat_x_c(x, c)
        x = self.l3(x)
        x = concat_x_c(x, c)
        x = self.l4(x)
        return x

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, c_dim):
        super(Generator, self).__init__()
        encoder = []
        decoder = []

        # my dims (reduces length dim by 2, keeps features dim until last conv, like in StarGAN-VC)
        encoder.append(GatedBlock(1, 32, kernel_size=(9,3), stride=(1,1), padding=(4,1)))
        encoder.append(GatedBlock(32, 64, kernel_size=(8,4), stride=(2,2), padding=(3,1)))
        encoder.append(GatedBlock(64, 128, kernel_size=(8,4), stride=(2,2), padding=(3,1)))
        encoder.append(GatedBlock(128, 64, kernel_size=(5,3), stride=(1,1), padding=(2,1)))
        encoder.append(GatedBlock(64, 5, kernel_size=(5,16), stride=(1,16), padding=(2,1)))

        decoder.append(GatedBlock(5 + c_dim, 64, kernel_size=(5,16), stride=(1,16), padding=(2,0), deconv=True))
        decoder.append(GatedBlock(64, 128, kernel_size=(5,3), stride=(1,1), padding=(2,1), deconv=True))
        decoder.append(GatedBlock(128, 64, kernel_size=(8,4), stride=(2,2), padding=(3,1), deconv=True))
        decoder.append(GatedBlock(64, 32, kernel_size=(8,4), stride=(2,2), padding=(3,1), deconv=True))
        decoder.append(nn.ConvTranspose2d(32, 1, kernel_size=(9,3), stride=(1,1), padding=(4,1)))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, c):
        x = self.encoder(x)
        x = concat_x_c(x, c)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, c_dim):
        super(Discriminator, self).__init__()
        # my dims (reduces length dim by 2, keeps features dim until last conv, like in StarGAN-VC)
        self.l1 = GatedBlock(1 + c_dim, 32, kernel_size=(9,3), stride=(1,1), padding=(4,1))
        self.l2 = GatedBlock(32 + c_dim, 32, kernel_size=(8,3), stride=(2,1), padding=(3,1))
        self.l3 = GatedBlock(32 + c_dim, 32, kernel_size=(8,3), stride=(2,1), padding=(3,1))
        self.l4 = GatedBlock(32 + c_dim, 32, kernel_size=(6,3), stride=(2,1), padding=(2,1))
        self.l5 = nn.Conv2d(32 + c_dim, 1, kernel_size=(5,40), stride=(1,40), padding=(2,0), bias=False)

    def forward(self, x, c):
        x = concat_x_c(x, c)
        x = self.l1(x)
        x = concat_x_c(x, c)
        x = self.l2(x)
        x = concat_x_c(x, c)
        x = self.l3(x)
        x = concat_x_c(x, c)
        x = self.l4(x)
        x = concat_x_c(x, c)
        x = self.l5(x)

        x = F.sigmoid(x)
        x = x.mean(2)
        x = x.squeeze(2)
        return x

class Classifier(nn.Module):
    def __init__(self, c_dim, dropout=0.5):
        super(Classifier, self).__init__()
        layers = []

        layers.append(GatedBlock(1, 8, kernel_size=(4,6), stride=(2,4), padding=(1,1)))
        layers.append(nn.Dropout2d(dropout))
        layers.append(GatedBlock(8, 8, kernel_size=(4,6), stride=(2,2), padding=(1,1)))
        layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(8, c_dim, kernel_size=(3,5), stride=(2,5), padding=(1,1), bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # now x of size (B, 1, SEQ_LEN, N_FEATS)

        x = self.layers(x)
        # now x is of size (B, #CLASSES, 8, 1)

        x = F.log_softmax(x, dim=1)
        x = x.mean(2)
        x = x.squeeze(2)
        return x

class UnifiedDiscriminator(nn.Module):
    def __init__(self, c_dim, c_do):
        super(UnifiedDiscriminator, self).__init__()

        self.D = Discriminator(c_dim)
        self.C = Classifier(c_dim, c_do)

    def forward(self, x, c):
        return self.D(x, c), self.C(x)
