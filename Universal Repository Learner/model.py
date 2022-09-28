import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import load_config

class ConvVAE(nn.Module):
    def __init__(self, config, cuda):
        super(ConvVAE, self).__init__()
        self.cuda = cuda
        self.hidden_size = config['hidden_size']
        self.z_size = config['z_size']

        self.batch_size = config['batch_size']

        self.in_channels = 1
        self.out_channels = 32
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.out_channels, self.out_channels*2, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*2, self.out_channels*2, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.out_channels*2, self.out_channels*4, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*4, self.out_channels*4, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*4, self.out_channels*4, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.out_channels*4, self.out_channels*8, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*8, self.out_channels*8, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*8, self.out_channels*8, self.kernel_size, self.stride, self.padding),
            nn.ReLU()
        )

        # self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc21 = nn.Linear(self.hidden_size, self.z_size)
        self.fc22 = nn.Linear(self.hidden_size, self.z_size)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.out_channels*8, self.out_channels*8, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*8, self.out_channels*8, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*8, self.out_channels*8, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.out_channels*8, self.out_channels*4, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*4, self.out_channels*4, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*4, self.out_channels*4, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.out_channels*4, self.out_channels*2, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels*2, self.out_channels*2, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.out_channels*2, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, self.kernel_size, self.stride, self.padding),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), self.out_channels*8, 4, 4)
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, reconstruction, input, mu, logvar):
        BCE = F.binary_cross_entropy(reconstruction, input, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE, KLD
