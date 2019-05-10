import torch
import torch.nn as nn

class MyAutoEncoder(nn.Module):
    def __init__(self):
        super(MyAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2, return_indices=True),
            nn.BatchNorm2d(6),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3),
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2, return_indices=True)

            nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.decoder = nn.Sequential(
            #nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(16,6,kernel_size=3),
            nn.ReLU(True),
            #nn.MaxUnpool2d(2, stride=2),
            nn.BatchNorm2d(6),
            nn.ConvTranspose2d(6,3,kernel_size=3),
            #nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        x, indices = self.encoder(x)
        x = self.unpool(x, indices)
        x = self.decoder(x)
        return x

class MyEncoder(nn.Module):
    def __init__(self):
        super(MyEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2, return_indices=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.decoder1 = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(6,16,kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
        )
        self.decoder2 = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(16,3,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Sigmoid()
        )
    def forward(self, x):
        x, indices1 = self.encoder1(x)
        x, indices2 = self.encoder1(x)
        x = self.unpool(x, indices)
        x = self.decoder(x)
        return x


class MyDecoder(nn.Module):
    def __init__(self):
        super(MyDecoder, self).__init__()

        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.decoder = nn.Sequential(
            #nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(16,6,kernel_size=3),
            nn.ReLU(True),
            #nn.MaxUnpool2d(2, stride=2),
            nn.BatchNorm2d(6),
            nn.ConvTranspose2d(6,3,kernel_size=3),
            #nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        x, indices = self.encoder(x)
        x = self.unpool(x, indices)
        x = self.decoder(x)
        return x