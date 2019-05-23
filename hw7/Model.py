import torch
import torch.nn as nn

class MyAutoEncoder(nn.Module):# 32
    def __init__(self):
        super(MyAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1),
            #nn.InstanceNorm2d(8, affine=True),
            nn.LeakyReLU(2e-3,True),
            #nn.MaxPool2d(2, stride=2, return_indices=True),

            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(2e-3,True),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(2e-3,True),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1),
            #nn.MaxPool2d(2, stride=2, return_indices=True)
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(2e-3,True),

            #nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64*4*4, 32),
            nn.LeakyReLU(2e-3,True),
            #nn.BatchNorm1d(64)
        )
        self.decoder_fc = nn.Sequential(
            #nn.BatchNorm1d(64),
            nn.Linear(32, 64*4*4),
            nn.ReLU(),
            nn.BatchNorm1d(64*4*4)
        )
        self.decoder = nn.Sequential(
            #nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxUnpool2d(2, stride=2),

            nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8,3,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        encoder_out_size = x.size()
        #print(x.size())
        x = self.encoder_fc(x.view(x.size(0), -1))
        x = self.decoder_fc(x).view(encoder_out_size)
        x = self.decoder(x)
        return x
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.encoder_fc(x.view(x.size(0), -1))
        return x
        '''
class MyAutoEncoderCNN(nn.Module):
    def __init__(self):
        super(MyAutoEncoderCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2,True),

            #nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.decoder = nn.Sequential(
            #nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(16,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxUnpool2d(2, stride=2),

            nn.ConvTranspose2d(32,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128,3,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        return x
'''
class MyAutoEncoderBest(nn.Module):
    def __init__(self):
        super(MyAutoEncoderBest, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1),
            #nn.InstanceNorm2d(8, affine=True),
            nn.LeakyReLU(2e-4,True),

            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(2e-4,True),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(2e-4,True),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1),
            #nn.MaxPool2d(2, stride=2, return_indices=True)
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(2e-4,True),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64*4*4, 64),
            nn.LeakyReLU(2e-4,True),
            #nn.BatchNorm1d(64)
        )
        self.decoder_fc = nn.Sequential(
            #nn.BatchNorm1d(64),
            nn.Linear(64, 64*4*4),
            nn.ReLU(),
            nn.BatchNorm1d(64*4*4)
        )
        self.decoder = nn.Sequential(
            #nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxUnpool2d(2, stride=2),

            nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8,3,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        encoder_out_size = x.size()
        #print(x.size())
        x = self.encoder_fc(x.view(x.size(0), -1))
        x = self.decoder_fc(x).view(encoder_out_size)
        x = self.decoder(x)
        return x
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.encoder_fc(x.view(x.size(0), -1))
        return x        

class MyAutoEncoder64(nn.Module):# 32
    def __init__(self):
        super(MyAutoEncoder64, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1),
            #nn.InstanceNorm2d(8, affine=True),
            nn.LeakyReLU(2e-3,True),
            #nn.MaxPool2d(2, stride=2, return_indices=True),

            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(2e-3,True),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(2e-3,True),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1),
            #nn.MaxPool2d(2, stride=2, return_indices=True)
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(2e-3,True),

            #nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64*4*4, 64),
            nn.LeakyReLU(2e-3,True),
            #nn.BatchNorm1d(64)
        )
        self.decoder_fc = nn.Sequential(
            #nn.BatchNorm1d(64),
            nn.Linear(64, 64*4*4),
            nn.ReLU(),
            nn.BatchNorm1d(64*4*4)
        )
        self.decoder = nn.Sequential(
            #nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxUnpool2d(2, stride=2),

            nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8,3,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        encoder_out_size = x.size()
        #print(x.size())
        x = self.encoder_fc(x.view(x.size(0), -1))
        x = self.decoder_fc(x).view(encoder_out_size)
        x = self.decoder(x)
        return x
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.encoder_fc(x.view(x.size(0), -1))
        return x

class MyAutoEncoder128(nn.Module):# 32
    def __init__(self):
        super(MyAutoEncoder128, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1),
            #nn.InstanceNorm2d(8, affine=True),
            nn.LeakyReLU(2e-3,True),
            #nn.MaxPool2d(2, stride=2, return_indices=True),

            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(2e-3,True),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(2e-3,True),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1),
            #nn.MaxPool2d(2, stride=2, return_indices=True)
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(2e-3,True),

            #nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64*4*4, 128),
            nn.LeakyReLU(2e-3,True),
            #nn.BatchNorm1d(64)
        )
        self.decoder_fc = nn.Sequential(
            #nn.BatchNorm1d(64),
            nn.Linear(128, 64*4*4),
            nn.ReLU(),
            nn.BatchNorm1d(64*4*4)
        )
        self.decoder = nn.Sequential(
            #nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxUnpool2d(2, stride=2),

            nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8,3,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        encoder_out_size = x.size()
        #print(x.size())
        x = self.encoder_fc(x.view(x.size(0), -1))
        x = self.decoder_fc(x).view(encoder_out_size)
        x = self.decoder(x)
        return x
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.encoder_fc(x.view(x.size(0), -1))
        return x