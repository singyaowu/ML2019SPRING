import torch
import torch.nn as nn

class MyAutoEncoder(nn.Module):
    def __init__(self):
        super(MyAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1),
            #nn.InstanceNorm2d(8, affine=True),
            nn.LeakyReLU(2e-4,True),
            #nn.MaxPool2d(2, stride=2, return_indices=True),

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

            #nn.MaxPool2d(2, stride=2, return_indices=True)
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
        self.unpool = nn.MaxUnpool2d(2, stride=2)
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
