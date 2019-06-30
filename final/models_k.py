import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PFFNet(nn.Module):
    def __init__(self):
        super(PFFNet, self).__init__()

        # Encoder
        self.conv_16 = nn.Sequential(
            nn.ReflectionPad2d(padding=5),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=11, stride=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_32 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_64 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_128 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_256 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )

        # Feature transformer
        modules = [ResidualBlock(256) for _ in range(18)]
        self.resblock = nn.Sequential(*modules)

        # Decoder
        self.dconv_128 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.dconv_64 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.dconv_32 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.dconv_16 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_output = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1),
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        enc_16 = self.conv_16(x)
        enc_32 = self.conv_32(enc_16)
        enc_64 = self.conv_64(enc_32)
        enc_128 = self.conv_128(enc_64)
        enc_256 = self.conv_256(enc_128)

        # Feature transformer
        residual = enc_256
        dec_256 = self.resblock(enc_256)
        dec_256 += residual

        # Decoder
        dec_128 = self.dconv_128(dec_256)
        dec_128 = F.interpolate(dec_128, size=(enc_128.size(2), enc_128.size(3)), mode="bilinear", align_corners=False)
        dec_128 += enc_128
        
        dec_64 = self.dconv_64(dec_128)
        dec_64 = F.interpolate(dec_64, size=(enc_64.size(2), enc_64.size(3)), mode="bilinear", align_corners=False)
        dec_64 += enc_64

        dec_32 = self.dconv_32(dec_64)
        dec_32 = F.interpolate(dec_32, size=(enc_32.size(2), enc_32.size(3)), mode="bilinear", align_corners=False)
        dec_32 += enc_32

        dec_16 = self.dconv_16(dec_32)
        dec_16 = F.interpolate(dec_16, size=(enc_16.size(2), enc_16.size(3)), mode="bilinear", align_corners=False)
        dec_16 += enc_16

        out = self.conv_output(dec_16)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(channels),
        )
        self.PReLU = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out) #* 0.1
        out += residual

        return self.PReLU(out)

class PFFNet_mine(nn.Module):
    def __init__(self):
        super(PFFNet_mine, self).__init__()

        # Encoder
        self.conv_16 = nn.Sequential(
            nn.ReflectionPad2d(padding=5),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=11, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_32 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_64 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_128 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_256 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )

        # Feature transformer
        modules = [ResidualBlock(256) for _ in range(18)]
        self.resblock = nn.Sequential(*modules)

        # Decoder
        self.dconv_128 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.dconv_64 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.dconv_32 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.dconv_16 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_output = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1),
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        enc_16 = self.conv_16(x)
        enc_32 = self.conv_32(enc_16)
        enc_64 = self.conv_64(enc_32)
        enc_128 = self.conv_128(enc_64)
        enc_256 = self.conv_256(enc_128)

        # Feature transformer
        residual = enc_256
        dec_256 = self.resblock(enc_256)
        dec_256 += residual

        # Decoder
        dec_128 = self.dconv_128(dec_256)
        dec_128 = F.interpolate(dec_128, size=(enc_128.size(2), enc_128.size(3)), mode="bilinear", align_corners=False)
        dec_128 += enc_128
        
        dec_64 = self.dconv_64(dec_128)
        dec_64 = F.interpolate(dec_64, size=(enc_64.size(2), enc_64.size(3)), mode="bilinear", align_corners=False)
        dec_64 += enc_64

        dec_32 = self.dconv_32(dec_64)
        dec_32 = F.interpolate(dec_32, size=(enc_32.size(2), enc_32.size(3)), mode="bilinear", align_corners=False)
        dec_32 += enc_32

        dec_16 = self.dconv_16(dec_32)
        dec_16 = F.interpolate(dec_16, size=(enc_16.size(2), enc_16.size(3)), mode="bilinear", align_corners=False)
        dec_16 += enc_16

        out = self.conv_output(dec_16)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 5, 1, 2),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 1, 2),
            nn.ReLU()
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, 5, 1, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class AOD_net(nn.Module):

    def __init__(self):
        super(AOD_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
    
        self.e_conv1 = nn.Conv2d(3,16,1,1,0,bias=True) 
        self.e_conv2 = nn.Conv2d(16,16,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(32,32,5,1,2,bias=True) 
        self.e_conv4 = nn.Conv2d(48,48,7,1,3,bias=True) 
        self.e_conv5 = nn.Conv2d(112,3,3,1,1,bias=True) 
        
    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        x3 = self.relu(self.e_conv3(torch.cat((x1, x2), 1)))

        x4 = self.relu(self.e_conv4(torch.cat((x2, x3), 1)))

        x5 = self.relu(self.e_conv5(torch.cat((x1, x2, x3, x4),1)))

        clean_image = self.relu((x5 * x) - x5 + 1) 
        
        return clean_image
