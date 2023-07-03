# Importing libraries
import torch
import torch.nn as nn

# Defining a simple and custom GAN architecture for image inpainting
class Generator(nn.Module):
    """GENERATOR MODEL"""
    def __init__(self):
        super(Generator,self).__init__()
        # ENCODER
        self.t1=nn.Sequential( nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
                nn.LeakyReLU(0.2))
        self.t2=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2))
        self.t3=nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2))
        self.t4=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2))
        self.t5=nn.Sequential(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2))
        # DECODER
        self.t6=nn.Sequential(nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU())
        self.t7=nn.Sequential(nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU())
        self.t8=nn.Sequential(nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.t9=nn.Sequential(nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.t10=nn.Sequential(nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(4,4),stride=2,padding=1),
                nn.Tanh())

    def forward(self,x):
        x=self.t1(x)
        x=self.t2(x)
        x=self.t3(x)
        x=self.t4(x)
        x=self.t5(x)
        x=self.t6(x)
        x=self.t7(x)
        x=self.t8(x)
        x=self.t9(x)
        x=self.t10(x)
        return x #output of generator


class Discriminator(nn.Module):
    """DISCRIMINATOR MODEL"""
    def __init__(self):
        super(Discriminator,self).__init__()
        self.t1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(4,4),stride=2,padding=1),
                nn.LeakyReLU(0.2))
        self.t2=nn.Sequential(nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2))
        self.t3=nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2))
        self.t4=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2))
        self.t5=nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2))
        self.t6=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512, kernel_size=(4,4),stride=1,padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2))
        self.t7=nn.Sequential(nn.Conv2d(in_channels=512,out_channels=1,kernel_size=(4,4),stride=1,padding=0),
                nn.Sigmoid())
    
    def forward(self,x):
        x=self.t1(x)
        x=self.t2(x)
        x=self.t3(x)
        x=self.t4(x)
        x=self.t5(x)
        x=self.t6(x)
        x=self.t7(x)
        return x #output of discriminator

