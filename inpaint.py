# Training GAN model for image reconstruction in TVR
from __future__ import print_function, division

# Providing path to working/project directory
import os
import sys
import math
import random
import numpy as np
import pandas as pd

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

# Import PyTorch's vision libraries
import torchvision
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, models, transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Import libraries from ImageNet-Patch benchmark GitHub
from utils.utils import set_all_seed
from utils.utils import target_transforms

# Import custom defined modules
from GAN import Generator, Discriminator

# Assign device as GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_all_seed(42)

# For convenience the preprocessing steps are splitted to compute also the clean predictions
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
inv_normalizer = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                    std=[1/0.229, 1/0.224, 1/0.255])

# Load the data
preprocess = Compose([Resize(256), CenterCrop(224), ToTensor()])    # ensure images are 224x224
mask_preprocess = Compose([Resize(224), ToTensor()])

_imagenette_classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
target_transforms = lambda y: _imagenette_classes[y]

# Train
real_dataset = ImageFolder('./real_data/train/', transform = preprocess, target_transform = target_transforms)
real_loader = DataLoader(real_dataset, batch_size=64, num_workers=2)
mask_dataset = ImageFolder('./mask_data/', transform = mask_preprocess, target_transform = target_transforms)
mask_loader = DataLoader(mask_dataset, batch_size=64, num_workers=2)
# Dataset iterator
real_iter = iter(real_loader)
mask_iter = iter(mask_loader)
lbl_dict = {0: 'n01440764', 217: 'n02102040', 482:'n02979186', 491:'n03000684', 497:'n03028079', 566:'n03394916', \
             569:'n03417042', 571:'n03425413', 574:'n03445777', 701:'n03888257'}

# Parameters of GAN training
epochs=300
Batch_Size=64
lr=0.0002
beta1=0.5

# Make directory to save generated and input samples
try:
    os.makedirs("Images/train/cropped")
    os.makedirs("Images/train/real")
    os.makedirs("Images/train/recon")
    os.makedirs("Model")
except OSError:
    pass

wtl2 = 0.999
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

resume_epoch=0
# Initializing the GAN network
netG = Generator()
netG.apply(weights_init)
netD = Discriminator()
netD.apply(weights_init)

# Loss for training
criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

# Initialize the tensors
input_real = torch.FloatTensor(Batch_Size, 3, 224, 224)
input_cropped = torch.FloatTensor(Batch_Size, 3, 224, 224)
label = torch.FloatTensor(Batch_Size)
real_label = 1
fake_label = 0

# Loading Tensors and GAN onto GPU
netD.cuda()
netG.cuda()
criterion.cuda()
criterionMSE.cuda()
input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

# Defining optimizers for training
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(resume_epoch,epochs):
    # Iterator
    real_iter = iter(real_loader)
    mask_iter = iter(mask_loader)
    for i in range(0,len(real_loader)):
        real_cpu, real_y = next(real_iter)
        real_cpu = real_cpu.to(device)
        mask_cpu, mask_y = next(mask_iter)
        mask_cpu = mask_cpu.to(device)
        batch_size = real_cpu.size(0)
        with torch.no_grad():
            input_real.resize_(real_cpu.size()).copy_(real_cpu)
            input_cropped.resize_(mask_cpu.size()).copy_(mask_cpu)

        # Train discriminator with real data
        netD.zero_grad()
        with torch.no_grad():
            label.resize_(batch_size).fill_(real_label)
        output = netD(input_real)
        label =  label.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()
        
        # Train discriminator with fake data
        fake = netG(input_cropped)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Train the generator now
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG_D = criterion(output, label)
        wtl2Matrix = real_cpu.clone()
        wtl2Matrix.data.fill_(wtl2*10)
        #wtl2Matrix.data[:,:,int(over):int(128/2 - over),int(over):int(128/2 - over)] = wtl2
        errG_l2 = (fake-real_cpu).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()
        errG = (1-wtl2) * errG_D + wtl2 * errG_l2
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
    
        recon_image = fake
        print('[%d / %d][%d / %d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'% (epoch, epochs, i, len(real_loader),
              errD.data, errG_D.data,errG_l2.data, D_x,D_G_z1, ))
        
        # Saving sampled images into required directory
        if i % 100 == 0:
            vutils.save_image(real_cpu,'Images/train/real/real_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(input_cropped.data,'Images/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(recon_image.data,'Images/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))

torch.save(netG.state_dict(), './Model/netG.pt')

