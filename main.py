# main file for TVR testing
from __future__ import print_function, division

# Providing path to working/project directory
import os
import sys
import math
import gzip
import pickle
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
from transforms.apply_patch import ApplyPatch

# Import custom defined modules
from image_resurfacer import Total_Variation_Resurfacer
from helper import test_Accuracy, select_Model, result_Log
from GAN import Generator, Discriminator


# Assign device as GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_all_seed(42)

# dictionary with the ImageNet label names
with open(os.path.join(os.getcwd(), "./assets/imagenet1000_clsidx_to_labels.txt")) as f:
    target_to_classname = eval(f.read())

# Load the patches
with gzip.open(os.path.join(os.getcwd(), "./assets/imagenet_patch.gz"), 'rb') as f:
    imagenet_patch = pickle.load(f)
patches, targets, info = imagenet_patch

# Mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# For convenience the preprocessing steps are splitted to compute also the clean predictions
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
inv_normalizer = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                        std=[1/0.229, 1/0.224, 1/0.255])

# Load the data
preprocess = Compose([Resize(256), CenterCrop(224), ToTensor()])    # ensure images are 224x224

_imagenette_classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
target_transforms = lambda y: _imagenette_classes[y]

set_all_seed(42)
# Test
test_dataset = ImageFolder('./assets/data/', transform = preprocess, target_transform = target_transforms)
test_idx = np.arange(len(test_dataset))
np.random.shuffle(test_idx)
testing_idx = test_idx[:50]
test_loader = DataLoader(test_dataset, batch_size=1, sampler=SubsetRandomSampler(testing_idx), num_workers=2)

# Metrics to Evaluate TVR
COLS = ['NAT_ACC_NAIVE', 'ADVER_ACC_NAIVE', 'SUCCESS_NAIVE', 'NAT_ACC_DEF',  'ADVER_ACC_DEF', 'SUCCESS_DEF']
# CNN Models List
#ROWS = ['alexnet', 'resnet18', 'squeezenet', 'vgg16', 'googlenet', 'inception_v3']
ROWS = ['resnet18']
# ImageNet-Patch Benchmark - Adversarial Patch List
patch_class = ['soap_dispenser', 'cornet', 'plate', 'banana', 'cup', 'typewriter', 'electric_guitar', 'hair_spray', 'sock', 'cellular_telephone']
model_names = ROWS
# Top-k Accuracy 
k=1
# 
#blk_list = [7, 14, 28, 56, 112]
blk_list = [28]
cls = 1
# Load Generator Model for image inpainting
PATH = './Model/netG.pt'
netG = Generator()
netG.load_state_dict(torch.load(PATH))
netG.eval()

for block_size in blk_list:

  patch = patches[cls]
  target = targets[cls]
  print(patch_class[cls], target)

  NAT_ACC_NAIVE, ADVER_ACC_NAIVE, SUCCESS_NAIVE = test_Accuracy(test_loader, patch, info, target, model_names, k, block_size, netG, defense=False)
  NAT_ACC_DEF, ADVER_ACC_DEF, SUCCESS_DEF = test_Accuracy(test_loader, patch, info, target, model_names, k, block_size, netG, defense=True)

  RES_SAVE_PATH = './results/' + patch_class[cls] + '_' + str(k) + '_blk_' + str(block_size) + '.csv'

  result_Log(RES_SAVE_PATH, ROWS, COLS, NAT_ACC_NAIVE, ADVER_ACC_NAIVE, SUCCESS_NAIVE, NAT_ACC_DEF, ADVER_ACC_DEF, SUCCESS_DEF)

