# Generating masked data for training Generator for image inpainting
from __future__ import print_function, division

import os
import sys
import math
import shutil
import numpy as np
import pandas as pd

# Import PyTorhc's libraries
import torch
import torch.nn as nn
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

# Import customed defined modules
from image_resurfacer import Total_Variation_Resurfacer

# Assign device as GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the patches
with gzip.open(os.path.join(os.getcwd(), "assets/imagenet_patch.gz"), 'rb') as f:
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
# Train
train_dataset = ImageFolder('./real_data/train', transform = preprocess, target_transform = target_transforms)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=2)
# Test
test_dataset = ImageFolder('./real_data/test', transform = preprocess, target_transform = target_transforms)
test_idx = np.arange(len(test_dataset))
np.random.shuffle(test_idx)
testing_idx = test_idx[:50]
test_loader = DataLoader(test_dataset, batch_size=1, sampler=SubsetRandomSampler(testing_idx), num_workers=2)

lbl_dict = {0: 'n01440764', 217: 'n02102040', 482:'n02979186', 491:'n03000684', 497:'n03028079', 566:'n03394916', \
             569:'n03417042', 571:'n03425413', 574:'n03445777', 701:'n03888257'}

BASE_PATH = './mask_data/'
# define a transform to convert a tensor to PIL image
transform = transforms.ToPILImage()
patch = patches[1]

apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                        translation_range=(.2, .2),
                        rotation_range=(-45, 45),
                        scale_range=(0.7, 1))
patch_normalizer = Compose([apply_patch, normalizer])
patch_applier = Compose([apply_patch])

for batch_idx, (image,label) in enumerate(train_loader):

  FINAL_PATH = BASE_PATH+lbl_dict[label.item()]+'/'+str(batch_idx)+'.JPEG'
  if batch_idx%50==0:
      print(batch_idx)
  image_adv = patch_applier(image.cpu()).to(device)
  image_adv = total_Variation_Defense(image_adv)
  # convert the tensor to PIL image using above transform
  pil = transform(image_adv[0])
  # display the PIL image
  pil.save(FINAL_PATH,"JPEG")
