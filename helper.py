# File containing utilities

# Importing libraries
from __future__ import print_function, division
# Import general Python libraries
import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import PyTorch vision libraries
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Import from ImageNet-patch benchmark github files
from utils.utils import set_all_seed
from utils.utils import target_transforms
from transforms.apply_patch import ApplyPatch

# Import custom defined modules
from GAN import Generator, Discriminator
from image_resurfacer import Total_Variation_Resurfacer

set_all_seed(42)
# Assigning device as GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dictionary with the ImageNet label names
with open(os.path.join(os.getcwd(), "assets/imagenet1000_clsidx_to_labels.txt")) as f:
    target_to_classname = eval(f.read())

# Mean and Standard deviation of ImageNet
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])

def select_Model(model_name, gpu=True):
    """Assign the desired pre-trained model and load it onto GPU"""
    print(model_name)
    # Assign pre-trained model
    model_name = model_name.lower()
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'squeezenet':
        model = models.squeezenet1_0(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    model.eval()
    # Load model onto the GPU
    if gpu:
        model = model.to(device)

    return model

def test_Accuracy(test_loader, patch, info, target, model_names, k, block_shape, netG, defense):
    """Testing the accuracy of defense over the test-set"""
    # Initialize list for tracking the accuracy
    clean_acc = []
    adv_acc = []
    success_rate = []
    # Define Expectation over Transformation for placing adversarial patch
    apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                          translation_range=(.2, .2),
                          rotation_range=(-45, 45),  
                          scale_range=(0.7, 1))  
    # Define patch applier with and without normalization
    patch_normalizer = Compose([apply_patch, normalizer])
    patch_applier = Compose([apply_patch])
    set_all_seed(42)

    # No computation graph for gradient during testing
    with torch.no_grad():
        # Calculate accuracy for each model
        for i in range(0, len(model_names)):
            model = select_Model(model_names[i])
            correct_clean = 0
            correct_adv = 0
            n_samples = 0
            n_success = 0
            for batch_idx, (x,y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                n_samples += x.shape[0]
                if batch_idx%10==0:
                    print(batch_idx)

                # Calculate accuracy clean samples
                clean_image = x
                #### Check if TVR Defense is available
                if defense:
                    ######################   DEFENSE STEPS  #######################
                    TVR =  Total_Variation_Resurfacer(clean_image, block_shape)
                    TVR.Image_to_Block()
                    TVR.calculate_TV_Score()
                    TVR.outlier_Detection()
                    TVR.obsfucated_Image()
                    clean_image = TVR.reconstructed_Image(netG)
                # Prediction
                clean_image = normalizer(clean_image).to(device)
                clean_pred = F.softmax(model(clean_image).data, dim=1).data[0]
                clean_topk = torch.topk(clean_pred, k)[1]
                if y in clean_topk:
                    correct_clean += 1

                # Calculate accuracy over adversarially patched samples
                adv_image = patch_applier(x.cpu()).to(device)
                # Check if TVR Defense is available
                if defense:
                    ######################   DEFENSE STEPS  #######################
                    TVR =  Total_Variation_Resurfacer(adv_image, block_shape)
                    TVR.Image_to_Block()
                    TVR.calculate_TV_Score()
                    TVR.outlier_Detection()
                    TVR.obsfucated_Image()
                    adv_image = TVR.reconstructed_Image(netG)
                # Prediction
                adv_image = normalizer(adv_image).to(device)
                adv_pred = F.softmax(model(adv_image).data, dim=1).data[0]
                adv_topk = torch.topk(adv_pred, k)[1]
                if y in adv_topk:
                    correct_adv += 1  
                if target in adv_topk:
                    n_success += 1

            clean_acc.append(round(100*(correct_clean/n_samples),2))
            adv_acc.append(round(100*(correct_adv/n_samples),2))
            success_rate.append(round(100*(n_success / n_samples),2))

    return clean_acc, adv_acc, success_rate

def result_Log(RES_SAVE_PATH, ROWS, COLS, NAT_ACC_NAIVE, ADVER_ACC_NAIVE, SUCCESS_NAIVE, NAT_ACC_DEF, ADVER_ACC_DEF, SUCCESS_DEF):
    """Log accuracy results (with and without TVR Defense) as a Dataframe into a .csv file"""
    # Initialise DataFrame 
    result_df = pd.DataFrame(index=ROWS, columns=COLS)
    # Without TVR Defense (Naive Model)
    result_df['NAT_ACC_NAIVE'] = NAT_ACC_NAIVE
    result_df['ADVER_ACC_NAIVE'] = ADVER_ACC_NAIVE
    result_df['SUCCESS_NAIVE'] = SUCCESS_NAIVE
    # With TVR Defense
    result_df['NAT_ACC_DEF'] = NAT_ACC_DEF
    result_df['ADVER_ACC_DEF'] = ADVER_ACC_DEF
    result_df['SUCCESS_DEF'] = SUCCESS_DEF

    print(result_df)
    # Save DataFrame as a CSV file
    result_df.to_csv(RES_SAVE_PATH)

