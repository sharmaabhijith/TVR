import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64

# Providing path to working/project directory
import os
import sys
import math
import gzip
import pickle
import random
import argparse
import numpy as np
import pandas as pd

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

##-------------------------------------------------------------------------------------------------------------------------------------------------------##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#preprocess_1 = Compose([transforms.PILToTensor()])
preprocess_2 = Compose([Resize(256), CenterCrop(224), ToTensor()])
# Initializing Generator
PATH = "./GAN_model/netG.pt"
netG = Generator()
netG.load_state_dict(torch.load(PATH, map_location=device))
netG.eval()
# Patch class list
patch_class = ['soap_dispenser', 'cornet', 'plate', 'banana', 'cup', 'typewriter', 'electric_guitar', 'hair_spray', 'sock', 'cellular_telephone']
# dictionary with the ImageNet label names
with open(os.path.join(os.getcwd(), "./assets/imagenet1000_clsidx_to_labels.txt")) as f:
    target_to_classname = eval(f.read())
# Load the patches
with gzip.open(os.path.join(os.getcwd(), "./assets/imagenet_patch.gz"), 'rb') as f:
    imagenet_patch = pickle.load(f)
patches, targets, info = imagenet_patch
# Mean and Standard deviation of ImageNet
normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
##-------------------------------------------------------------------------------------------------------------------------------------------------------##

st.set_page_config(layout="wide", page_title="TVR Demo")

st.write("# Total Variation based Image Resurfacer (TVR)")
st.write("## Who am I?")
st.markdown(
    "*A model-agnostic defense against single and multi patch attacks based on total variation for image resurfacing (TVR).\
            The TVR is an image-cleansing method that processes images to remove probable adversarial regions.*"
)
st.sidebar.write("CHALLENGE ME IF YOU CAN! :gear:")

opt_model_name = st.sidebar.selectbox('PLEASE SELECT A CNN MODEL', ('Alexnet', 'ResNet18', 'SqueezeNet', 'VGG16', 'GoogleNet', 'Inception_v3'))
opt_adv_patch = st.sidebar.selectbox('## PLEASE SELECT AN ADVERSARIAL PATCH', ('soap_dispenser', 'cornet', 'plate', 'banana', 'cup', 'typewriter',
    'electric_guitar', 'hair_spray', 'sock', 'cellular_telephone'))
opt_patch_num = st.sidebar.selectbox('## SELECT THE NUMBER OF PATCHES',(1, 2, 3, 4, 5))
opt_block_size = st.sidebar.selectbox('## PLEASE SELECT A BLOCK-SIZE', (7, 14, 28, 56, 112))

# Download the cleansed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def convert_img_2_tensor(image):
    #img_tensor =preprocess_1(image)
    img_tensor = image
    img_tensor = preprocess_2(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def convert_tensor_2_img(img_tensor):
    #image = Image.fromarray(img_tensor[0].cpu().detach().numpy())
    #img.save("faces.png")
    T = transforms.ToPILImage()
    image = T(img_tensor)
    return image

# Defend an image using TVR
def cleanse_image(img_tensor, block_size, netG):
    img_tensor = img_tensor.to(device)
    TVR =  Total_Variation_Resurfacer(img_tensor, (block_size, block_size))
    TVR.Image_to_Block()
    TVR.calculate_TV_Score()
    TVR.outlier_Detection()
    TVR.obsfucated_Image()
    cleansed_img_tensor = TVR.reconstructed_Image(netG)
    return cleansed_img_tensor

# Predict a given image using CNN
def predict_image(img_tensor, model_name):
    if(device=='cpu'):
        gpu=False
    else:
        gpu=True
    cnn_model = select_Model(model_name, gpu)
    img_tensor = normalizer(img_tensor).to(device)
    pred = F.softmax(cnn_model(img_tensor).data, dim=1).data[0]
    top1 = torch.topk(pred, 1)[1].item()
    class_name = target_to_classname[top1]
    class_name = class_name.split(",")
    class_name = class_name[0]
    return class_name

# Performing patch attack
def patch_attack(img_tensor, adv_patch, patch_num):
    cls = patch_class.index(adv_patch)
    patch = patches[cls]
    target = targets[cls]
    patch_sizes = [0.99, 0.69, 0.59, 0.49, 0.44]
    adv_image = img_tensor
    for i in range(patch_num):
        apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                          translation_range=(.2, .2),
                          rotation_range=(-45, 45),
                          scale_range=(patch_sizes[patch_num-1], patch_sizes[patch_num-1]))
        # Define patch applier with and without normalization
        patch_normalizer = Compose([apply_patch, normalizer])
        patch_applier = Compose([apply_patch])
        adv_image = patch_applier(adv_image.cpu()).to(device)
    return adv_image

def fix_image(upload, block_size, netG):
    image = Image.open(upload)
    # Original clean image
    img_tensor = convert_img_2_tensor(image)
    clean_image = convert_tensor_2_img(img_tensor[0])    
    col1.write("### Original Image :camera:")
    col1.image(clean_image)
    class_name = predict_image(img_tensor, opt_model_name)
    with col1:
        st.write(opt_model_name,"predicts as: ", class_name)
    
    # Adversarially patched image
    adv_img_tensor = patch_attack(img_tensor, opt_adv_patch, opt_patch_num)
    adv_image = convert_tensor_2_img(adv_img_tensor[0])
    col2.write("### Patched Image :rotating_light:")
    col2.image(adv_image)
    class_name = predict_image(adv_img_tensor, opt_model_name)
    with col2:
        st.write(opt_model_name,"predicts as: ", class_name)

    # Cleaning of patched image
    cleansed_img_tensor = cleanse_image(adv_img_tensor, block_size, netG)
    cleansed_image = convert_tensor_2_img(cleansed_img_tensor[0])
    col3.write("### Cleansed Image :wrench:")
    col3.image(cleansed_image)
    class_name = predict_image(cleansed_img_tensor, opt_model_name)
    with col3:
        st.write(opt_model_name,"predicts as: ", class_name)

    
    st.sidebar.markdown("\n")
    st.sidebar.download_button("DOWNLOAD CLEANSED IMAGE", convert_image(cleansed_image), "cleansed_image.png", "image/png")

##-----------------------------------------------------------*****MAIN*****-------------------------------------------------------------------------------------##
col1, col2, col3 = st.columns(3)
my_upload = st.sidebar.file_uploader("## UPLOAD IMAGE TO BE ATTACKED", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload, block_size=opt_block_size, netG=netG)
#else:
#    fix_image("./Figures/Placeholder_view_vector.png")

