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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#preprocess_1 = Compose([transforms.PILToTensor()])
preprocess_2 = Compose([Resize(256), CenterCrop(224), ToTensor()])
PATH = "./GAN_model/netG.pt"
netG = Generator()
netG.load_state_dict(torch.load(PATH, map_location=device))
netG.eval()

##-------------------------------------------------------------------------------------------------------------------------------------------------------##

st.set_page_config(layout="wide", page_title="TVR Demo")

st.write("# Total Variation based Image Resurfacer (TVR)")
st.write("## Who am I?")
st.markdown(
    "*A model-agnostic defense against single and multi patch attacks based on total variation for image resurfacing (TVR). The TVR is an image-cleansing method that processes images to remove probable adversarial regions.*"
)
st.sidebar.write("CHALLENGE ME IF YOU CAN! :gear:")

opt_model_name = st.sidebar.selectbox('PLEASE SELECT A CNN MODEL', ('Alexnet', 'ResNet18', 'SqueezeNet', 'VGG16', 'GoogleNet', 'Inception_v3'))
#st.sidebar.write('You selected:', opt_model_name)

opt_adv_patch = st.sidebar.selectbox('## PLEASE SELECT AN ADVERSARIAL PATCH', ('soap_dispenser', 'cornet', 'plate', 'banana', 'cup', 'typewriter', 'electric_guitar', 'hair_spray', 'sock', 'cellular_telephone'))
#st.sidebar.write('You selected:', opt_adv_patch)

opt_block_size = st.sidebar.selectbox('## PLEASE SELECT A BLOCK-SIZE', (7, 14, 28, 56, 112))
#st.sidebar.write('You selected:', opt_block_size)

# Download the fixed image
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


def cleanse_image(img_tensor, block_size, netG):
    img_tensor = img_tensor.to(device)
    TVR =  Total_Variation_Resurfacer(img_tensor, (block_size, block_size))
    TVR.Image_to_Block()
    TVR.calculate_TV_Score()
    TVR.outlier_Detection()
    TVR.obsfucated_Image()
    cleansed_img_tensor = TVR.reconstructed_Image(netG)
    return cleansed_img_tensor

#def predict_image(img_tensor):

def fix_image(upload, block_size, netG):
    image = Image.open(upload)
    col1.write("### Adversarial Image :camera:")
    col1.image(image)

    img_tensor = convert_img_2_tensor(image)
    cleansed_img_tensor = cleanse_image(img_tensor, block_size, netG)
    cleansed_image = convert_tensor_2_img(cleansed_img_tensor[0])
    col2.write("### Cleansed Image :wrench:")
    col2.image(cleansed_image)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("DOWNLOAD CLEANSED IMAGE", convert_image(cleansed_image), "cleansed_image.png", "image/png")

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("## UPLOAD IMAGE TO BE ATTACKED", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload, block_size=opt_block_size, netG=netG)
#else:
#    fix_image("./Figures/Placeholder_view_vector.png")

