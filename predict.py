# Imports
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, models, transforms
import time
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os, random
from matplotlib.ticker import FormatStrFormatter

# setting up parameters for entry in command line
parser = argparse.ArgumentParse(
         description = 'parser for predict.py'
parser.add_argument('input', default = './flowers/test/8/' , nargs = '?', action = 'store' , type = str)
parser.add_argument('--dir', action = 'store', dest = 'data_dir', default = './flowers/')
parser.add_argument('checkpoint', default = './checkpoint.pth', nargs = '?', action = 'store', type = str)
parser.add_argument('--top_k', default = 5, dest = 'top_k', action = 'store', type = int)
parser.add_argument('--category_names', dest = 'category_names', action = 'store', default = 'cat_to_name.json')
parser.add_argument('--gpu', default = 'gpu', action = 'store', dest = 'gpu')
    
args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
device = args.gpu
path = args.checkpoint

# Processing image before prediction
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #  Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img = img.resize((256,256))
    value = 0.5* (256 - 224)
    img = img.crop((value,value, 256-value, 256-value))
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406 ])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std
    return img.transpose(2, 0, 1)

# Function of predicting image    
def predict(image_path,model,topk=5):
    '''
    predict the class of an image using trained deep learning model
    '''
    # Implementing the code that can predict class from an image file
    # Moving to cuda
    cuda = torch.cuda.is_available()
    # Checking GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if cuda:
        model.cuda()
        print('Utility name:', torch.cuda.get_device_name(torch.cuda.device_count()-1))
        print('Total Number of GPU:', torch.cuda.device_count())
        
    else:
        model.cpu()
        print('We are now using CPU')
    # Turning off dropout
    model.eval()
    # Image variable Assignment
    image = process_image(image_path)
    # Transfering image to tensor
    image = torch.from_numpy(np.array([image])).float()
    # Making an image to become an input
    image = Variable(image)
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    probabilities = torch.exp(output).data
        
    probs = torch.topk(probabilities, topk) [0].tolist()[0]
    index = torch.topk(probabilities,topk) [1].tolist()[0]
    
    ind = []
    for i in range (len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])
        
    
    # transfering our index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])
    return probs, label