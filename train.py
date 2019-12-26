# Training a dataset that has been provided and calculate accuracy and validation loss , and subsequently make predictions
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

arch = {'vgg 16': 25088,
        'densenet121': 1024}


# Setting up parameters for entry in command line
parser = argparse.ArguementParser(
         description = 'parser for train.py')
parser.add_argument('data_dir', action = 'store', default = './flowers/')
parser.add_argument('--save_dir', action = 'store', default = './checkpoint.pth')
parser.add_argument('--arch', action = 'store', default = 'vgg16')
parser.add_argument('--learning_rate', action= 'store', type = float, default = 0.0001)
parser.add_argument('--hidden_units', action = 'store', dest = 'hidden_units', type = int, default = 512)
parser.add_argument('epochs', action = 'store', default = 8, type = int)
parser.add_argument('--dropout', action = 'store' , type = float, default = 0.5)
parser.add_argument('--gpu', action = 'store', default = 'gpu'
                    
# Getting arguments from commandline
args = parser.parse_args()
where = args.data_dir
path = args.save_dir
lr = args.learning_rate
struct = args.arch
hidden_units = args.hidden_units
epochs = args.epochs
power = args.gpu
dropout = args.dropout
                    
# checking GPU
if power =='gpu':
   device = 'cuda'
else:
    device = 'cpu'

# Loading & processing images from data directory                    
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
                    
                    
# Loading transforms
data_transforms = {

    'trainer': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomRotation(30),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
                                     
                                                    
                  
    'validater': transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),
    'tester' :transforms.Compose ([transforms.Resize(224), transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])}
 # Loading datasets with Image folder
image_datasets = {
     'trainer': datasets.ImageFolder(train_dir, transform = data_transforms['trainer']),
     'validater' : datasets.ImageFolder(valid_dir, transform = data_transforms['validater']),
     'tester' : datasets.ImageFolder(test_dir, transform = data_transforms['tester'])}

# Using image datasets and train transforms to define dataloaders
dataloaders = {

    'trainer' : torch.utils.data.DataLoader(image_datasets['trainer'] ,batch_size = 64, shuffle = True),
    'validater' : torch.utils.data.DataLoader(image_datasets['validater'] ,batch_size = 32 ,shuffle = True),
    'tester' : torch.utils.data.DataLoader(image_datasets['tester'] ,batch_size = 32 , shuffle = True)}


# Building and  training  network while freezing weights
model = models.vgg19(pretrained = True)
model
# freezing weights
for param in model.parameters():
    param.requires_grad = False
                    
                    
# Importing json
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
                    
# Adjusting last output for us to match with the number of categories we have(102). Using ReLu activation function at each hidden layer and apply 
# layer and apply softmax Loss function to calculate error
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('drop', nn.Dropout(p=0.5)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim = 1))]))

model.classifier = classifier
model

                    
# Training model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

epochs = 8
pass_count = 5
steps = 0
loss_show = []
cuda = torch.cuda.is_available
# checking GPU

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

    
running_loss = 0

start = time.time()
print('Training has started, please wait')

for epoch in range(epochs):
    
    training_mode = 'trainer'
    validation_mode = 'validater'
    
    for mode in [training_mode, validation_mode]:
        if mode == training_mode:
            model.train()
        else:
            model.eval()
            validation_loss = 0
            accuracy = 0
            
        pass_count = 0
        
        for data in dataloaders[mode]:
            pass_count += 1
            inputs,labels = data
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Moving forward
            output = model.forward(inputs)
            loss = criterion(output,labels)
            # Moving backward
            if mode == training_mode:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            
            inputs,labels = inputs.to(device),labels.to(device)
            model.to(device)
            
            with  torch.no_grad():
                output = model.forward(inputs)
                validation_loss = criterion(output,labels)
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type(torch.FloatTensor).mean()

            
        if mode == training_mode:
            print('Epoch: {}/{} '.format(epoch+1, epochs),
                     'Training Loss: {:.4f} '.format(running_loss/pass_count),
                     'Validation Loss:  {:.4f} ' .format(validation_loss))
        else:
            print('Accuracy: {:.4f}'.format(accuracy))
        
        running_loss
    
    
                  
time_elapsed = time.time() - start
print('\nTotal time: {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed % 60))
                    
# Validating test data
model.eval
accuracy = 0
cuda = torch.cuda.is_available()
# Checking GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
pass_count = 0
for data in dataloaders['tester']:
    pass_count += 1
    images, labels = data
    images,labels = images.to(device),labels.to(device)
    output = model.forward(images)
    ps = torch.exp(output).data 
    equality = (labels.data == ps.max(1)[1])
    accuracy += equality.type(torch.FloatTensor).mean()
    
print('Accuracy Test: {:.3f}' .format(accuracy/pass_count))
    


# Model to save  training data
model.class_to_idx = image_datasets['trainer'].class_to_idx
model.cpu()

checkpoint = {'input_size': 25088,
              'arch': 'vgg19',
              'output_size': 102,
              'learning_rate': 0.0001,
              'classifier':classifier,
              'batch_size': 64,
              'epochs':epochs,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint, 'checkpoint.pth')