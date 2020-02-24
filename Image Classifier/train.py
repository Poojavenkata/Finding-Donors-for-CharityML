import argparse
import os
import random 
import time
import copy
from os import listdir
from collections import OrderedDict
from torchvision import transforms, datasets

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.models as models
from torchvision import models
import sys


parser=argparse.ArgumentParser()
parser.add_argument('-d','--data_dir',type=str, help= "Data directory of images")
parser.add_argument('-a','--arch',action='store',type=str, choices=['densenet121', 'vgg19', 'alexnet'], help="specify the classifier")
parser.add_argument('-hu','--hidden_units',nargs="+", type=int, default=[512, 256], help='no of hidden layers')
parser.add_argument('-l','--learning_rate',action='store',type=float, help='select the learning rate for the model')
parser.add_argument('-e','--epochs',action='store',type=int, help='for gradient descent select the number of epochs')
parser.add_argument('-s','--save_dir',action='store', type=str, help='Select name of file to save the trained model')
parser.add_argument('-g','--gpu',action='store_true',help='Use GPU if available for training')

args = parser.parse_args()

if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.data_dir:
    data_dir = args.data_dir
if args.gpu:        
    gpu = args.gpu
    

def create_model(arch='densenet121', output_size=102, hidden_units=[512,256]):
    model =  getattr(models,arch)(pretrained=True)
    
    input_features = {'vgg19':25088, 'alexnet':9216, 'densenet121':1024}
    input_size=input_features[arch]

    ''' Select a model from pretrained models and build a classifier for that model'''
    for param in model.parameters():
        param.requires_grad=False
    

    classifier=nn.Sequential(OrderedDict([
                            ('fc1',nn.Linear(input_size,hidden_units[0])),
                            ('relu',nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2',nn.Linear(hidden_units[0],hidden_units[1])),
                            ('relu',nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc3',nn.Linear(hidden_units[1],output_size)),
                            ('output',nn.LogSoftmax(dim=1)),
                            ]))
    model.classifier= classifier

    return model    

def train_model(image_datasets, arch='densenet121', hidden_units=[512,256], epochs=40, learning_rate=0.0001, gpu=False, checkpoint=''):
    
    if args.arch:
        arch = args.arch     
        
    if args.hidden_units:
        hidden_units = args.hidden_units

    if args.epochs:
        epochs = args.epochs
            
    if args.learning_rate:
        learning_rate = args.learning_rate

    if args.gpu:
        gpu = args.gpu
        
    dataset_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=0)
                  for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    Image_names = image_datasets['train'].classes
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    output_size = len(image_datasets['train'].classes)
    model = create_model(arch=arch, output_size=output_size, hidden_units=hidden_units)
    
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-5)
    model = model.to(device) 

    for i1,(inputs, labels) in enumerate(dataset_loader['train']):
        inputs, labels = inputs.to(device), labels.to(device)
        start=time.time()
        outputs=model.forward(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
    epochs = epochs
    print_every = 40
    steps = 0

    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataset_loader['train']):
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
            
                running_loss = 0
                
   
    model.class_to_idx = image_datasets['train'].class_to_idx
    save_dir = ''
    checkpoint = {
        'arch': arch,
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict(),
        'hidden_units': hidden_units
    }
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'

    torch.save(checkpoint, save_dir)
    return model



train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'valid', 'test']}
train_model(image_datasets)
    
