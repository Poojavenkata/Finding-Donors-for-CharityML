import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models
import seaborn as sb
from PIL import Image
import json


def main():
    input = get_args()
    path_to_image = input.image_path
    checkpoint = input.checkpoint
    num = input.top_k
    cat_names = input.category_names
    gpu = input.gpu
    
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model = load(checkpoint)
    img = Image.open(path_to_image)
    image = process_image(path_to_image)
    probs, classes = predict(path_to_image, model, num)
    check(image, path_to_image, model, cat_to_name)
    

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("--checkpoint", type=str, help="checkpoint in which trained model is contained")
    parser.add_argument("--top_k", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    
    return parser.parse_args()

def create_model(arch='densenet121', output_size=102, hidden_units=[512,256]):
    model =  getattr(models,arch)(pretrained=True)
    input_features = {'vgg19':25088, 'alexnet':9216, 'densenet121':1024}
    input_size=input_features[arch]
    
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

  
def load(x):
    """
        Load the saved trained model inorder to use for prediction
    """
    checkpoint = torch.load(x)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    dicti = checkpoint['class_to_idx']
    num_labels = len(dicti)
    hidden_units = checkpoint['hidden_units']
    model = create_model(arch=arch, output_size=num_labels, hidden_units=hidden_units)
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    image_loader = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    final = image_loader(img)
    return final

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    img=process_image(image_path)
    img.unsqueeze_(0)
    img.requires_grad_(False)
    model=model.eval()
    model = model.float()
    out=model.forward(img)
    probs = torch.exp(out)
    large_values=probs.topk(topk)
    print(large_values)
    probabilities = large_values[0].data[0].tolist()
    indices = large_values[1].data[0].tolist()
    idx_to_class = {idx: classification for classification, idx in model.class_to_idx.items()}
    classes = []
    for i in indices:
        print("i","class",i,idx_to_class[i])
        classes.append(idx_to_class[i])
    return probabilities, classes

def check(image, image_path, model, cat_to_name):
    """
        Ouput a picture of the image and a graph representing its top 'k' class labels
    """
    probs, classes = predict(image_path, model)
    Names = {}
    for i in classes:
        Names[i]=cat_to_name[i]
    print(Names)
    flower_names=list(Names.values())
    flower_names=list(Names.values())
    plt.subplot(2,1,2)
    sb.barplot(x=probs, y=flower_names, color=sb.color_palette()[0]); 
    plt.show()
    ax.imshow(image)
    
    
if __name__ == "__main__":
    main()
    