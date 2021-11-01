# -*- coding: utf-8 -*-
import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import torchvision.models as models
import sys

class VGG(Dataset):
    def __init__(self, root, transform=None):
        # Intialize the VGG dataset 
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        self.filenames = [file for file in os.listdir(root) if file.endswith('.png')]
        self.filenames.sort()
        self.filenames.sort(key=lambda file:len(file))
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        # Get a sample from the dataset 
        image_fn = self.filenames[index]
        image = Image.open(os.path.join(self.root, image_fn))
            
        if self.transform is not None:
            image = self.transform(image)

        return  self.filenames[index], image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

transformdata = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = VGG(root = sys.argv[1], transform=transformdata)
testset_loader = DataLoader(testset, batch_size=50, shuffle=False, num_workers=1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = models.vgg16_bn()
model.classifier._modules['6'] = nn.Linear(4096, 50)
model.to(device)
state = torch.load('mymodel.pth', map_location=device)
model.load_state_dict(state)

def pred(model, root):
    model.eval()  # Important: set evaluation mode
    filenames = []
    predictlabels = []
    outputcsv = [['image_id','label']]
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for file, data in testset_loader:
            file, data = file, data.to(device)
            filenames.extend(list(file))
            output = model(data)
            predict = output.max(1)[1]
            predict = predict.squeeze().cpu().numpy().tolist()
            predictlabels.extend(predict)
    for i in range(len(filenames)):
        outputcsv.append([filenames[i],predictlabels[i]])
    with open(root, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(outputcsv)
    print('write in {}'.format(root))

pred(model, root=sys.argv[2])
