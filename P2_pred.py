import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import torchvision.models as models
import sys

colormap = [[0,255,255], [255,255,0], [255,0,255], [0,255,0], [0,0,255], [255,255,255], [0,0,0]]
cm = np.array(colormap).astype('uint8')
n_class = len(colormap)
colormaphash = np.zeros(256**3)

for i ,col in enumerate(colormap):
    colormaphash[(col[0]*256 + col[1])*256 + col[2]] = i

def masktolabel(img):
    data = np.array(img, dtype='int32')
    ind = (data[:,:,0]*256 + data[:,:,1])*256 + data[:,:,2]
    return colormaphash[ind].astype('int32')

class FCN(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the VGG dataset """
        self.images = None
        self.labels = None
        self.root = root
        self.transform = transform

        # read filenames
        self.outnames = [file.split('.')[0]+'.png' for file in os.listdir(root) if file.endswith('.jpg')]
        self.outnames.sort()
        self.imagenames = [os.path.join(root,file) for file in os.listdir(root) if file.endswith('.jpg')]
        self.imagenames.sort()
        self.masknames = [os.path.join(root,file) for file in os.listdir(root) if file.endswith('.png')]
        self.masknames.sort()
        
        self.len = len(self.imagenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.imagenames[index]
        image = Image.open(image_fn)
        mask_fn = self.masknames[index]
        mask = Image.open(mask_fn)
        label = masktolabel(mask)
        
        if self.transform is not None:
            image = self.transform['image'](image)
            label = self.transform['label'](label)

        return image, label.long(), self.outnames[index]

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

transformdata = {
    'image':transforms.ToTensor(),
    'label':transforms.ToTensor()
                }

testset = FCN(root = sys.argv[1], transform=transformdata)
testset_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class FCN8(nn.Module):
    def __init__(self, n_class=7):
        VGG_model = models.vgg16(pretrained = True)
        super(FCN8, self).__init__()
        #conv1-3
        self.conv3 = nn.Sequential(
            *list(VGG_model.features.children())[:17]
        )
        self.conv4 = nn.Sequential(
            *list(VGG_model.features.children())[17:24]
        )
        self.conv5 = nn.Sequential(
            *list(VGG_model.features.children())[24:]
        )
        self.scorepl3 = nn.Conv2d(256, n_class, 1)
        self.scorepl4 = nn.Conv2d(512, n_class, 1)
        self.scorepl5 = nn.Conv2d(512, n_class, 1)
        self.upsamplepl4 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=False)
        self.upsamplepl5 = nn.Upsample(scale_factor = 4, mode = 'bilinear', align_corners=False)
        self.upsample8 = nn.Upsample(scale_factor = 8, mode = 'bilinear', align_corners=False)
    def forward(self, x):
        x = self.conv3(x)
        pool3 = self.scorepl3(x)
        x = self.conv4(x)
        pool4 = self.scorepl4(x)
        x = self.conv5(x)
        x = self.upsample8(pool3 + self.upsamplepl4(pool4) + self.upsamplepl5(self.scorepl5(x)))
        return x

def load_checkpoint(checkpoint_path, model, optimizer = None):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if optimizer!=None:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def outputimg(model, outroot):
    model.eval()  # Important: set evaluation mode
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target, filenames in testset_loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(1)[1].cpu().numpy()
            predimg = cm[pred]
            for i in range(predimg.shape[0]):
                img = Image.fromarray(predimg[i,:,:,:])
                img.save(os.path.join(outroot, filenames[i]))
    print('write in {}'.format(outroot))

model = FCN8().to(device)
load_checkpoint('mymodel2.pth', model)
outputimg(model, sys.argv[2])