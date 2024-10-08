import timeit
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms, datasets
import numpy as np
import random

from torchvision.models import resnet18, ResNet18_Weights
weight = ResNet18_Weights.DEFAULT
preprocess = weight.transforms()

#Function for reproducibilty. You can check out: https://pytorch.org/docs/stable/notes/randomness.html
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(100)

#TODO: Populate the dictionary with your hyperparameters for training
def get_config_dict(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need different configs for part 1 and 2.
    """
    if pretrain == 0:

        config = {
            "batch_size": 100,
            "lr": 1e-3,
            "num_epochs": 15,
            "weight_decay": 1e-3,   #set to 0 if you do not want L2 regularization
            "save_criteria": None,     #Str. Can be 'accuracy'/'loss'/'last'. (Only for part 2)
        }
    else:

        config = {
            "batch_size": 100,
            "lr": 1e-5,
            "num_epochs": 8,
            "weight_decay": 1e-3,
            "save_criteria": "accuracy",
        }
    return config
    

#TODO: Part 1 - Complete this with your CNN architecture. Make sure to complete the architecture requirements.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5) #Convolution layer
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #Convolutions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        #Linear Classifiers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


#TODO: Part 2 - Complete this with your Pretrained CNN architecture. 
class PretrainedNet(nn.Module):
    def __init__(self):
        super(PretrainedNet, self).__init__()
        self.model = resnet18(weights=weight)
        print("Model summary:",self.model)

    def forward(self, x):
        x = self.model(x)
        return x 


#Feel free to edit this with your custom train/validation splits, transformations and augmentations for CIFAR-10, if needed.
def load_dataset(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need to define different dataset splits/transformations/augmentations for part 2.

    returns:
    train_dataset, valid_dataset: Dataset for training your model
    test_transforms: Default is None. Edit if you would like transformations applied to the test set. 

    """
    if pretrain == 1:

        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=preprocess)
        train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])
        test_transforms = preprocess
    else:

         full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
         train_dataset, valid_dataset = random_split(full_dataset, [38000,12000])
         test_transforms = None
    
    return train_dataset, valid_dataset, test_transforms




