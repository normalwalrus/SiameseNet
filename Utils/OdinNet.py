import numpy as np
import torch
import glob
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torchvision.models import vgg19
from torchvision.models import VGG19_Weights
from torchvision.models import resnet50
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from torch.nn import BCELoss
from torch.utils.data import DataLoader

import torchvision.models as models
import torch.nn as nn

from sklearn.model_selection import train_test_split

import pandas as pd

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  def forward(self, x):
    return x

class Custom_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = resnet50(pretrained=True)
        self.fc = Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, imageA, imageB):
        featuresA = None
        featuresB = None
        with torch.no_grad():
          featuresA = torch.squeeze(self.features(imageA), dim=(-1,-2))
          featuresB = torch.squeeze(self.features(imageB), dim=(-1,-2))

        combined_featuresA = torch.subtract(torch.square(featuresA), torch.square(featuresB))
        combined_featuresB = torch.square(torch.subtract(featuresA, featuresB))
        combined_featuresC = torch.multiply(featuresA, featuresB)
        combined_features = torch.concat((combined_featuresA, combined_featuresC), dim=-1)
        return self.classifier(combined_features)

    def get_classifier(self):
      return self.classifier