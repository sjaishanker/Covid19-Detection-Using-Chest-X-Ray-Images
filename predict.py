import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)
from torch.autograd import Variable

model = torch.load('./model.pth')

loader = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size = (224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean =[0.485,0.456,0.406],std=[0.229,0.224,0.225]),    
])
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

image = image_loader('archive/COVID-19 Radiography Database/test/covid/COVID-19 (1).png')

res_dict = {0:"Normal",1:"Viral",2:"Covid"}
output = model(image)
# print(output)
_, preds = torch.max(output, 1)
# print(preds)
print(res_dict[preds.tolist()[0]])