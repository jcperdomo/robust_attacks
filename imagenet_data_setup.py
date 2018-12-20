import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from random import shuffle
import numpy as np
from torch.utils.data import TensorDataset
import pandas as pd
from PIL import Image
import os
from torch_models import DNN


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


valdir = "ILSVRC2012_img_val/"
df = pd.read_csv('imagenet_validation_labels.csv')

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

images = []
labels = []
file_list = os.listdir(valdir)
random_ims = np.random.choice(file_list, 200, replace=False)

for img in random_ims:
    if img[-4:] == "JPEG":
        image = transform(pil_loader(valdir + img))
        label = df[df['name'] == img]['label'].values[0]
        images.append(image)
        labels.append(label)


images = torch.stack(images)
labels = torch.tensor(labels)

resnet18 = models.resnet18(pretrained=True).eval()
resnet50 = models.resnet50(pretrained=True).eval()
vgg13 = models.vgg13(pretrained=True).eval()
vgg19 = models.vgg19_bn(pretrained=True).eval()
densenet = models.densenet161(pretrained=True).eval()

model_list = [DNN(resnet18, cuda=False), DNN(resnet50, cuda=False), DNN(vgg13, cuda=False),
              DNN(vgg19, cuda=False), DNN(densenet, cuda=False)]
print("finished loading models")

correct_predictions = torch.ones(images.size()[0], dtype=torch.uint8)

for i, model in enumerate(model_list):
    print("Testing predictions for model ", i)
    out = model(images)
    _, preds = torch.max(out, 1)
    correct_predictions = correct_predictions & (preds == labels)

correct_images = images[correct_predictions]
correct_labels = labels[correct_predictions]

print("Verifying Predictions")
for i, model in enumerate(model_list):
    out = model(correct_images)
    _, preds = torch.max(out, 1)
    assert(torch.sum(preds == correct_labels) == correct_images.size()[0])
    print("Correct ", i)

torch.save(correct_images, 'imagenet_images.pt')
torch.save(correct_labels, 'imagenet_labels.pt')
