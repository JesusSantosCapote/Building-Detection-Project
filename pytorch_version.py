# -*- coding: utf-8 -*-
"""pytorch_version.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lBbTvYDQnrQunovtQs5Kc7SRvk9JiCS2
"""

import os
import shutil
from distutils.dir_util import copy_tree
import random
import numpy as np

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# helper libraries
# from vision.engine import evaluate
# import vision.utils
# import vision.transforms as T
import json
from sklearn import preprocessing
import math
import sys

# for image augmentations


from dataset import BuildingImageDataset
from models import get_object_detection_model
from train import train
from image_preparation import get_transform
from output_filters import apply_nms
from image_preparation import plot_img_bbox, torch_to_pil

def collate_fn(batch):
    return tuple(zip(*batch))


path = os.getcwd()
train_path = os.path.join(path, "data", "train")
val_path = os.path.join(path, "data", "val")

raw_train_path = os.path.join(path, "data", "train_raw_dataset")
raw_val_path = os.path.join(path, "data", "val_raw_dataset")

train_annot_path = os.path.join(train_path,"annotations")
train_images_path = os.path.join(train_path, "images")

val_annot_path = os.path.join(val_path,"annotations")
val_images_path = os.path.join(val_path, "images")

# copy_raw_datasets(raw_train_path, train_images_path, train_annot_path)
# copy_raw_datasets(raw_val_path, val_images_path, val_annot_path)

classes = ['background', 'airport']
# use our dataset and defined transformations
train_dataset = BuildingImageDataset(train_images_path, train_annot_path, 480, 480, transforms=get_transform(train=False))
val_dataset = BuildingImageDataset(val_images_path, val_annot_path, 480, 480, transforms=get_transform(train=False))

# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=10,
  shuffle=True,
  num_workers=0,
  collate_fn=collate_fn,
)

valid_loader = torch.utils.data.DataLoader(
  val_dataset,
  batch_size=10,
  shuffle=False,
  num_workers=0,
  collate_fn=collate_fn,
)


num_classes = 2 # one class (class 0) is dedicated to the "background"

model = get_object_detection_model(num_classes)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(
  optimizer,
  step_size=3,
  gamma=0.1
)


train(model, optimizer, train_loader, valid_loader, lr_scheduler)


"""# Testing our Model

Now lets take an image from the test set and try to predict on it
"""

# pick one image from the test set
img, target = val_dataset[7]

# put the model in evaluation mode
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.eval()
with torch.no_grad():
  prediction = model([img.to(device)])[0]


print('MODEL OUTPUT\n')
nms_prediction = apply_nms(prediction, iou_thresh=0.01)
print(nms_prediction)

plot_img_bbox(torch_to_pil(img), nms_prediction, classes)