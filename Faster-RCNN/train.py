import os
import torch
from dataset import BuildingImageDataset
from models import get_object_detection_model
from train_methods import train
from utils import get_transform
from output_filters import apply_nms
from utils import plot_img_bbox, torch_to_pil

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

best_model_path = os.path.join(path, "checkpoint", "best.pt ")

# copy_raw_datasets(raw_train_path, train_images_path, train_annot_path)
# copy_raw_datasets(raw_val_path, val_images_path, val_annot_path)

classes = ['background', 'stadium']
# use our dataset and defined transformations
train_dataset = BuildingImageDataset(train_images_path, train_annot_path, 480, 480, transforms=get_transform(do_transform=False))
val_dataset = BuildingImageDataset(val_images_path, val_annot_path, 480, 480, transforms=get_transform(do_transform=False))

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
