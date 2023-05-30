import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import os

from torch.utils.tensorboard import SummaryWriter

"""Let the training begin!"""
path = os.getcwd()
checkpoint_path = os.path.join(path, "checkpoint", "checkpoint.pt ")

writer = SummaryWriter()
metric = MeanAveragePrecision()

def train(model, optimizer, train_loader, valid_loader, lr_scheduler):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    num_epochs = 1
    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            writer.add_scalar('Training Loss', losses.item(), epoch * len(train_loader) + batch_idx)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(valid_loader):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predict = model(images)
                metric.update(predict, targets)
                pprint(metric.compute())
                print(metric.compute()['map'].item())
                writer.add_scalar('Validation mAP', metric.compute()['map'].item(), epoch * len(train_loader) + batch_idx)


        # Save a checkpoint of the model and optimizer
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        # Print the loss and accuracy for the epoch
        print(f'Epoch {epoch}/{num_epochs}, Loss: {losses:.4f}')