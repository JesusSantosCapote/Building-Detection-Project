import torch
import os
import cv2
import json
from image_preparation import resize_image
import numpy as np

# we create a Dataset class which has a __getitem__ function and a __len__ function
class BuildingImageDataset(torch.utils.data.Dataset):

  def __init__(self, images_dir, annot_dir, width, height, transforms=None):
    self.transforms = transforms
    self.images_dir = images_dir
    self.annot_dir = annot_dir
    self.height = height
    self.width = width
    
    # sorting the images for consistency
    # To get images, the extension of the filename is checked to be jpg
    self.imgs = [image for image in sorted(os.listdir(self.images_dir)) if image[-4:]=='.jpg']
    
    # classes: 0 index is reserved for background
    self.classes = ['background', 'airport']

  def __getitem__(self, idx):
    img_name = self.imgs[idx]
    image_path = os.path.join(self.images_dir, img_name)

    # reading the images and converting them to correct size and color    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # annotation file
    annot_filename = img_name[:-4] + '.json'
    annot_file_path = os.path.join(self.annot_dir, annot_filename)
    
    image_info = []

    with open(annot_file_path) as f:

      json_dict = json.load(f)

      for elem in json_dict["bounding_boxes"]:
        label = self.classes.index(elem["category"])

        box = elem['box']

        xmin = max(0, int(float(box[0])))
        ymin = max(0, int(float(box[1])))
        xmax = min(int(json_dict["img_width"]), int(float(box[2])))
        ymax = min(int(json_dict["img_height"]), int(float(box[3])))

        image_info.append([xmin, ymin, xmax, ymax, label])

      f.close()

    image_info = np.array(image_info)

    transformed_dict = resize_image(img_rgb, image_info, 224, 224)

    # contains the image as array
    img_res = np.asarray(transformed_dict["image"])

    # diving by 255
    img_res /= 255.0

    boxes = []
    labels = []
    
    for elem in transformed_dict["bboxes"]:
      print(elem)
      boxes.append([elem[0], elem[1], elem[2], elem[3]])
      labels.append(elem[4])
    
    # convert boxes into a torch.Tensor
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    
    # getting the areas of the boxes
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # suppose all instances are not crowd
    iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
    
    labels = torch.as_tensor(labels, dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["area"] = area
    target["iscrowd"] = iscrowd
    image_id = torch.tensor([idx])
    target["image_id"] = image_id

    if self.transforms:
      sample = self.transforms(image = img_res,
                                bboxes = target['boxes'],
                                labels = labels)
      img_res = sample['image']
      target['boxes'] = torch.Tensor(sample['bboxes'])
        
    return img_res, target

  def __len__(self):
    return len(self.imgs)