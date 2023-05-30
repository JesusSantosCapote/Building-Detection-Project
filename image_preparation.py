import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms as torchtrans


def resize_image(img_arr, bboxes, h, w):
    """
    :param img_arr: original image as a numpy array
    :param bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    :param h: resized height dimension of image
    :param w: resized weight dimension of image
    :return: dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}
    """
    # create resize transform pipeline
    transform = albumentations.Compose(
        [albumentations.Resize(height=h, width=w, always_apply=True)],
        bbox_params=albumentations.BboxParams(format='pascal_voc'))

    transformed = transform(image=img_arr, bboxes=bboxes)

    return transformed


# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target, classes):
  # plot the image and bboxes
  # Bounding boxes are defined as follows: x-min y-min width height
  if "scores" not in target.keys():
    scores = [100 for i in range(len(target['boxes']))]
  else:
    scores = target['scores']
  
  fig, a = plt.subplots(1,1)
  fig.set_size_inches(5,5)
  try:
    a.imshow(img)
  except Exception:
    a.imshow(img.T)
  for box, label, score in zip(target['boxes'], target['labels'], scores):
    x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
    print(x, y, width, height)
    rect = patches.Rectangle(
      (x, y),
      width, height,
      linewidth = 2,
      edgecolor = 'r',
      facecolor = 'none',
    )
    a.text(x, y, f"{classes[label]}:{score}")
    # Draw the bounding box on top of the image
    a.add_patch(rect)
  plt.show()


def copy_raw_datasets(actual_folder, img_destiny, annot_destiny):

  for directory in os.listdir(actual_folder):
    source = os.path.join(actual_folder, directory)
    if directory.endswith(".json"):
      shutil.copy(source, annot_destiny)
    else:
      shutil.copy(source, img_destiny)


# Send train=True for training transforms and False for val/test transforms
def get_transform(train):
  if train:
    return albumentations.Compose(
      [
        albumentations.HorizontalFlip(0.5),
        # ToTensorV2 converts image to pytorch tensor without div by 255
        ToTensorV2(p=1.0) 
      ],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )
  else:
    return albumentations.Compose(
      [ToTensorV2(p=1.0)],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )


# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
  return torchtrans.ToPILImage()(img).convert('RGB')