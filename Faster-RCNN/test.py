import torch
from models import get_object_detection_model
from settings import NUM_CLASSES
from dataset import TestImageDataset
import os
from output_filters import apply_nms
from utils import plot_img_bbox, torch_to_pil


path = os.getcwd()
best_model_path = os.path.join(path, "checkpoint", "best2.pt ")
test_image_path = os.path.join(path, "data", "test")

classes = ['background', 'stadium']

test_dataset = TestImageDataset(test_image_path, 480, 480)

# pick one image from the test set
img = test_dataset[1]

model = get_object_detection_model(NUM_CLASSES)

checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))

try:
  model.load_state_dict(checkpoint['model_state_dict'])
except:
  model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))

# put the model in evaluation mode
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.eval()
with torch.no_grad():
  prediction = model([img.to(device)])[0]


print('MODEL OUTPUT\n')
nms_prediction = apply_nms(prediction, iou_thresh=0.01)
print(nms_prediction)

plot_img_bbox(torch_to_pil(img), nms_prediction, classes)