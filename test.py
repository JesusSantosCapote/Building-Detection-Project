import torch
from models import get_object_detection_model
from settings import NUM_CLASSES


"""# Testing our Model

Now lets take an image from the test set and try to predict on it
"""

# pick one image from the test set
img, target = val_dataset[7]

model.load_state_dict(torch.load(best_model_path))

# put the model in evaluation mode
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.eval()
with torch.no_grad():
  prediction = model([img.to(device)])[0]


print('MODEL OUTPUT\n')
nms_prediction = apply_nms(prediction, iou_thresh=0.01)
print(nms_prediction)

plot_img_bbox(torch_to_pil(img), nms_prediction, classes)