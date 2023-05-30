import torchvision

# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'].cpu(), orig_prediction['scores'].cpu(), iou_thresh)
  
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'].cpu()[keep]
  final_prediction['scores'] = final_prediction['scores'].cpu()[keep]
  final_prediction['labels'] = final_prediction['labels'].cpu()[keep]
  
  return final_prediction