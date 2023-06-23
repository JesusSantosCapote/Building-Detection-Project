import os

NUM_CLASSES = 2

path = os.getcwd()

TRAIN_PATH = os.path.join(path, "data", "train")

VAL_PATH = os.path.join(path, "data", "val")

RAW_TRAIN_PATH = os.path.join(path, "data", "train_raw_dataset")
RAW_VAL_PATH = os.path.join(path, "data", "val_raw_dataset")

TRAIN_ANNOT_PATH = os.path.join(TRAIN_PATH,"annotations")
TRAIN_IMAGES_PATH = os.path.join(TRAIN_PATH, "images")

VAL_ANNOT_PATH = os.path.join(VAL_PATH,"annotations")
VAL_IMAGES_PATH = os.path.join(VAL_PATH, "images")



BEST_MODEL_PATH = os.path.join(path, "checkpoint", "best.pt ")


