from utils import delete_bad_data, move_raw_datasets
from settings import *

move_raw_datasets(RAW_TRAIN_PATH, TRAIN_IMAGES_PATH, TRAIN_ANNOT_PATH)
move_raw_datasets(RAW_VAL_PATH, VAL_IMAGES_PATH, VAL_ANNOT_PATH)

print(delete_bad_data(TRAIN_IMAGES_PATH, TRAIN_ANNOT_PATH))
print(delete_bad_data(VAL_IMAGES_PATH, VAL_ANNOT_PATH))