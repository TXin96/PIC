import os
import numpy as np
import model as modellib

from picConfig import PicConfig
from picConfig import PicDataset

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
LOG_SAVE_PATH = os.path.join(ROOT_DIR, "logs")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "model")

# Local path to trained weights file
COCO_MODEL_PATH = "model/mask_rcnn_coco.h5"

# Directory of the images
TRAIN_IMAGE_PATH = "image/train"
TRAIN_SEGMENTATION_PATH = "segmentation/train"

VAL_IMAGE_PATH = "image/val"
VAL_SEGMENTATION_PATH = "segmentation/val"

# Configuration
config = PicConfig()
config.display()

# Training dataset
dataset_train = PicDataset()
dataset_train.load_pic(TRAIN_IMAGE_PATH, TRAIN_SEGMENTATION_PATH)
dataset_train.prepare()

# Validation dataset
dataset_val = PicDataset()
dataset_val.load_pic(VAL_IMAGE_PATH, VAL_SEGMENTATION_PATH)
dataset_val.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 1)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    print(mask.shape)
    print(class_ids)
    # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=LOG_SAVE_PATH)

# Load weights
print("Loading weights ", COCO_MODEL_PATH)
model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
                   )

# Training
print("Start training ")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers='4+')

# Save weights manually
# Typically not needed because callbacks save after every epoch
model_path = os.path.join(MODEL_SAVE_PATH, "pic_new.h5")
model.keras_model.save_weights(model_path)
print("Weights saved at ", model_path)
