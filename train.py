import os
import numpy as np
import model as modellib
import visualize
import cv2

from picConfig import PicConfig
from picConfig import PicDataset

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = "model/mask_rcnn_coco.h5"

# Directory of the images
IMAGE_PATH = "api/demo_example/pic/images"
INSTANCE_PATH = "api/demo_example/pic/instance"
SEMANTIC_PATH = "api/demo_example/pic/semantic"

# Configuration
config = PicConfig()
config.display()

# Training dataset
dataset_train = PicDataset()
dataset_train.load_pic(IMAGE_PATH)
dataset_train.prepare()

# Validation dataset
dataset_val = PicDataset()
dataset_val.load_pic(IMAGE_PATH)
dataset_val.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 1)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

    # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Load weights
print("Loading weights ", COCO_MODEL_PATH)
model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
                   )

# Training
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers='5+')

# Save weights manually
# Typically not needed because callbacks save after every epoch
model_path = os.path.join(MODEL_DIR, "pic.h5")
model.keras_model.save_weights(model_path)
