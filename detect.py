import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json

import utils
import model as modellib
import visualize
from picConfig import PicConfig
from picConfig import PicDataset

# Detection
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_PATH = os.path.join(MODEL_DIR, "pic.h5")

# Directory of the images
TRAIN_IMAGE_PATH = "image/train"
TRAIN_SEGMENTATION_PATH = "segmentation/train"

VAL_IMAGE_PATH = "image/val"
VAL_SEGMENTATION_PATH = "segmentation/val"

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


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class InferenceConfig(PicConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")

# Load trained weights
print("Loading weights from ", MODEL_PATH)
model.load_weights(MODEL_PATH, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)
info = dataset_val.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset_val.image_reference(image_id)))
# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]

visualize.display_instances(image, boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'],
                            class_names=dataset_val.class_names, scores=r['scores'], ax=ax,
                            figsize=[image.shape[0], image.shape[1]],
                            title="Predictions")

for i in r:
    r[i] = r[i].tolist()
r.pop('masks')
r.pop('scores')
r['image_name'] = info['id'] + '.jpg'
with open('result.json', 'w') as outfile:
    json.dump(r, outfile, ensure_ascii=False)


# Evaluation