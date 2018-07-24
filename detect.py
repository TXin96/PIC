import os
import json
import numpy as np
from PIL import Image
import cv2
import model as modellib
from picConfig import PicConfig

# Detection
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
LOG_SAVE_PATH = os.path.join(ROOT_DIR, "logs")
MODEL_PATH = os.path.join(ROOT_DIR, "model/pic.h5")

# Directory of the images
TRAIN_IMAGE_PATH = "image/train"
TRAIN_SEGMENTATION_PATH = "segmentation/train"

VAL_IMAGE_PATH = "image/val"
VAL_SEGMENTATION_PATH = "segmentation/val"

TEST_IMAGE_PATH = "test/image"

INSTANCE_SAVE_PATH = os.path.join(ROOT_DIR, "output/instance")
SEMANTIC_SAVE_PATH = os.path.join(ROOT_DIR, "output/semantic")


class InferenceConfig(PicConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def detect(image_dir, instance_save_path, semantic_save_path):
    if not os.path.exists(instance_save_path):
        os.makedirs(instance_save_path)

    if not os.path.exists(semantic_save_path):
        os.makedirs(semantic_save_path)
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=LOG_SAVE_PATH)

    # Load trained weights
    print("Loading weights from ", MODEL_PATH)
    model.load_weights(MODEL_PATH, by_name=True)

    res = []
    with open('result.json', 'w') as outfile:
        datafiles = os.listdir(image_dir)

        for file in datafiles:
            print('name: ', file)
            image = cv2.imread(os.path.join(image_dir, file))
            results = model.detect([image], verbose=1)
            r = results[0]
            masks = r['masks']
            class_id = r['class_ids']
            non_zero_area = np.sum(np.sum(masks, axis=0), axis=0)
            sorted_mask_index = np.argsort(non_zero_area)
            print('非0面积: ', non_zero_area)
            print('排序: ', sorted_mask_index)
            print('id: ', class_id)

            instance_num = r['rois'].shape[0]

            instance_image = np.zeros((masks.shape[0], masks.shape[1])).astype(np.uint8)
            semantic_image = np.zeros((masks.shape[0], masks.shape[1])).astype(np.uint8)

            for j in range(instance_num-1, -1, -1):
                print(j)
                j = sorted_mask_index[j]
                print(j)
                mask = masks[:, :, j]
                print(mask.shape)
                instance_image[:, :] = np.where(mask == 1, j + 1, instance_image[:, :])
                semantic_image[:, :] = np.where(mask == 1, class_id[j], semantic_image[:, :])

            instance_image = Image.fromarray(instance_image)
            instance_image.save(os.path.join(instance_save_path, file.replace('.jpg', '.png')))
            semantic_image = Image.fromarray(semantic_image)
            semantic_image.save(os.path.join(semantic_save_path, file.replace('.jpg', '.png')))

            for k in r:
                r[k] = r[k].tolist()
            r.pop('masks')
            r.pop('scores')
            r['image_name'] = file
            res.append(r)
        json.dump(res, outfile, ensure_ascii=False)


if __name__ == '__main__':
    detect(VAL_IMAGE_PATH, INSTANCE_SAVE_PATH, SEMANTIC_SAVE_PATH)
