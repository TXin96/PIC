import numpy as np
import cv2
import os

from config import Config
import utils

CATEGORY_NAME = [
    "background",
    "human",
    "floor",
    "bed",
    "window",
    "cabinet",
    "door",
    "table",
    "potting-plant",
    "curtain",
    "chair",
    "sofa",
    "shelf",
    "rug",
    "lamp",
    "fridge",
    "stairs",
    "pillow",
    "kitchen-island",
    "sculpture",
    "sink",
    "document",
    "painting/poster",
    "barrel",
    "basket",
    "poke",
    "stool",
    "clothes",
    "bottle",
    "plate",
    "cellphone",
    "toy",
    "cushion",
    "box",
    "display",
    "blanket",
    "pot",
    "nameplate",
    "banners/flag",
    "cup",
    "pen",
    "digital",
    "cooker",
    "umbrella",
    "decoration",
    "straw",
    "certificate",
    "food",
    "club",
    "towel",
    "pet/animals",
    "tool",
    "household-appliances",
    "pram",
    "car/bus/truck",
    "grass",
    "vegetation",
    "water",
    "ground",
    "road",
    "street-light",
    "railing/fence",
    "stand",
    "steps",
    "pillar",
    "awnings/tent",
    "building",
    "mountrain/hill",
    "stone",
    "bridge",
    "bicycle",
    "motorcycle",
    "airplane",
    "boat/ship",
    "balls",
    "swimming-equipment",
    "body-building-apparatus",
    "gun",
    "smoke",
    "rope",
    "amusement-facilities",
    "prop",
    "military-equipment",
    "bag",
    "instruments"
]


class PicConfig(Config):
    # Configuration name
    NAME = "pic"

    GPU_COUNT = 1

    IMAGES_PER_GPU = 2

    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1536

    # Number of classes
    NUM_CLASSES = 85


class PicDataset(utils.Dataset):
    instance_image_path = ""
    semantic_image_path = ""

    def load_pic(self, image_dir, segmentation_dir):
        """Load the PIC dataset.

        image_dir: The directory of the PIC dataset.
        """

        self.instance_image_path = os.path.join(segmentation_dir, "instance")
        self.semantic_image_path = os.path.join(segmentation_dir, "semantic")

        # Add classes
        for i in range(len(CATEGORY_NAME)):
            self.add_class("pic", i, CATEGORY_NAME[i])

        # Add images
        datafiles = os.listdir(image_dir)
        for file in datafiles:
            img = cv2.imread(os.path.join(image_dir, file))
            height = img.shape[0]
            width = img.shape[1]
            self.add_image(source="pic", image_id=file.replace('.jpg', ''),
                           path=os.path.join(image_dir, file),
                           width=width,
                           height=height)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID."""
        info = self.image_info[image_id]
        image = cv2.imread(info['path'])
        return image

    def image_reference(self, image_id):
        """Return the id of the image."""

        info = self.image_info[image_id]
        if info["source"] == "pic":
            return info["id"]
        else:
            super(self.__class__).image_reference(image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        if info["source"] != "pic":
            return super(self.__class__, self).load_mask(image_id)

        image_name = info['id'] + ".jpg"

        semantic_image = cv2.imread(os.path.join(self.semantic_image_path, image_name.replace('.jpg', '.png')),
                                    cv2.IMREAD_GRAYSCALE)
        instance_image = cv2.imread(os.path.join(self.instance_image_path, image_name.replace('.jpg', '.png')),
                                    cv2.IMREAD_GRAYSCALE)

        category = np.unique(semantic_image)
        category = list(category[category != 0])

        image_size = instance_image.shape
        instance_masks = []
        class_ids = []
        for categoryId in category:
            # cate_type = CATEGORY_NAME[categoryId]
            # print(str("category: ") + cate_type)

            # instance image which only has instances of this category
            this_cat_ins_image = instance_image.copy()
            this_cat_ins_image[semantic_image != categoryId] = 0

            # create mask
            category_mask = np.zeros((image_size[0], image_size[1]))

            instance_id = np.unique(this_cat_ins_image)
            instance_id = list(instance_id[instance_id != 0])
            # print(str("instance_id: ") + str(instance_id))

            for i, instance in enumerate(instance_id):
                #    print(str("instance: ") + str(instance))
                #    print(str("i: ") + str(i))
                category_mask[instance_image == instance] = (i + 1) / 255

            instance_masks.append(category_mask)
            class_ids.append(categoryId)

        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids
