import re
import os
import gc
import time
import random
import datetime

import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage.io import imread
from imgaug import augmenters as iaa

from cytomata.process.mrcnn.config import Config
from cytomata.process.mrcnn.utils import Dataset, download_trained_weights
from cytomata.process.mrcnn import model as modellib
from cytomata.process.mrcnn import visualize
from cytomata.utils.io import list_img_files


seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)


class CytoDataset(Dataset):
    """Handles image loading and dataset information for the Cytomata dataset.
    """
    def load_data(self, data_dir):
        """Construct dataset information.
        Args:
            data_dir (str): Dataset directory containing folders for each sample.
        """
        # Add classes from data source
        self.add_class(source='cyto', class_id=1, class_name='cell')

        # Add images information
        for sample_id in next(os.walk(data_dir))[1]:
            samp = os.path.join(data_dir, sample_id)
            imgf = next(os.walk(samp))[2][0]
            imgp = os.path.join(samp, imgf)
            self.add_image(
                source='cyto',
                image_id=sample_id,
                path=imgp,
                data_dir=data_dir
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Args:
            image_id (str): Unique id assigned to each sample.
        Returns:
            masks (ndarray): A bool array of shape
                [height, width, instances] with one mask per instance.
            class_ids (ndarray): A 1D array of class IDs corresponding to each instance masks.
        """
        info = self.image_info[image_id]
        mask_dir = os.path.join(info['data_dir'], info['id'], 'masks')
        masks = []
        class_ids = []
        for imgf in next(os.walk(mask_dir))[2]:
            mask = imread(os.path.join(mask_dir, imgf)).astype(np.bool)
            masks.append(mask)
            # class_id specified by number between parentheses
            if re.search('\(([^)]+)', imgf):
                class_id = re.search('\(([^)]+)', imgf).group(1)
            else:
                class_id = 1
            class_ids.append(class_id)
        return np.stack(masks, axis=-1), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image (for debugging).
        Args:
            image_id (str): Unique id assigned to each sample.
        """
        info = self.image_info[image_id]
        if info["source"]:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class CytoTrainConfig(Config):
    """Configuration for training Mask-RCNN on the Cytomata dataset.
    """
    NAME = "cyto_train"

    # Background + Cell
    NUM_CLASSES = 2

    # Depends on GPU memory
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of training and validation steps per epoch
    NUM_TRAIN_IMGS = 600
    NUM_VAL_IMGS = 64
    STEPS_PER_EPOCH = NUM_TRAIN_IMGS // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, NUM_VAL_IMGS // IMAGES_PER_GPU)
    EPOCHS = 8

    # Don't exclude based on confidence.
    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture (resnet50 or resnet101)
    BACKBONE = "resnet101"
    OPTIMIZER = 'ADAM'
    LEARNING_RATE = 1e-5
    GRADIENT_CLIP_NORM = 5.0

    # Image Scaling
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = False

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    # Number of ROIs per image to feed to classifier/mask heads
    TRAIN_ROIS_PER_IMAGE = 200

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

    # # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    #
    # # Non-max suppression threshold to filter RPN proposals.
    # # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9
    #
    # # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    #
    # # Image mean (RGB)
    MEAN_PIXEL = np.array([0.0, 0.0, 0.0])


class CytoInferConfig(CytoTrainConfig):
    """Configuration for performing inference after training Mask-RCNN on the Cytomata dataset.
    """
    NAME = "cyto_infer"

    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # No cropping for inference
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MIN_SCALE = False

    # Scale small images up to minimum 512px
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = False

    # Detection Constraints
    DETECTION_MAX_INSTANCES = 512
    DETECTION_NMS_THRESHOLD =  0.2
    DETECTION_MIN_CONFIDENCE = 0.9

    # # Non-max suppression threshold to filter RPN proposals.
    # # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


def train(train_dir, val_dir, weights=None):
    """Train a mask rcnn model on the image dataset.
    """
    root_dir = os.path.abspath(os.path.join(train_dir, os.pardir))
    results_dir = os.path.join(root_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Training dataset
    dataset_train = CytoDataset()
    dataset_train.load_data(train_dir)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CytoDataset()
    dataset_val.load_data(val_dir)
    dataset_val.prepare()

    # Training Configuration
    config_train = CytoTrainConfig()
    config_train.display()

    # Create Models
    model_train = modellib.MaskRCNN(
        mode='training',
        config=config_train,
        model_dir=results_dir
    )
    model_infer = modellib.MaskRCNN(
        mode='inference',
        config=CytoInferConfig(),
        model_dir=results_dir
    )

    # Calculate mAP for each epoch
    mAP_callback = modellib.MeanAveragePrecisionCallback(
        train_model=model_train,
        inference_model=model_infer,
        dataset=dataset_val,
        calculate_at_every_X_epoch=1,
        verbose=1)

    # Select weights file to load
    if weights.lower() == "coco":
        weights_path = os.path.join(results_dir, 'mask_rcnn_coco.h5')
        # Download weights file
        if not os.path.exists(weights_path):
            download_trained_weights(weights_path)
    elif weights.lower() == "last":
        # Find last trained weights
        weights_path = model_train.find_last()
    elif weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model_train.get_imagenet_weights()
    else:
        weights_path = weights

    # Previously trained weights file
    # https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
    # Exclude the last layers because they require a matching number of classes
    if weights.lower() == "coco":
        model_train.load_weights(
            weights_path,
            by_name=True,
            exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"
        ])
    else:
        model_train.load_weights(weights_path, by_name=False)

    # Image augmentation
    augmentation = iaa.SomeOf((0, 3), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([
            iaa.Affine(rotate=(0, 90)),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.Affine(shear=(-8, 8))
        ]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
        iaa.AdditiveGaussianNoise(scale=(0.0, 0.01*255))
    ])

    # Training Schedule
    # lr = config_train.LEARNING_RATE
    # epochs = config_train.EPOCHS
    # for i, lr in enumerate(np.linspace(lr, lr/epochs, epochs)):
    #     if weights.lower() == "coco" and i < epochs//10:
    #         layers = 'heads'
    #     else:
    #         layers = 'all'
    model_train.train(dataset_train, dataset_val,
        learning_rate=1e-6,
        epochs=50,
        augmentation=augmentation,
        layers='all',
        custom_callbacks=[mAP_callback]
    )


def predict(dir, weights_path):
    """Run prediction on images in the given directory."""
    root_dir = os.path.abspath(os.path.join(dir, os.pardir))
    results_dir = os.path.join(root_dir, 'results')
    # Create directories
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    os.makedirs(os.path.join(results_dir, submit_dir))

    # Training Configuration
    config = CytoInferConfig()
    config.display()

    model_infer = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=results_dir
    )
    model_infer.load_weights(weights_path, by_name=False)

    # Load over images
    for imgf in list_img_files(dir):
        # Load image and run detection
        image = imread(os.path.join(dir, imgf))
        # Detect objects
        results = model_infer.detect([image], verbose=0)[0]
        rois = results['rois']
        masks = results['masks']
        class_ids = results['class_ids']
        scores = results['scores']
        for i in range(masks.shape[2]):
            masks[:,:,i] = ndimage.binary_dilation(masks[:,:,i])
        # Save image with masks
        visualize.display_instances(
            image, result['rois'], result['masks'], result['class_ids'],
            dataset.class_names, result['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, imgf + '.png'))


def validate(dir, weights_path):
    """Run validation on images in the given directory."""
    root_dir = os.path.abspath(os.path.join(dir, os.pardir))
    results_dir = os.path.join(root_dir, 'results')
    # Create directories
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    os.makedirs(os.path.join(results_dir, submit_dir))

    # Read dataset
    dataset = CytoDataset()
    dataset.load_data(dir)
    dataset.prepare()

    # Inference Configuration
    config = CytoInferConfig()
    config.display()

    model_infer = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=results_dir
    )
    model_infer.load_weights(weights_path, by_name=False)

    APs = []
    for image_id in np.random.choice(dataset.image_ids, 10):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset,
            config,
            image_id,
            use_mini_mask=False
        )
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model_infer.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(
            gt_bbox, gt_class_id, gt_mask,
            r["rois"], r["class_ids"], r["scores"], r['masks']
        )
        APs.append(AP)

    return np.mean(APs)
