import os
import argparse
import sys
import matplotlib
# matplotlib.use('GTK3Agg')  #I had to use GTKAgg for this to work, GTK threw errors
import matplotlib.pyplot as plt  #...  and now do whatever you need..
import random
import skimage.io
import tensorflow as tf
import keras
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import pickle
import numpy as np
from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--dldir', type=str, required=True, help='The directory to download the dataset.',
                    default='../')
parser.add_argument('--classification', type=int, required=True, choices=[2,3],
                        help='specifies whether to classify only to distractor, target or to low-distractor, high-distractor, target.', default=2)
args = parser.parse_args()

# Root directory of the project
ROOT_DIR = os.path.join(args.dldir, "Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import visualize
from mrcnn.visualize import display_images

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(args.dldir, "distraction")

MASK_DIR = os.path.join(IMAGE_DIR, "mask")
TRAIN_PATH = os.path.join(IMAGE_DIR, "train/")
VALID_PATH = os.path.join(IMAGE_DIR, "valid/")
TEST_PATH = os.path.join(IMAGE_DIR, "test/")

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# Import COCO config
import coco

class ShapesConfig(coco.CocoConfig):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides values specific
    to the dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU. Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + args.classification  # background +  distractor, target

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_SHAPE = (320 , 512 ,3)
    #IMAGE_RESIZE_MODE = ""
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = ( 8, 16, 64, 128, 256)  # anchor side in pixels

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 800

    # set number of epoch
    STEPS_PER_EPOCH = 83

    # set validation steps 
    VALIDATION_STEPS = 16

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class ShapesDataset(utils.Dataset):
    
    def load_shapes(self, mode):
        
        # Add classes
        if args.classification==2:
            self.add_class("shapes", 1, "target")
            self.add_class("shapes", 2, "distractor")

        elif  args.classification==3:
            self.add_class("shapes", 1, "target")
            self.add_class("shapes", 2, "low-distractor")
            self.add_class("shapes", 3, "high distractor")
        # Get train and test IDs
        self.train_ids = next(os.walk(TRAIN_PATH))[2]
        self.valid_ids = next(os.walk(VALID_PATH))[2]
        self.test_ids = next(os.walk(TEST_PATH))[2]

        if mode == "train":  
            for n, id_ in enumerate(self.train_ids):
                #path = TRAIN_PATH + id_
                self.add_image("shapes", image_id=id_, path=TRAIN_PATH)
              
        if mode == "val":   
            for n, id_ in enumerate(self.valid_ids):
                  #path = VALID_PATH + id_ 
                  self.add_image("shapes", image_id=id_, path=VALID_PATH)   
        
        if mode == "test":   
            for n, id_ in enumerate(self.test_ids):
                  #path = TEST_PATH + id_
                  self.add_image("shapes", image_id=id_, path=TEST_PATH)    


    def load_image(self, image_id):
        
        info = self.image_info[image_id]
        info = info.get("id")
       
        path = TRAIN_PATH + info
        
        if info in self.train_ids:
           path = TRAIN_PATH + info
        elif info in self.valid_ids:
           path = VALID_PATH + info
        else:
           path = TEST_PATH + info

        img = imread(path)[:,:,:3]
        img = resize(img, (320, 512), mode='constant', preserve_range=True)
       
        return img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        
        info = self.image_info[image_id]
        info = info.get("id")

        path = os.path.join(MASK_DIR , info.split('.')[0]+ '.pickle')
        with open(path, "rb") as f_in:
          mask = pickle.load(f_in)
          number_of_masks = mask.shape[-1]
          
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(number_of_masks-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            
        # Map class names to class IDs.
        class_ids = np.asarray([np.max(mask[:,:,i]) for i in range(number_of_masks)])
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    
if __name__ == '__main__':

    config = ShapesConfig()
    config.display()

    # Training dataset
    dataset_train = ShapesDataset()
    dataset_train.load_shapes("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShapesDataset()
    dataset_val.load_shapes("val")
    dataset_val.prepare()

    dataset_test = ShapesDataset()
    dataset_test.load_shapes("test")
    dataset_test.prepare()

    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id )
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    filepath = os.path.join(args.dldir , "unet_best_weight.hdf5")
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [ checkpoint]
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE, 
                epochs=20, 
                layers='heads',
                custom_callbacks=callbacks_list)

    
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights

    # Load trained weights (fill in path to trained weights here)
    assert filepath != "", "Provide path to trained weights"
    print("Loading weights from ", filepath)
    model.load_weights( filepath, by_name=True)

    # Test on a random image
    image_id = random.choice(dataset_test.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, inference_config, 
                            image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset_train.class_names, figsize=(8, 8))

    # Get activations of a few sample layers
    activations = model.run_graph([original_image], [
        ("input_image",        tf.identity(model.keras_model.get_layer("input_image").output)),
        ("res2c_out",          model.keras_model.get_layer("res2c_out").output),
        ("res3c_out",          model.keras_model.get_layer("res3c_out").output),
        ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
        ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
        ("roi",                model.keras_model.get_layer("ROI").output),
    ])

    # Backbone feature map
    display_images(np.transpose(activations["res2c_out"][0,:,:,:4], [2, 0, 1]), cols=4)

    activations["input_image"].shape
    model.keras_model.summary()

    filters , bias= model.keras_model.get_layer("res5a_branch2b").get_weights()

    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    n_filters =6
    ix=1
    fig = plt.figure(figsize=(20,15))
    for i in range(n_filters):
        # get the filters
        f = filters[:,:,:,i]
        for j in range(3):
            # subplot for 6 filters and 3 channels
            plt.subplot(n_filters,3,ix)
            plt.imshow(f[:,:,j] ,cmap='gray')
            ix+=1
    #plot the filters 
    plt.show()