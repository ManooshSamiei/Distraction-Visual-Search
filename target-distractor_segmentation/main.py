import os
import shutil
import argparse
import sys
import matplotlib
# matplotlib.use('GTK3Agg')  #I had to use GTKAgg for this to work, GTK threw errors
import matplotlib.pyplot as plt  #...  and now do whatever you need..
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import gc
import random
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras
from keras import backend as K
from keras.callbacks import Callback

import skimage.io
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.measure import find_contours
import pickle
import numpy as np
from confusion_matrix import plot_confusion_matrix_from_data, gt_pred_lists

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import pandas as pd
import imgaug

parser = argparse.ArgumentParser()
parser.add_argument('--dldir', type=str, required=True, help='The directory to download the dataset.',
                    default='../')
parser.add_argument('--classification', type=int, required=True, choices=[2,3],
                        help='specifies whether to classify only to distractor, target or to low-distractor, high-distractor, target.', default=2)
parser.add_argument('--outdir', type=str, required=True, help='The directory to store the prediction results.', default='../results/')
parser.add_argument('--category', type=str, required=True, choices=['bottle', 'bowl', 'car'],
                        help='The target object category.', default='bottle')
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

if os.path.exists(os.path.join(args.outdir, args.category)):
    shutil.rmtree(os.path.join(args.outdir, args.category))

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(os.path.join(args.outdir, args.category),  exist_ok=True) 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(args.outdir, args.category, 'checkpoints') #os.path.join(ROOT_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
MAIN_DIR = os.path.join(args.dldir, "distraction", args.category)

MASK_DIR = os.path.join(MAIN_DIR, "mask")
IMAGE_DIR = os.path.join(MAIN_DIR, "image/")
print('number of images in the dataset', len(next(os.walk(IMAGE_DIR))[2]))


sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# Import COCO config
import coco

class VisAttentionConfig(coco.CocoConfig):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides values specific
    to the dataset.
    """
    # Give the configuration a recognizable name
    NAME = "VisualAttention"

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU. Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + args.classification  # background +  distractor, target

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"
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
    STEPS_PER_EPOCH = round(len(next(os.walk(IMAGE_DIR))[2])*4/5)

    # set validation steps 
    VALIDATION_STEPS = round(len(next(os.walk(IMAGE_DIR))[2])*1/5)

class InferenceConfig(VisAttentionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class VisAttentionDataset(utils.Dataset):
    
############################################################
#  K-fold cross validation
############################################################
    def load_custom_K_fold(self, mode , fold):
        
        # Add classes
        if args.classification==2:
            self.add_class("VisualAttention", 1, "target")
            self.add_class("VisualAttention", 2, "distractor")

        elif  args.classification==3:
            self.add_class("VisualAttention", 1, "target")
            self.add_class("VisualAttention", 2, "low-distractor")
            self.add_class("VisualAttention", 3, "high distractor")

        assert mode in ["train", "val"]
        # Get train and test IDs
        self.train_ids = next(os.walk(IMAGE_DIR))[2]

        N_Folds = 5
        k_fold = KFold(n_splits = N_Folds, random_state = 42, shuffle = True) 
        #k_fold = StratifiedKFold(n_splits = N_Folds, random_state = 42, shuffle = True) 
         
        for i, (train, val) in enumerate(k_fold.split(self.train_ids)):
            if mode == "train" and fold == i:
                for index in train:
                    #print(self.train_ids[index])
                    self.add_image("VisualAttention", image_id=self.train_ids[index], path=IMAGE_DIR)

            elif mode == "val" and fold == i:
                for index in val:
                    self.add_image("VisualAttention", image_id=self.train_ids[index], path=IMAGE_DIR) 
  
    def load_image(self, image_id):
        
        info = self.image_info[image_id]
        info = info.get("id")
       
        path = IMAGE_DIR + info

        img = imread(path)[:,:,:3]
        img = resize(img, (320, 512), mode='constant', preserve_range=True)
       
        return img

    def image_reference(self, image_id):
        """Return the data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "VisualAttention":
            return info["VisualAttention"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for types of the given image ID.
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
        mask[mask > 1] = 1
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


def display_instance(result_save_path, image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.savefig(result_save_path)
        plt.clf()
    plt.close(fig)
        #plt.show()


def compute_ar(pred_boxes, gt_boxes, list_iou_thresholds):

  AR = []
  for iou_threshold in list_iou_thresholds:

      try:
          recall, _ = utils.compute_recall(pred_boxes, gt_boxes, iou=iou_threshold)

          AR.append(recall)

      except:
        AR.append(0.0)
        pass

  AUC = 2 * (metrics.auc(list_iou_thresholds, AR))
  return AUC

class MAP_MAR_F1Score_Callback(Callback):
    def __init__(self, train_model: modellib.MaskRCNN, inference_model: modellib.MaskRCNN, dataset: utils.Dataset,
                 calculate_at_every_X_epoch: int = 2, dataset_limit: int = None,
                 verbose: int = 1):
        """
        Callback which calculates the mAP, mAR, and f1-score on the defined test/validation dataset
        :param train_model: Mask RCNN model in training mode
        :param inference_model: Mask RCNN model in inference mode
        :param dataset: test/validation dataset, it will calculate the mAP on this set
        :param calculate_at_every_X_epoch: With this parameter we can define if we want to do the calculation at
        every epoch or every second, etc...
        :param dataset_limit: When we have a huge dataset calculation can take a lot of time, with this we can set a
        limit to the number of data points used
        :param verbose: set verbosity (1 = verbose, 0 = quiet)
        """

        super().__init__()

        if train_model.mode != "training":
            raise ValueError("Train model should be in training mode, instead it is in: {0}".format(train_model.mode))

        if inference_model.mode != "inference":
            raise ValueError(
                "Inference model should be in inference mode, instead it is in: {0}".format(train_model.mode))

        if inference_model.config.BATCH_SIZE != 1:
            raise ValueError("This callback only works with the batch size of 1, instead: {0} was defined".format(
                inference_model.config.BATCH_SIZE))

        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.calculate_at_every_X_epoch = calculate_at_every_X_epoch
        self.dataset_limit = len(self.dataset.image_ids)
        if dataset_limit is not None:
            self.dataset_limit = dataset_limit
        self.dataset_image_ids = self.dataset.image_ids.copy()

        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch > 0 and epoch % self.calculate_at_every_X_epoch == 0:
            self._verbose_print("Calculating mAP...")
            self._load_weights_for_model()

            APs, ARs = self._calculate_mean_average_precision_recall()
            mAP = np.mean(APs) * 100
            mAR = np.mean(ARs) * 100
            f1_score = 2 * ((mAP * mAR) / (mAP + mAR))

            if logs is not None:
                logs["val_mean_average_precision"] = mAP
                logs["val_mean_average_recall"] = mAR
                logs["val_f1_score"] = f1_score

            self._verbose_print("At epoch {0} valid mAP is: {1} , valid mAR is: {2}, valid f1-score is: {3}%".format(epoch, mAP, mAR, f1_score))

        super().on_epoch_end(epoch, logs)

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
            last_weights_path))
        self.inference_model.load_weights(last_weights_path,
                                          by_name=True)

    def _calculate_mean_average_precision_recall(self):
        APs = []
        ARs = []
        np.random.shuffle(self.dataset_image_ids)

        for image_id in self.dataset_image_ids[:self.dataset_limit]:
            
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.dataset, self.inference_model.config,
                                                                             image_id, use_mini_mask=False)
            results = self.inference_model.detect([image], verbose=0)
            r = results[0]
            # Compute mAP - VOC uses IoU 0.5
            AP, precisions, recalls, overlap = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                           r["class_ids"], r["scores"], r['masks'])
            AR, positive_ids = utils.compute_recall(r['rois'], gt_bbox, iou=0.5)
            APs.append(AP)
            ARs.append(AR)

        return np.array(APs), np.array(ARs)

if __name__ == '__main__':

    config = VisAttentionConfig()
    config.display()

    #list_iou_thresholds = np.arange(0.5, 1, 0.05)
    gt_tot = np.array([])
    pred_tot = np.array([])
    mAPs = []
    mARs = []
    mAPs_50 = []
    recall = []
    precision = []
  
    os.makedirs(os.path.join(args.outdir, args.category, 'groundtruth'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, args.category, 'prediction'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, args.category, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, args.category, 'checkpoints'), exist_ok=True)   
    log_tensorboard = os.path.join(args.outdir, args.category, 'tensorboard/')

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    # Create model in inference mode
    inference_config = InferenceConfig()
    model_inference = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

    # training cross-validation with 5 fold
    for i in range(5):

        epoch_count = 0
        print("Training fold", i)
        # Training dataset.
        dataset_train = VisAttentionDataset()
        dataset_train.load_custom_K_fold("train", i)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = VisAttentionDataset()
        dataset_val.load_custom_K_fold("val", i)
        dataset_val.prepare()

        augmentation = imgaug.augmenters.Sometimes(0.5, [
                        imgaug.augmenters.Fliplr(0.5),
                        imgaug.augmenters.Flipud(0.5)])

        # # Load and display random samples
        # image_ids = np.random.choice(dataset_train.image_ids, 4)
        # for image_id in image_ids:
        #     image = dataset_train.load_image(image_id)
        #     mask, class_ids = dataset_train.load_mask(image_id )
        #     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


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
        
        mAP_callback = MAP_MAR_F1Score_Callback(model, model_inference, dataset_val, 
                                                        calculate_at_every_X_epoch=1, dataset_limit=500, verbose=1)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_tensorboard, histogram_freq=0)
        
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        model_path = os.path.join(args.outdir, args.category, 'checkpoints', "unet_best_weight_fold_" + str(i) + ".hdf5")
        #checkpoint = keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # MAP_MAR_F1Score_Callback needs a batch size of 1 
        checkpoint = keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [ mAP_callback, checkpoint, tensorboard_callback]#

        epoch_count +=10
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE *2,
                    epochs= epoch_count,
                    layers='heads',
                    custom_callbacks=callbacks_list)
                    #augmentation=augmentation)

        epoch_count +=5
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs= epoch_count,
                    layers='4+',
                    custom_callbacks=callbacks_list,
                    augmentation=augmentation)

        epoch_count +=5
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs= epoch_count,
                    layers='all',
                    custom_callbacks=callbacks_list,
                    augmentation=augmentation)  

        # Get path to saved weights
        # Either set a specific path or find last trained weights

        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model_inference.load_weights( model_path, by_name=True)

        # Test on a test images
        ARs = []
        APs = []
        # print(dataset_val.image_ids)
        for image_id in dataset_val.image_ids:
            original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, inference_config, 
                                        image_id, use_mini_mask=False)

            log("original_image", original_image)
            log("image_meta", image_meta)
            log("gt_class_id", gt_class_id)
            log("gt_bbox", gt_bbox)
            log("gt_mask", gt_mask)

            results_save_dir = os.path.join(os.path.join(args.outdir, args.category, 'groundtruth'), str(image_id) + '_' + str(i) + '.png')
            display_instance(results_save_dir, original_image, gt_bbox, gt_mask, gt_class_id, 
                                        dataset_train.class_names, figsize=(8, 8))

            scaled_image = modellib.mold_image(original_image, config)
            sample = np.expand_dims(scaled_image, 0)
            results =model_inference.detect(sample, verbose=1)

            r = results[0]
            #print(r)
            indices = []

            for ind , id in enumerate(r['class_ids']):
                if id==1 and r['scores'][ind]>0.00:
                    indices.append(ind)
                if id==2 and r['scores'][ind]>0.0:
                    indices.append(ind)
                if id==3 and r['scores'][ind]>0.00:
                    indices.append(ind)

            results_save_dir = os.path.join(os.path.join(args.outdir, args.category, 'prediction'), str(image_id) + '_' + str(i)+ '.png')
            
            if len(indices)!=0:
                ind = np.asarray(indices)   
                if len(r['masks'][...,ind].shape)==4:
                    mask_input = np.squeeze(r['masks'][...,ind])
                else:
                    mask_input = r['masks'][...,ind]
                display_instance(results_save_dir, original_image, r['rois'][ind], mask_input , r['class_ids'][ind], 
                                            dataset_val.class_names, r['scores'][ind], figsize=(8, 8))

            gt, pred = gt_pred_lists(gt_class_id, gt_bbox, r["class_ids"], r["rois"])
            gt_tot = np.append(gt_tot, gt)
            #print(gt_tot)
            pred_tot = np.append(pred_tot, pred)
            #print(pred_tot)
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.5)
            #visualize.plot_precision_recall(AP, precisions, recalls)
            precision.append(precisions)
            recall.append(recalls)
            #print(AP)
            AR, positive_ids = utils.compute_recall(r['rois'], gt_bbox, iou=0.5)
            #AR = compute_ar(r['rois'], gt_bbox, list_iou_thresholds)
            ARs.append(AR)
            APs.append(AP)

        mAPs.append(np.mean(APs))
        print('mAP for the best cross-validated model of fold {}:{}'.format(i, np.mean(APs)))
        mARs.append(np.mean(ARs))
        print('mAR for the best cross-validated model of fold {}:{}'.format(i, np.mean(ARs)))
        f1_score = 2 * ((np.mean(APs) * np.mean(ARs)) / (np.mean(APs) + np.mean(ARs)))
        print('f1-score for the best cross-validated model of fold {}:{}'.format(i, f1_score))
        df = pd.DataFrame({'Fold Number': [i], 'mAP0.5': [np.mean(APs)], 'mAR0.5':[np.mean(ARs)], 'f1_score':[f1_score]})
        df.to_csv(os.path.join(args.outdir, args.category, 'prediction', 'scores.xlsx'),  mode='a', header=not os.path.exists(os.path.join(args.outdir, args.category, 'prediction', 'scores.xlsx')), index=False) 

        # K.clear_session()
        # del model
        # del model_inference
        # gc.collect()
    #ax.bar(x, y, width=0.5, color='b', align='center')
    #visualize.plot_precision_recall(mAP, precision, recall)
    mAR = np.mean(mARs)
    mAP = np.mean(mAPs)
    f1_score = 2 * ((mAP * mAR) / (mAP + mAR))
    print('final mAP', mAP)
    print('final mAR', mAR)
    print('final f1_score', f1_score)
    # print('gt_tot', gt_tot)
    # print('pred_tot' , pred_tot)
    result_dir = os.path.join(os.path.join(args.outdir, args.category, 'prediction'), 'confusion_matrix.png')
    tp,fp,fn=plot_confusion_matrix_from_data(result_dir, args.classification, gt_tot, pred_tot,fz=12, figsize=(8,8), lw=0.5)    
    df = pd.DataFrame({'final mAP': [mAP], 'final mAR':[mAR], 'final f1_score':[f1_score]})
    df.to_csv(os.path.join(args.outdir, args.category, 'prediction', 'scores.xlsx'),  mode='a', header=not os.path.exists(os.path.join(args.outdir, args.category, 'prediction', 'scores.xlsx')), index=False) 

    recall = [x for xs in recall for x in xs] 
    precision = [x for xs in precision for x in xs] 
    precision.sort(reverse=True)
    recall.sort()
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(mAP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recall, precision)
    plt.savefig(os.path.join(args.outdir, args.category, 'prediction', 'precision-recall-curve.png'))

    # # Get activations of a few sample layers
    # activations = model.run_graph([original_image], [
    #     ("input_image",        tf.identity(model.keras_model.get_layer("input_image").output)),
    #     ("res2c_out",          model.keras_model.get_layer("res2c_out").output),
    #     ("res3c_out",          model.keras_model.get_layer("res3c_out").output),
    #     ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
    #     ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
    #     ("roi",                model.keras_model.get_layer("ROI").output),
    # ])

    # Backbone feature map
    # display_images(np.transpose(activations["res2c_out"][0,:,:,:4], [2, 0, 1]), cols=4)

    # activations["input_image"].shape
    # model.keras_model.summary()

    # filters , bias= model.keras_model.get_layer("res5a_branch2b").get_weights()

    # f_min, f_max = filters.min(), filters.max()
    # filters = (filters - f_min) / (f_max - f_min)
    
    # n_filters =6
    # ix=1
    # fig = plt.figure(figsize=(20,15))
    # for i in range(n_filters):
    #     # get the filters
    #     f = filters[:,:,:,i]
    #     for j in range(3):
    #         # subplot for 6 filters and 3 channels
    #         plt.subplot(n_filters,3,ix)
    #         plt.imshow(f[:,:,j] ,cmap='gray')
    #         ix+=1
    # plot the filters 
    # plt.show()