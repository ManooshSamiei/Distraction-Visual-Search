import numpy as np
from os.path import join
from itertools import groupby
import argparse
import os
import json
import random
import cv2
from dataset_download import download_cocosearch


def GaussianMask(sizex, sizey, sigma=11, center=None, fix=1):
    """Blurs each fixation point by convolving it
        with a Gaussian kernel. This function is adopted from
        https://github.com/takyamamoto/Fixation-Densitymap repository.
    args:
        sizex (int): mask width
        sizey (int): mask height
        sigma (int): gaussian std
        center (tuple): gaussian mean
        fix (int or float): gaussian max
    returns:
        gaussian mask
    """

    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)

    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0]) == False and np.isnan(center[1]) == False:
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey, sizex))

    return fix * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def preprocess_fixations(phase,
                         task_img_pair,
                         trajs,
                         im_h,
                         im_w,
                         bbox,
                         sigma,
                         dldir,
                         datadir,
                         truncate_num=-1):
    """Processes fixation data and creates 
        fixation maps. Resizes all search and target images 
        and save them in the corresponding directories. 
        Splits data into train-validation-test sets. 
        Augments the training data.
        saves unblurred fixation maps for saliency metric computation.
        saves target bbox overlayed test images for the purpose of results visualization.
    Args:
        phase (str): train or valid set (test set is separated from train set)
        task_img_pair (list): a list of task-image pairs 
        trajs (list): a list of all trials 
        im_h (int): resize search images to this height
        im_w (int): resize search images to this width
        bbox  (dict): target object bbox for each task-image pair
        sigma (int): sigma for Gaussian blurring function
        dldir (str): directory of downloaded data
        datadir (str): directory to save the preprocessed train/val/test sets
        truncate_num (int): maximum number of fixations to be processed from each trial
    """
    fix_labels = []
    stimuli = []
    heat_maps_list = []
    min_fix_x = 100000
    max_fix_x = -100000
    min_fix_y = 100000
    max_fix_y = -100000
    flat_test_task_img_pair = []

    if phase == 'train':
        test_task_img_pair = []
        for key, group in groupby(task_img_pair, lambda x: x.split('_')[0]):
            key_and_group = {key: random.sample(list(group), 18)}
            test_task_img_pair.append(key_and_group[key])

        flat_test_task_img_pair = [item for sublist in test_task_img_pair for item in sublist]

    for traj in trajs:
        for i in range(len(traj['X'])):
            if traj['X'][i] < 0 or traj['Y'][i] < 0 or traj['X'][i] > 1680 or traj['Y'][i] > 1050:
                continue

            if traj['X'][i] < min_fix_x:
                min_fix_x = traj['X'][i]

            if traj['X'][i] > max_fix_x:
                max_fix_x = traj['X'][i]

            if traj['Y'][i] < min_fix_y:
                min_fix_y = traj['Y'][i]

            if traj['Y'][i] > max_fix_y:
                max_fix_y = traj['Y'][i]

    for task_img in task_img_pair:

        heatmap = np.zeros((im_h, im_w), np.float32)
        heatmap_unblurred = np.zeros((im_h, im_w), np.float32)

        x1 = bbox[task_img][0]
        y1 = bbox[task_img][1]
        w_image = bbox[task_img][2]
        h_image = bbox[task_img][3]

        for traj in trajs:

            if (traj['task'] + '_' + traj['name']) == task_img:

                # first fixations are fixed at the screen center
                traj['X'][0], traj['Y'][0] = im_w / 2, im_h / 2
                if truncate_num < 1:
                    traj_len = len(traj['X'])
                else:
                    traj_len = min(truncate_num, len(traj['X']))

                for i in range(1, traj_len):
                    # remove out of boundary fixations
                    if traj['X'][i] < 0 or traj['Y'][i] < 0 or traj['X'][i] > 1680 or traj['Y'][i] > 1050:
                        continue
                    fix = (
                        ((traj['X'][i] - min_fix_x) / max_fix_x) * (512),
                        ((traj['Y'][i] - min_fix_y) / max_fix_y) * (320))
                    
                    # masking the target, uncomment if you want to mask the target
                    '''
                    if (x1<=fix[0]<=x1+w_image and y1<=fix[1]<=y1+h_image):
                        continue
                    else:
                    '''
                    heatmap += GaussianMask(im_w, im_h, sigma, (fix[0], fix[1]))
                    heatmap_unblurred[int(fix[1]), int(fix[0])] = 1

        # Normalization
        heatmap = heatmap / np.amax(heatmap)
        heatmap_np = heatmap * 255
        heatmap = heatmap_np.astype("uint8")

        heatmap_unblurred = heatmap_unblurred / np.amax(heatmap_unblurred)
        heatmap_unblurred_np = heatmap_unblurred * 255
        heatmap_unblurred = heatmap_unblurred_np.astype("uint8")

        source = os.path.join(dldir , 'images' , str(task_img.split('_')[0]) , str(task_img.split('_')[1]))
        heatmap_flip = cv2.flip(heatmap, 1)
        img = cv2.imread(source)
        img_resized = cv2.resize(img, (im_w, im_h), interpolation=cv2.INTER_AREA)
        # bbox = [top left x position, top left y position, width, height].
        img_resized_flip = cv2.flip(img_resized, 1)

        target_0 = cv2.imread(os.path.join(dldir , 'targets' , (task_img.split('_')[
            0] + '_0.png')))  # img_resized[y1:y1+h_image , x1:x1+w_image, :]

        target_1 = cv2.imread(os.path.join(dldir , 'targets' , (task_img.split('_')[
            0] + '_1.png'))) 

        target_2 = cv2.imread(os.path.join(dldir , 'targets' , (task_img.split('_')[
            0] + '_2.png'))) 

        target_3 = cv2.imread(os.path.join(dldir , 'targets' , (task_img.split('_')[
            0] + '_3.png'))) 

        target_4 = cv2.imread(os.path.join(dldir , 'targets' ,(task_img.split('_')[
            0] + '_4.png'))) 

        target_0 = cv2.resize(target_0, (64, 64), interpolation=cv2.INTER_AREA)
        target_flip_0 = cv2.flip(target_0, 1)

        target_1 = cv2.resize(target_1, (64, 64), interpolation=cv2.INTER_AREA)
        target_flip_1 = cv2.flip(target_1, 1)

        target_2 = cv2.resize(target_2, (64, 64), interpolation=cv2.INTER_AREA)
        target_flip_2 = cv2.flip(target_2, 1)

        target_3 = cv2.resize(target_3, (64, 64), interpolation=cv2.INTER_AREA)
        target_flip_3 = cv2.flip(target_3, 1)

        target_4 = cv2.resize(target_4, (64, 64), interpolation=cv2.INTER_AREA)
        target_flip_4 = cv2.flip(target_4, 1)

        img_target_frame=cv2.rectangle(img_resized.copy(),(x1,y1),(x1+w_image,y1+h_image),(0,255,0),2)
        
        unblur = False
        flip_f = False

        if phase == 'train':

            if task_img in flat_test_task_img_pair:
 
                unblur = True

                out_name = os.path.join(datadir , 'saliencymap/test' , str(task_img))
                out_name_np = os.path.join(datadir , 'saliencymap/test' , (os.path.splitext(str(task_img))[0]+'.npy'))
                
                with open(out_name_np, "wb") as file:
                    np.save(file, heatmap_np )

                destination = os.path.join(datadir , 'stimuli/test' , str(task_img))
                target_path_0 = os.path.join(datadir , 'target_0/test' , str(task_img))
                target_path_1 = os.path.join(datadir , 'target_1/test' , str(task_img))
                target_path_2 = os.path.join(datadir , 'target_2/test' , str(task_img))
                target_path_3 = os.path.join(datadir , 'target_3/test' , str(task_img))
                target_path_4 = os.path.join(datadir , 'target_4/test' , str(task_img))

                img_target_rect_path = os.path.join(datadir , 'stimuli/test_targ_bbox' , str(task_img))
                cv2.imwrite(img_target_rect_path, img_target_frame)
                
                out_name_unblur = os.path.join(datadir , 'saliencymap/test_unblur' , str(task_img))
                out_name_unblur_npy = os.path.join(datadir , 'saliencymap/test_unblur' , (os.path.splitext(str(task_img))[0]+'.npy'))

            else:

                flip_f = True

                out_name = os.path.join(datadir , 'saliencymap/train' , str(task_img))
                destination = os.path.join(datadir , 'stimuli/train' , str(task_img))

                target_path_0 = os.path.join(datadir , 'target_0/train' , str(task_img))
                target_path_1 = os.path.join(datadir , 'target_1/train' , str(task_img))
                target_path_2 = os.path.join(datadir , 'target_2/train' , str(task_img))
                target_path_3 = os.path.join(datadir , 'target_3/train' , str(task_img))
                target_path_4 = os.path.join(datadir , 'target_4/train' , str(task_img))

                sal_out_flip = os.path.join(datadir , 'saliencymap/train' , (str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])))
                stim_out_flip = os.path.join(datadir , 'stimuli/train' , (str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])))

                tar_out_flip_0 = os.path.join(datadir , 'target_0/train' , (str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])))
                tar_out_flip_1 = os.path.join(datadir , 'target_1/train' , (str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])))
                tar_out_flip_2 = os.path.join(datadir , 'target_2/train' , (str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])))
                tar_out_flip_3 = os.path.join(datadir , 'target_3/train' , (str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])))
                tar_out_flip_4 = os.path.join(datadir , 'target_4/train' , (str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])))
        else:
            
            out_name = os.path.join(datadir , 'saliencymap/valid' , str(task_img))
            destination = os.path.join(datadir , 'stimuli/valid' , str(task_img))

            target_path_0 = os.path.join(datadir , 'target_0/valid' , str(task_img))
            target_path_1 = os.path.join(datadir , 'target_1/valid' , str(task_img))
            target_path_2 = os.path.join(datadir , 'target_2/valid' , str(task_img))
            target_path_3 = os.path.join(datadir , 'target_3/valid' , str(task_img))
            target_path_4 = os.path.join(datadir , 'target_4/valid' , str(task_img))

        #uncomment this part to save colorful heatmap of fixations overlayed on images
        '''#create groundtruth heatmaps
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Create mask
        threshold = 30 #10

        mask = np.where(heatmap <= threshold, 1, 0)
        mask = np.reshape(mask, (im_h, im_w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images

        marge = img_resized * mask + heatmap_color * (1 - mask)
        marge = marge.astype("uint8")
        alpha = 0.5

        marge = cv2.addWeighted(img_resized, 1 - alpha, marge, alpha, 0)'''

        cv2.imwrite(destination, img_resized)
        cv2.imwrite(out_name, heatmap)
        #cv2.imwrite(out_name, marge)
        cv2.imwrite(target_path_0, target_0)
        cv2.imwrite(target_path_1, target_1)
        cv2.imwrite(target_path_2, target_2)
        cv2.imwrite(target_path_3, target_3)
        cv2.imwrite(target_path_4, target_4)

        if flip_f:
            cv2.imwrite(stim_out_flip, img_resized_flip)
            cv2.imwrite(sal_out_flip, heatmap_flip)
            cv2.imwrite(tar_out_flip_0, target_flip_0)
            cv2.imwrite(tar_out_flip_1, target_flip_1)
            cv2.imwrite(tar_out_flip_2, target_flip_2)
            cv2.imwrite(tar_out_flip_3, target_flip_3)
            cv2.imwrite(tar_out_flip_4, target_flip_4)

        if unblur:
            cv2.imwrite(out_name_unblur, heatmap_unblurred)
            with open(out_name_unblur_npy, "wb") as file:
                    np.save(file, heatmap_unblurred_np)

    return 

def process_data(trajs_train,
                 trajs_valid,
                 target_annos,
                 sigma,
                 dldir,
                 datadir):
    """creates task-image pairs for training and validation sets
        then calls preprocess_fixations func to 
        create fixation maps and train-test-valid split.
    args:
        trajs_train (list): a list of all trials in the original dataset training split 
        trajs_valid (list): a list of all trials in the original dataset validation split 
        target_annos (dict):  contains target object bbox for each task-image pair
        sigma (int): sigma for Gaussian blurring function
        dldir (str): directory of downloaded data
        datadir (str): directory to save the preprocessed train/val/test sets
    """

    im_w = 512
    im_h = 320
    #max_traj_length = 6

    target_init_fixs = {}
    cat_names = list(np.unique([x['task'] for x in trajs_train]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    # training fixation data
    train_task_img_pair = np.unique(
        [traj['task'] + '_' + traj['name'] for traj in trajs_train])

    # uncomment this part to process train data for only a single category
    '''train_task_img_pair = []
    for traj in trajs_train:
      if traj['task'] =='tv':
        train_task_img_pair.append(traj['task'] + '_' + traj['name'])
    train_task_img_pair = np.unique(np.asarray(train_task_img_pair))'''

    preprocess_fixations(
        'train',
        train_task_img_pair,
        trajs_train,
        im_h,
        im_w,
        target_annos,
        sigma,
        dldir,
        datadir,
        truncate_num=-1)

    # validation fixation data
    valid_task_img_pair = np.unique(
        [traj['task'] + '_' + traj['name'] for traj in trajs_valid])

    # uncomment this part to process valid data for only a single category
    '''valid_task_img_pair = []
    for traj in trajs_valid:
      if traj['task'] =='tv':
        valid_task_img_pair.append(traj['task'] + '_' + traj['name'])
    valid_task_img_pair = np.unique(np.array(valid_task_img_pair))'''

    preprocess_fixations(
        'valid',
        valid_task_img_pair,
        trajs_valid,
        im_h,
        im_w,
        target_annos,
        sigma,
        dldir,
        datadir,
        truncate_num=-1)

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dldir', type=str, required=True, help='The directory to download the dataset.' , default='../')
    parser.add_argument('--sigma', type=int, required=True,
                        help='Gaussian standard deviation to blur the fixation maps.', default=11)

    args = parser.parse_args()

    #The directory to save images along with their fixation maps.
    datadir = os.path.join(args.dldir , 'cocosearch/')

    sl_map = os.path.join(datadir, 'saliencymap')
    tr_sl_map = os.path.join(sl_map, 'train')
    v_sl_map = os.path.join(sl_map, 'valid')
    te_sl_map = os.path.join(sl_map, 'test')
    te_unbur_sl_map = os.path.join(sl_map, 'test_unblur')

    stimuli = os.path.join(datadir, 'stimuli')
    tr_stimuli = os.path.join(stimuli, 'train')
    v_stimuli = os.path.join(stimuli, 'valid')
    te_stimuli = os.path.join(stimuli, 'test')
    targ_t_stimuli = os.path.join(stimuli, 'test_targ_bbox')

    target_0 = os.path.join(datadir, 'target_0')
    tr_target_0 = os.path.join(target_0, 'train')
    v_target_0 = os.path.join(target_0, 'valid')
    te_target_0 = os.path.join(target_0, 'test')

    target_1 = os.path.join(datadir, 'target_1')
    tr_target_1 = os.path.join(target_1, 'train')
    v_target_1 = os.path.join(target_1, 'valid')
    te_target_1 = os.path.join(target_1, 'test')

    target_2 = os.path.join(datadir, 'target_2')
    tr_target_2 = os.path.join(target_2, 'train')
    v_target_2 = os.path.join(target_2, 'valid')
    te_target_2 = os.path.join(target_2, 'test')

    target_3 = os.path.join(datadir, 'target_3')
    tr_target_3 = os.path.join(target_3, 'train')
    v_target_3 = os.path.join(target_3, 'valid')
    te_target_3 = os.path.join(target_3, 'test')

    target_4 = os.path.join(datadir, 'target_4')
    tr_target_4 = os.path.join(target_4, 'train')
    v_target_4 = os.path.join(target_4, 'valid')
    te_target_4 = os.path.join(target_4, 'test')

    os.makedirs(args.dldir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    os.makedirs(sl_map, exist_ok=True)
    os.makedirs(tr_sl_map, exist_ok=True)
    os.makedirs(v_sl_map, exist_ok=True)
    os.makedirs(te_sl_map, exist_ok=True)
    os.makedirs(te_unbur_sl_map, exist_ok=True)

    os.makedirs(stimuli, exist_ok=True)
    os.makedirs(tr_stimuli, exist_ok=True)
    os.makedirs(v_stimuli, exist_ok=True)
    os.makedirs(te_stimuli, exist_ok=True)
    os.makedirs( targ_t_stimuli, exist_ok=True)

    os.makedirs(target_0, exist_ok=True)
    os.makedirs(tr_target_0, exist_ok=True)
    os.makedirs(v_target_0, exist_ok=True)
    os.makedirs(te_target_0, exist_ok=True)

    os.makedirs(target_1, exist_ok=True)
    os.makedirs(tr_target_1, exist_ok=True)
    os.makedirs(v_target_1, exist_ok=True)
    os.makedirs(te_target_1, exist_ok=True)

    os.makedirs(target_2, exist_ok=True)
    os.makedirs(tr_target_2, exist_ok=True)
    os.makedirs(v_target_2, exist_ok=True)
    os.makedirs(te_target_2, exist_ok=True)

    os.makedirs(target_3, exist_ok=True)
    os.makedirs(tr_target_3, exist_ok=True)
    os.makedirs(v_target_3, exist_ok=True)
    os.makedirs(te_target_3, exist_ok=True)

    os.makedirs(target_4, exist_ok=True)
    os.makedirs(tr_target_4, exist_ok=True)
    os.makedirs(v_target_4, exist_ok=True)
    os.makedirs(te_target_4, exist_ok=True)

    dataset_root = args.dldir

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()

    # load ground-truth human scanpaths

    with open(join(dataset_root,
                   'coco_search18_fixations_TP_train_split1.json')) as json_file:
        human_scanpaths_train = json.load(json_file)

    with open(join(dataset_root,
                   'coco_search18_fixations_TP_validation_split1.json')) as json_file:
        human_scanpaths_valid = json.load(json_file)

    # exclude incorrect scanpaths
    human_scanpaths_train = list(
        filter(lambda x: x['correct'] == 1, human_scanpaths_train))
    human_scanpaths_valid = list(
        filter(lambda x: x['correct'] == 1, human_scanpaths_valid))

    # process fixation data
    process_data(human_scanpaths_train, human_scanpaths_valid, bbox_annos,
                           args.sigma, dataset_root, datadir)

    train = next(os.walk(tr_stimuli))[2] 
    print(len(train))

    valid = next(os.walk(v_stimuli))[2] 
    print(len(valid))

    test = next(os.walk(te_stimuli))[2] 
    print(len(test))