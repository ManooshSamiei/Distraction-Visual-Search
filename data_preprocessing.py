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
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
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
                         train_task_img_pair,
                         trajs,
                         im_h,
                         im_w,
                         bbox,
                         sigma,
                         dldir,
                         datadir,
                         truncate_num=-1,
                         need_label=True):
    fix_labels = []
    fixs = []
    stimuli = []
    heat_maps_list = []
    min_fix_x = 100000
    max_fix_x = -100000
    min_fix_y = 100000
    max_fix_y = -100000
    flat_test_task_img_pair = []
    '''not_augment = ['bottle', 'bowl' , 'cup', 'car', 'chair', 
                   'clock', 'fork', 'keyboard', 'knife', 
                   'laptop', 'microwave', 'mouse', 'oven',
                   'potted plant', 'sink', 'stop sign', 
                   'toilet', 'tv']'''
    not_augment = []
    if phase == 'train':
        test_task_img_pair = []
        for key, group in groupby(train_task_img_pair, lambda x: x.split('_')[0]):
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

    for task_img in train_task_img_pair:

        heatmap = np.zeros((im_h, im_w), np.float32)
        heatmap_unblurred = np.zeros((im_h, im_w), np.float32)

        x1 = bbox[task_img][0]
        y1 = bbox[task_img][1]
        w_image = bbox[task_img][2]
        h_image = bbox[task_img][3]

        # observer =0
        for traj in trajs:

            if (traj['task'] + '_' + traj['name']) == task_img:
                # observer +=1
                # first fixations are fixed at the screen center
                traj['X'][0], traj['Y'][0] = im_w / 2, im_h / 2
                # fixs = [(traj['X'][0], traj['Y'][0])]
                if truncate_num < 1:
                    traj_len = len(traj['X'])
                else:
                    traj_len = min(truncate_num, len(traj['X']))

                for i in range(1, traj_len):
                    if traj['X'][i] < 0 or traj['Y'][i] < 0 or traj['X'][i] > 1680 or traj['Y'][i] > 1050:
                        continue
                    fix = (
                        ((traj['X'][i] - min_fix_x) / max_fix_x) * (512),
                        ((traj['Y'][i] - min_fix_y) / max_fix_y) * (320))
                    # remove returning fixations (enforce inhibition of return)
                    if fix in fixs:
                        continue
                    '''else:
                        fixs.append(fix)'''
                    # masking the target
                    '''if (x1<=fix[0]<=x1+w_image and y1<=fix[1]<=y1+h_image):
                        continue
                    else:'''
                    heatmap += GaussianMask(im_w, im_h, sigma, (fix[0], fix[1]))
                    heatmap_unblurred[int(fix[1]), int(fix[0])] = 1

        # Normalization
        heatmap = heatmap / np.amax(heatmap)
        heatmap_np = heatmap * 255
        heatmap = heatmap_np.astype("uint8")

        heatmap_unblurred = heatmap_unblurred / np.amax(heatmap_unblurred)
        heatmap_unblurred_np = heatmap_unblurred * 255
        heatmap_unblurred = heatmap_unblurred_np.astype("uint8")

        source = dldir + '/images/' + str(task_img.split('_')[0]) + '/' + str(task_img.split('_')[1])
        heatmap_flip = cv2.flip(heatmap, 1)
        img = cv2.imread(source)
        img_resized = cv2.resize(img, (im_w, im_h), interpolation=cv2.INTER_AREA)
        # bbox = [top left x position, top left y position, width, height].
        img_resized_flip = cv2.flip(img_resized, 1)

        target_0 = cv2.imread(dldir + '/targets/' + task_img.split('_')[
            0] + '_' + '0' + '.png')  # img_resized[y1:y1+h_image , x1:x1+w_image, :]

        target_1 = cv2.imread(dldir + '/targets/' + task_img.split('_')[
            0] + '_' + '1' + '.png')  # img_resized[y1:y1+h_image , x1:x1+w_image, :]

        target_2 = cv2.imread(dldir + '/targets/' + task_img.split('_')[
            0] + '_' + '2' + '.png')  # img_resized[y1:y1+h_image , x1:x1+w_image, :]

        target_3 = cv2.imread(dldir + '/targets/' + task_img.split('_')[
            0] + '_' + '3' + '.png')  # img_resized[y1:y1+h_image , x1:x1+w_image, :]

        target_4 = cv2.imread(dldir + '/targets/' + task_img.split('_')[
            0] + '_' + '4' + '.png')  # img_resized[y1:y1+h_image , x1:x1+w_image, :]

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

        if phase == 'train':

            if task_img in flat_test_task_img_pair:
                f = False
                out_name = datadir + '/saliencymap/test/' + str(task_img)
                out_name_np = datadir + '/saliencymap/test/' + os.path.splitext(str(task_img))[0]+'.npy'
                
                with open(out_name_np, "wb") as file:
                    np.save(file, heatmap_np )

                destination = datadir + '/stimuli/test/' + str(task_img)
                target_path_0 = datadir + '/target_0/test/' + str(task_img)
                target_path_1 = datadir + '/target_1/test/' + str(task_img)
                target_path_2 = datadir + '/target_2/test/' + str(task_img)
                target_path_3 = datadir + '/target_3/test/' + str(task_img)
                target_path_4 = datadir + '/target_4/test/' + str(task_img)

                img_target_rect_path = datadir + '/stimuli/test_targ_bbox/' + str(task_img)
                cv2.imwrite(img_target_rect_path, img_target_frame)
                
                unblur = True
                out_name_unblur = datadir + '/saliencymap/test_unblur/' + str(task_img)
                out_name_unblur_npy = datadir + '/saliencymap/test_unblur/' + os.path.splitext(str(task_img))[0]+'.npy'

            else:
                f = True
                out_name = datadir + '/saliencymap/train/' + str(task_img)
                destination = datadir + '/stimuli/train/' + str(task_img)

                target_path_0 = datadir + '/target_0/train/' + str(task_img)
                target_path_1 = datadir + '/target_1/train/' + str(task_img)
                target_path_2 = datadir + '/target_2/train/' + str(task_img)
                target_path_3 = datadir + '/target_3/train/' + str(task_img)
                target_path_4 = datadir + '/target_4/train/' + str(task_img)

                sal_out_flip = datadir + '/saliencymap/train/' + str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])
                stim_out_flip = datadir + '/stimuli/train/' + str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])

                tar_out_flip_0 = datadir + '/target_0/train/' + str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])
                tar_out_flip_1 = datadir + '/target_1/train/' + str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])
                tar_out_flip_2 = datadir + '/target_2/train/' + str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])
                tar_out_flip_3 = datadir + '/target_3/train/' + str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])
                tar_out_flip_4 = datadir + '/target_4/train/' + str(task_img.split('.')[0]) + '_flip.' + str(
                    task_img.split('.')[1])
        else:
            f = False
            out_name = datadir + '/saliencymap/valid/' + str(task_img)
            destination = datadir + '/stimuli/valid/' + str(task_img)

            target_path_0 = datadir + '/target_0/valid/' + str(task_img)
            target_path_1 = datadir + '/target_1/valid/' + str(task_img)
            target_path_2 = datadir + '/target_2/valid/' + str(task_img)
            target_path_3 = datadir + '/target_3/valid/' + str(task_img)
            target_path_4 = datadir + '/target_4/valid/' + str(task_img)

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Create mask
        threshold = 10

        mask = np.where(heatmap <= threshold, 1, 0)
        mask = np.reshape(mask, (im_h, im_w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images

        marge = img_resized * mask + heatmap_color * (1 - mask)
        marge = marge.astype("uint8")
        alpha = 0.5

        marge = cv2.addWeighted(img_resized, 1 - alpha, marge, alpha, 0)

        cv2.imwrite(destination, img_resized)
        cv2.imwrite(out_name, heatmap)
        cv2.imwrite(target_path_0, target_0)
        cv2.imwrite(target_path_1, target_1)
        cv2.imwrite(target_path_2, target_2)
        cv2.imwrite(target_path_3, target_3)
        cv2.imwrite(target_path_4, target_4)

        if f and (str(task_img.split('_')[0]) not in not_augment):
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
                 datadir,
                 is_testing=False):
    im_w = 512
    im_h = 320
    max_traj_length = 6

    target_init_fixs = {}
    cat_names = list(np.unique([x['task'] for x in trajs_train]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    # training fixation data
    train_task_img_pair = np.unique(
        [traj['task'] + '_' + traj['name'] for traj in trajs_train])

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
        truncate_num=max_traj_length)

    # validation fixation data
    valid_task_img_pair = np.unique(
        [traj['task'] + '_' + traj['name'] for traj in trajs_valid])

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
        truncate_num=max_traj_length)

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dldir', type=str, required=True, help='The directory to download the dataset.')
    parser.add_argument('--sigma', type=int, required=True,
                        help='Gaussian standard deviation to blur the fixation maps.')
    parser.add_argument('--datadir', type=str, required=True,
                        help='The directory to save images along with their fixation maps.')

    args = parser.parse_args()

    sl_map = os.path.join(args.datadir, 'saliencymap')
    tr_sl_map = os.path.join(sl_map, 'train')
    v_sl_map = os.path.join(sl_map, 'valid')
    te_sl_map = os.path.join(sl_map, 'test')
    te_unbur_sl_map = os.path.join(sl_map, 'test_unblur')

    stimuli = os.path.join(args.datadir, 'stimuli')
    tr_stimuli = os.path.join(stimuli, 'train')
    v_stimuli = os.path.join(stimuli, 'valid')
    te_stimuli = os.path.join(stimuli, 'test')
    targ_t_stimuli = os.path.join(stimuli, 'test_targ_bbox')

    target_0 = os.path.join(args.datadir, 'target_0')
    tr_target_0 = os.path.join(target_0, 'train')
    v_target_0 = os.path.join(target_0, 'valid')
    te_target_0 = os.path.join(target_0, 'test')

    target_1 = os.path.join(args.datadir, 'target_1')
    tr_target_1 = os.path.join(target_1, 'train')
    v_target_1 = os.path.join(target_1, 'valid')
    te_target_1 = os.path.join(target_1, 'test')

    target_2 = os.path.join(args.datadir, 'target_2')
    tr_target_2 = os.path.join(target_2, 'train')
    v_target_2 = os.path.join(target_2, 'valid')
    te_target_2 = os.path.join(target_2, 'test')

    target_3 = os.path.join(args.datadir, 'target_3')
    tr_target_3 = os.path.join(target_3, 'train')
    v_target_3 = os.path.join(target_3, 'valid')
    te_target_3 = os.path.join(target_3, 'test')

    target_4 = os.path.join(args.datadir, 'target_4')
    tr_target_4 = os.path.join(target_4, 'train')
    v_target_4 = os.path.join(target_4, 'valid')
    te_target_4 = os.path.join(target_4, 'test')

    os.makedirs(args.dldir, exist_ok=True)
    os.makedirs(args.datadir, exist_ok=True)

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

    # Downloading COCOSearch Dataset

    #download_cocosearch(args.dldir)

    dataset_root = args.dldir

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()
    # print(bbox_annos)

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
                           args.sigma, args.dldir, args.datadir)

    train = next(os.walk(tr_stimuli))[2] 
    print(len(train))

    valid = next(os.walk(v_stimuli))[2] 
    print(len(valid))

    test = next(os.walk(te_stimuli))[2] 
    print(len(test))