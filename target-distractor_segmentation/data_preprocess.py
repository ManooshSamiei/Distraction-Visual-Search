import numpy as np
import argparse
import alphashape
import os
import json
import random
import cv2
from shapely.geometry import Polygon, Point
import pickle

np.random.seed(42619)

adjusted_width = 512
adjusted_height = 320


def crop_seg(img, pts):
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    copy_img = img.copy()
    cv2.drawContours(mask, [pts.astype(int)], 0, (255, 255, 255), -1)
    dst = cv2.bitwise_and(copy_img, copy_img, mask=mask)

    return dst


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def preprocess(data_dir, category, task_img_pair, human_scanpaths, data_list_ann, phase):
    # We set maximum number of distractors to 20 and compute a distraction score for each distractor
    distraction_scores = np.zeros((len(task_img_pair), 20))

    # Calculate the size and adjusted display size

    show_polys = True
    draw_bbox = False

    fixs = []
    distraction = []
    min_fix_x = 100000
    max_fix_x = -100000
    min_fix_y = 100000
    max_fix_y = -100000

    # dictionary containing the category id of target object category
    cat_id = {'bottle': 1, 'bowl': 4, 'car': 5}
    data = {'images': []}

    rem_list = ['bottle_000000026454.jpg', 'bottle_000000024823.jpg', 'bottle_000000044569.jpg',
                'bottle_000000062348.jpg', 'bottle_000000053491.jpg', 'bottle_000000092288.jpg',
                'bottle_000000134769.jpg', 'bottle_000000204981.jpg', 'bottle_000000283479.jpg',
                'bottle_000000325992.jpg', 'bottle_000000341437.jpg', 'bottle_000000478155.jpg',
                'bottle_000000481185.jpg', 'bottle_000000517068.jpg', 'bottle_000000536127.jpg',
                'bottle_000000541147.jpg', 'bottle_000000553554.jpg', 'bottle_000000580685.jpg',
                'bottle_000000581205.jpg', 'bottle_000000263924.jpg', 'bottle_000000060360.jpg']

    for traj in human_scanpaths:
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

    for ind, search_image in enumerate(task_img_pair):

        print('file name:', search_image.split('_')[1])

        if search_image in rem_list:
            continue

        image_path = os.path.join(data_dir, 'images', category, search_image.split('_')[1])

        d = data_list_ann[np.where(data_list_ann[:, 0] == search_image)]
        target = d[np.where(d[:, 4] == cat_id[category])]

        if len(target) == 0:
            print('target is not found!')
            continue

        # remove the target from the list to just include the distractors
        distactors = list(np.delete(d, np.where(d[:, 4] == cat_id[category]), 0))

        print('Number of segmented distractors:', len(distactors))
        img = cv2.resize(cv2.imread(image_path), (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
        # cv2_imshow(img)
        image_width, image_height = int(target[0, 2]), int(target[0, 3])

        poly_t_l = []
        poly_targ = []
        poly_targ_list = []

        # add masked target
        img_new = np.zeros(np.shape(img), dtype="uint8")

        for seg in target[0, 1]:
            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
            img_new = cv2.addWeighted(img_new, 1, crop_seg(img, poly), 1, 1)

            poly_t_l.append([[[poly[i, 0], poly[i, 1]]] for i in range(0, len(poly[:, 0]))])
            poly_targ.append(poly)
            poly_targ_list.append(poly.tolist())

        data['images'].append({
            'file_name': search_image,
            'segmentation': poly_targ_list,
            'label': 11})  # 11 refers to target object

        # cv2_imshow(img_new)

        poly_distractors = []
        count = 0

        for distr in distactors:

            poly_list_dist = []
            distractor_pass = False

            img_f = img_new

            for seg in distr[1]:

                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                polyg_d = Polygon([(poly[i, 0], poly[i, 1]) for i in range(0, len(poly[:, 0]))])
                alpha_shape = alphashape.alphashape(poly, 0.)

                '''for poly_tar in poly_targ:
    
                  poly_target= Polygon([(poly_tar[i,0], poly_tar[i,1]) for i in range(0, len(poly_tar[:,0]))])
    
                  if poly_target.is_valid:
    
                    intersection = alpha_shape.intersection(poly_target)
                    percent_area = intersection.area / alpha_shape.area * 100
                    #print(percent_area)
                    if (poly_target.contains(polyg_d) or percent_area > 75.0):            
                      distractor_pass=True
    
                    if len(poly_distractors)!=0:
                      for l in poly_distractors:
                        for p in l:
                          p_d= Polygon([(p[i,0], p[i,1]) for i in range(0, len(p[:,0]))])
                          if p_d.is_valid:
                            intersection = alpha_shape.intersection(p_d)
                            percent_area = intersection.area / alpha_shape.area * 100
    
                            if (p_d.contains(polyg_d) or percent_area > 75.0): 
                              distractor_pass=True
    
                print('\n')
    
                if (PolyArea(poly[:,0],poly[:,1])> 100000):
                  distractor_pass=True'''

                if not (distractor_pass):
                    img_f = cv2.addWeighted(img_f, 1, crop_seg(img, poly), 1, 1)
                    poly_list_dist.append(poly.tolist())

            if not distractor_pass:

                count += 1
                if count > 20:
                    continue

                cv2.imwrite(os.path.join(data_dir, 'parsed_distractor', str(phase),
                                         search_image.split('.')[0] + '_' + str(count) + '.jpg'), img_f)
                # cv2_imshow(img_f)
                poly_distractors.append(poly_list_dist)

        observer = np.zeros((len(poly_distractors)))
        observer_new = np.zeros((len(poly_distractors)))

        for traj in human_scanpaths:

            if (traj['name'] == search_image.split('_')[1]) and (traj['task'] == category):

                observer[:] += 1
                # first fixations are fixed at the screen center
                traj['X'][0], traj['Y'][0] = adjusted_width / 2, adjusted_height / 2
                fixs = [(traj['X'][0], traj['Y'][0])]

                traj_len = len(traj['X'])

                for i in range(1, traj_len):

                    fix = (
                        ((traj['X'][i] - min_fix_x) / max_fix_x) * (512),
                        ((traj['Y'][i] - min_fix_y) / max_fix_y) * (320))

                    # remove returning fixations (enforce inhibition of return)
                    if fix in fixs:
                        continue
                    else:
                        fixs.append(fix)

                    target_fixation = False
                    # print(fix)
                    for poly_t in poly_t_l:

                        # check whether the fixation lies on the polygon
                        c = cv2.pointPolygonTest(np.asarray(poly_t).astype(int), (fix[0], fix[1]), measureDist=False)

                        if c >= 0:
                            target_fixation = True

                    if target_fixation:
                        continue

                    for index, poly_g_l in enumerate(poly_distractors):
                        for polyg in np.asarray(poly_g_l):
                            # for polyg in item:
                            # polyg_dist = Polygon([(polyg[i,0], polyg[i,1]) for i in range(0, len(polyg[:,0]))])
                            c = cv2.pointPolygonTest(np.asarray(polyg).astype(int), (fix[0], fix[1]), measureDist=False)
                            # point = Point(fix[0],fix[1])

                            if c >= 0 and observer[index] != observer_new[index]:
                                # print('fixation on distractor')
                                # print('index of fixated distractor object:', index)
                                distraction_scores[ind, index] += 1
                                observer_new[index] = observer[index]

                                # print(observer)
        # np.sum(distraction_scores[ind, :])
        for j, s in enumerate(distraction_scores[ind, :]):

            if os.path.exists(os.path.join(data_dir, 'parsed_distractor', str(phase),
                                           search_image.split('.')[0] + '_' + str(j + 1) + '.jpg')):
                print('distractor index: ', j)
                print('len(poly_distractors): ', len(poly_distractors))
                distraction.append(int(s))
                data['images'].append({
                    'file_name': search_image,
                    'segmentation': poly_distractors[j],
                    'label': int(s)})

        # print('distractor score for distractors:', distraction_scores[ind, :])

    return data


def mask_seg(mask, pts, label, color):
    new_mask = cv2.fillPoly(mask, [pts.astype(int)], label)  # color[label])

    return new_mask


def process_mask(data_dir, category, data, phase):
    # dictionary for assigning a different color for each distraction mask
    color = {0: (0, 0, 20), 1: (0, 0, 40), 2: (0, 0, 80),
             3: (0, 0, 100), 4: (0, 0, 120), 5: (0, 0, 140),
             6: (0, 0, 160), 7: (0, 0, 180), 8: (0, 0, 200),
             9: (0, 0, 220), 10: (0, 0, 255), 11: (0, 255, 0)}

    task_img_pair = []

    for traj in data['images']:
        task_img_pair.append(traj['file_name'])

    task_img_pair = np.unique(np.asarray(task_img_pair))

    for im in task_img_pair:
        image_path = os.path.join(data_dir, 'images', category, im.split('_')[1])
        img = cv2.resize(cv2.imread(image_path), (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(data_dir, 'distraction', phase, im.split('.')[0] + '.png'), img)

        instances = 0

        for traj in data['images']:
            if traj['file_name'] == im and traj['label'] > 0:
                # print(traj['label'])
                instances += 1
        # print(instances)

        mask = np.zeros((img.shape[0], img.shape[1], instances), dtype="uint8")

        channel = 0

        for traj in data['images']:
            if traj['file_name'] == im:

                label = traj['label']

                if label == 11:  # write target masks on the first channel
                    for seg in traj['segmentation']:
                        poly = np.array(seg)
                        mask[:, :, channel] = mask_seg(mask[:, :, channel].astype("uint8"), poly, 1, color)

                    channel += 1

                if label > 0 and label < 11:

                    for seg in traj['segmentation']:
                        poly = np.array(seg)
                        mask[:, :, channel] = mask_seg(mask[:, :, channel].astype("uint8"), poly, 2, color)
                    channel += 1

        # mask = to_categorical(mask, num_classes=3)

        with open(os.path.join(data_dir, 'distraction', 'mask', im.split('.')[0] + '.pickle'), "wb") as f_out:
            pickle.dump(mask, f_out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dldir', type=str, required=True, help='The directory to download the dataset.',
                        default='../')
    parser.add_argument('--category', type=str, required=True, choices=['bottle', 'bowl', 'car'],
                        help='The target object category.', default='bottle')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.dldir, 'parsed_distractor')):
        os.mkdir(os.path.join(args.dldir, 'parsed_distractor'))
    if not os.path.exists(os.path.join(args.dldir, 'parsed_distractor', 'train')):
        os.mkdir(os.path.join(args.dldir, 'parsed_distractor', 'train'))
    if not os.path.exists(os.path.join(args.dldir, 'parsed_distractor', 'valid')):
        os.mkdir(os.path.join(args.dldir, 'parsed_distractor', 'valid'))
    if not os.path.exists(os.path.join(args.dldir, 'parsed_distractor', 'test')):
        os.mkdir(os.path.join(args.dldir, 'parsed_distractor', 'test'))
    if not os.path.exists(os.path.join(args.dldir, 'distraction')):
        os.mkdir(os.path.join(args.dldir, 'distraction'))
    if not os.path.exists(os.path.join(args.dldir, 'distraction', 'train')):
        os.mkdir(os.path.join(args.dldir, 'distraction', 'train'))
    if not os.path.exists(os.path.join(args.dldir, 'distraction', 'valid')):
        os.mkdir(os.path.join(args.dldir, 'distraction', 'valid'))
    if not os.path.exists(os.path.join(args.dldir, 'distraction', 'test')):
        os.mkdir(os.path.join(args.dldir, 'distraction', 'test'))
    if not os.path.exists(os.path.join(args.dldir, 'distraction', 'mask')):
        os.mkdir(os.path.join(args.dldir, 'distraction', 'mask'))

    with open(os.path.join(args.dldir,
                           'coco_search18_fixations_TP_train_split1.json')) as json_file:
        human_scanpaths_train = json.load(json_file)

    with open(os.path.join(args.dldir,
                           'coco_search18_fixations_TP_validation_split1.json')) as json_file:
        human_scanpaths_valid = json.load(json_file)

    train_task_img_pair = []

    for search_image in human_scanpaths_train:
        if search_image['task'] == args.category:
            train_task_img_pair.append(args.category + '_' + search_image['name'])
    train_task_img_pair = np.unique(np.asarray(train_task_img_pair))

    # separate test set from the training set.
    test_task_img_pair = random.sample(list(train_task_img_pair), 17)

    # remove test set from the training set
    train_task_img_pair = [x for x in train_task_img_pair if x not in test_task_img_pair]
    train_task_img_pair = np.unique(np.asarray(train_task_img_pair))

    valid_task_img_pair = []

    for search_image in human_scanpaths_valid:
        if search_image['task'] == args.category:
            valid_task_img_pair.append(args.category + '_' + search_image['name'])
    valid_task_img_pair = np.unique(np.asarray(valid_task_img_pair))

    with open(os.path.join(args.dldir, args.category + '_segmentation.json')) as json_file:
        annotations = json.load(json_file)

    data_list = []
    for traj in annotations['images']:
        if traj["file_name"] in train_task_img_pair:
            data_list.append((traj["file_name"], traj["id"], traj['width'], traj['height']))

    data_list = np.asarray(data_list)
    data_list_ann = []

    test_list = []
    for traj in annotations['images']:
        if traj["file_name"] in test_task_img_pair:
            test_list.append((traj["file_name"], traj["id"], traj['width'], traj['height']))

    test_list = np.asarray(test_list)
    test_list_ann = []

    for traj in annotations['annotations']:
        if str(traj["image_id"]) in data_list[:, 1]:
            d = data_list[np.where(data_list[:, 1] == str(traj["image_id"]))][0]
            data_list_ann.append([d[0], traj['segmentation'], d[2], d[3],
                                  traj['category_id']])  # [filename, segmentation, width, height, category]
    data_list_ann = np.asarray(data_list_ann)

    for traj in annotations['annotations']:
        if str(traj["image_id"]) in test_list[:, 1]:
            d = test_list[np.where(test_list[:, 1] == str(traj["image_id"]))][0]
            test_list_ann.append([d[0], traj['segmentation'], d[2], d[3],
                                  traj['category_id']])  # [filename, segmentation, width, height, category]
    test_list_ann = np.asarray(test_list_ann)

    valid_list = []
    for traj in annotations['images']:
        if traj["file_name"] in valid_task_img_pair:
            valid_list.append((traj["file_name"], traj["id"], traj['width'], traj['height']))

    valid_list = np.asarray(valid_list)
    valid_list_ann = []

    for traj in annotations['annotations']:
        if str(traj["image_id"]) in valid_list[:, 1]:
            d = valid_list[np.where(valid_list[:, 1] == str(traj["image_id"]))][0]
            valid_list_ann.append([d[0], traj['segmentation'], d[2], d[3],
                                   traj['category_id']])  # [filename, segmentation, width, height, category]
    valid_list_ann = np.asarray(valid_list_ann)

    data_train = preprocess(args.dldir, args.category, train_task_img_pair, human_scanpaths_train, data_list_ann,
                            'train')
    # with open(os.path.join(args.dldir, 'training.txt'), 'w') as outfile:
    #     json.dump(data_train, outfile)

    data_valid = preprocess(args.dldir, args.category, valid_task_img_pair, human_scanpaths_valid, valid_list_ann,
                            'valid')
    # with open(os.path.join(args.dldir, 'validating.txt'), 'w') as outfile:
    #     json.dump(data_valid, outfile)

    data_test = preprocess(args.dldir, args.category, test_task_img_pair, human_scanpaths_train, test_list_ann,
                           'test')
    # with open(os.path.join(args.dldir, 'testing.txt'), 'w') as outfile:
    #     json.dump(data_test, outfile)
    process_mask(args.dldir, args.category, data_train, 'train')
    process_mask(args.dldir, args.category, data_valid, 'valid')
    process_mask(args.dldir, args.category, data_test, 'test')
