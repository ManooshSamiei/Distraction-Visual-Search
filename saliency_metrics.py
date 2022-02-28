import pysaliency
from pysaliency.baseline_utils import BaselineModel, GoldModel
import argparse
import os
import json
import numpy as np
import cv2
import metrics
import tensorflow as tf
import random
import glob
import shutil

tf.compat.v1.enable_eager_execution()

def compute_saliency_metrics(data_path, use_pysaliency):

    test_list = []

    test_stimuli_path = data_path + "cocosearch/stimuli/test"

    for subdir, dirs, files in os.walk(test_stimuli_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                test_list.append(os.path.join(subdir, file))

    dir_saliency_test = data_path + "cocosearch/saliencymap/test"
    dir_saliency_test_unblur = data_path + "cocosearch/saliencymap/test_unblur"
    dir_stimuli_test = pysaliency.FileStimuli(test_list)
    dir_saliency_test_img = os.path.join(dir_saliency_test , 'pysaliency_sal_img')
    
    if os.path.exists(dir_saliency_test_img):
        shutil.rmtree(dir_saliency_test_img)
        os.makedirs(dir_saliency_test_img)
    else:
        os.makedirs(dir_saliency_test_img)
     
    dir_results_test_img = os.path.join(data_path + 'results/images' , 'pysaliency_result_img')

    if os.path.exists(dir_results_test_img):
        shutil.rmtree(dir_results_test_img)
        os.makedirs(dir_results_test_img)
    else:
        os.makedirs(dir_results_test_img)

    for item in glob.glob(os.path.join(dir_saliency_test , '*.jpg')):
        os.symlink(item , os.path.join(dir_saliency_test_img , os.path.basename(item)))

    for item in glob.glob(os.path.join(data_path + 'results/images' , '*.jpg')):
        os.symlink(item, os.path.join(dir_results_test_img , os.path.basename(item)))


    with open(data_path + 'coco_search18_fixations_TP_train_split1.json') as json_file:
        human_scanpaths_train = json.load(json_file)

    xs = []
    ys = []
    ts = []
    ns = []
    train_subjects = []
    min_fix_x = 100000
    max_fix_x = -100000
    min_fix_y = 100000
    max_fix_y = -100000

    for traj in human_scanpaths_train:
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

    m_kld_error, m_cc_error, m_sim_error, m_nss_error, m_auc_error, m_infog_error, m_sauc_error, m_auc_b_error = 0,0,0,0,0,0,0,0
    count = len(os.listdir(test_stimuli_path))
    for subdir, dirs, files in os.walk(test_stimuli_path):


        for n, stimulus in enumerate(files):


            base_line_salmap = np.zeros((320 , 512))
            for j in files:
                 if j!= stimulus:
                     base_line_salmap = base_line_salmap + np.load(os.path.join(dir_saliency_test , os.path.splitext(j)[0]+'.npy'), allow_pickle=True)

            base_line_salmap /= np.max(base_line_salmap)
            M = 1
            random_ind = random.sample(range(1, count), M)
            base_line_fixmap = np.zeros((320 , 512))
            for i in random_ind:

                if files[i]!=stimulus:
                    base_line_fixmap = base_line_fixmap + (np.load(os.path.join(dir_saliency_test_unblur , os.path.splitext(files[i])[0]+'.npy'), allow_pickle=True))/255
                    
                else:
                    rand_ind = random.randint(0, count)
                    base_line_fixmap = base_line_fixmap + (np.load(os.path.join(dir_saliency_test_unblur , os.path.splitext(files[rand_ind])[0]+'.npy'), allow_pickle=True))/255

            base_line_fixmap[np.where(base_line_fixmap>1.0)] = 1.0

            ##gnd_map = tf.image.decode_jpeg(tf.read_file(os.path.join(dir_saliency_test , stimulus)), channels=1).numpy()
            ##gnd_bin_map = tf.image.decode_jpeg(tf.read_file(os.path.join(dir_saliency_test_unblur , stimulus)), channels=1).numpy()
            ##pred_map = tf.image.decode_jpeg(tf.read_file(os.path.join(data_path + 'results/images' , os.path.splitext(stimulus)[0]+'.png')), channels=1).numpy()

            gnd_map  = np.load(os.path.join(dir_saliency_test , os.path.splitext(stimulus)[0]+'.npy'), allow_pickle=True)      
            gnd_bin_map  = np.load(os.path.join(dir_saliency_test_unblur , os.path.splitext(stimulus)[0]+'.npy'), allow_pickle=True)
            pred_map = np.load(os.path.join(data_path + 'results/images' , os.path.splitext(stimulus)[0]+'.npy'), allow_pickle=True)

            kl_test_error, cc_test_error, sim_test_error, nss_test_error, auc_test_error, infog_test_error, sauc_test_error, auc_b_test_error = metrics.calculate_metrics(pred_map, gnd_map, gnd_bin_map, base_line_fixmap, base_line_salmap)

            m_kld_error += kl_test_error
            m_cc_error += cc_test_error
            m_sim_error += sim_test_error
            m_nss_error += nss_test_error
            m_auc_error += auc_test_error
            m_infog_error += infog_test_error
            m_sauc_error += sauc_test_error
            m_auc_b_error += auc_b_test_error
            
            if use_pysaliency:

                stimulus_size = dir_stimuli_test.sizes[n]
                height, width = stimulus_size

                for traj in human_scanpaths_train:

                    
                    if ((traj['task'] + '_' + traj['name'])==stimulus and traj['correct'] == 1):
                        subject_name = traj['subject']

                        xs_ = []
                        ys_ = []
                        durations_ = []

                        for ind in range( len(traj['X'])):
                            
                            if 1680>=traj['X'][ind]>=0 and 1050>=traj['Y'][ind]>=0:
                                if not (traj['X'][ind] in xs_ and traj['Y'][ind] in ys_):
                                    xs_.append(((traj['X'][ind] - min_fix_x) / max_fix_x)*(512))
                                    ys_.append(((traj['Y'][ind] - min_fix_y) / max_fix_y)*(320))
                                    durations_.append(traj['T'][ind])


                        time_l_ = durations_ [:-1]
                        time_l_.insert(0,0)
                        time_l_array_ = np.array(time_l_)
                        ts_ = [np.sum(time_l_array_[0:ind]) for ind in range(1, len(time_l_array_)+1)]

                        xs.append(xs_)
                        ys.append(ys_)
                        ts.append(ts_)
                        ns.append(n)
                        train_subjects.append(subject_name )

    if use_pysaliency:

        fixations = pysaliency.FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects, attributes=False, scanpath_attributes=None)

        my_model = pysaliency.SaliencyMapModelFromDirectory(dir_stimuli_test, dir_results_test_img)

        ground_truth = pysaliency.SaliencyMapModelFromDirectory(dir_stimuli_test,  dir_saliency_test_img)

        auc = my_model.AUC(dir_stimuli_test , fixations)
        sauc = my_model.sAUC(dir_stimuli_test , fixations)
        nss = my_model.NSS(dir_stimuli_test , fixations)
        cc = my_model.CC(dir_stimuli_test , ground_truth)
        sim = my_model.SIM(dir_stimuli_test , ground_truth)

        img_kld = my_model.image_based_kl_divergence(dir_stimuli_test , ground_truth)

        print('metrics computed by pysaliency:')
        print('AUC:', auc )
        print('sAUC:', sauc )
        print('NSS:', nss )
        print('image_KLD:', img_kld) 
        print('CC:', cc )
        print('SIM:', sim )

    kl_mean_test_error = m_kld_error / count
    cc_mean_test_error = m_cc_error / count
    sim_mean_test_error = m_sim_error / count 
    nss_mean_test_error = m_nss_error / count
    auc_mean_test_error = m_auc_error / count
    infog_mean_test_error = m_infog_error / count
    sauc_mean_test_error = m_sauc_error / count
    auc_b_mean_test_error = m_auc_b_error / count

    print('locally computed:')
    print('AUC:',  auc_mean_test_error)
    print('AUC Borji:',  auc_b_mean_test_error)
    print('SAUC:',  sauc_mean_test_error)
    print('NSS:',  nss_mean_test_error )
    print('image_KLD:', kl_mean_test_error) # this should be same as our loss function
    print('CC:', cc_mean_test_error )
    print('SIM:',sim_mean_test_error )
    print('INFO GAIN:', infog_mean_test_error )

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--path",  type=str, 
                        help="specify the root path to data")
    parser.add_argument('--use-pysaliency', type=str2bool, required=True,
                        default=False,
                        help='Determines whether to use pysaliency for the saliency metrics computation.')
    args = parser.parse_args()

    compute_saliency_metrics(args.path, args.use_pysaliency)
        

if __name__ == "__main__":
    main()