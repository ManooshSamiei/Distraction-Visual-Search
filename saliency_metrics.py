
import pysaliency
from pysaliency.baseline_utils import BaselineModel, GoldModel
import argparse
import os
import json
import numpy as np

def compute_saliency_metrics(data_path):

    test_list = []

    test_stimuli_path = data_path + "cocosearch/stimuli/test"

    for subdir, dirs, files in os.walk(test_stimuli_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                test_list.append(os.path.join(subdir, file))

    dir_saliency_test = data_path + "cocosearch/saliencymap/test"
    dir_stimuli_test = pysaliency.FileStimuli(test_list)

    with open(data_path + 'coco_search18_fixations_TP_train_split1.json') as json_file:
        human_scanpaths_train = json.load(json_file)

    xs = []
    ys = []
    ts = []
    ns = []
    train_subjects = []

    for subdir, dirs, files in os.walk(test_stimuli_path):
        for n, stimulus in enumerate(files):            
            stimulus_size = dir_stimuli_test.sizes[n]
            height, width = stimulus_size

            for traj in human_scanpaths_train:
                if ((traj['task'] + '_' + traj['name'])==stimulus):
                    subject_name = traj['subject']

                    xs_ = []
                    ys_ = []
                    durations_ = []

                    for ind in range(len(traj['X'])):
                        
                        if 1680>=traj['X'][ind]>=0 and 1050>=traj['Y'][ind]>=0:
                            xs_.append(traj['X'][ind]*(512/1680))
                            ys_.append(traj['Y'][ind]*(320/1050))
                            durations_.append(traj['T'][ind])

                    time_l_ = durations_ [:-1]
                    time_l_.insert(0,0)
                    time_l_array_ = np.array(time_l_)
                    ts_ = [np.sum(time_l_array_[0:ind]) for ind in range(1, len(time_l_array_)+1)]

                    #print(ts_)
                    #print(xs_)
                    #print(durations_)
                    #print(stimulus)

                    xs.append(xs_)
                    ys.append(ys_)
                    ts.append(ts_)
                    ns.append(n)
                    train_subjects.append(subject_name )

    fixations = pysaliency.FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects, attributes=False, scanpath_attributes=None)

    my_model = pysaliency.SaliencyMapModelFromDirectory(dir_stimuli_test, data_path + 'results/images')

    ground_truth = pysaliency.SaliencyMapModelFromDirectory(dir_stimuli_test,  dir_saliency_test)

    auc = my_model.AUC(dir_stimuli_test , fixations)
    sauc = my_model.sAUC(dir_stimuli_test , fixations)
    nss = my_model.NSS(dir_stimuli_test , fixations)
    fix_kld = my_model.fixation_based_KL_divergence(dir_stimuli_test , fixations)
    cc = my_model.CC(dir_stimuli_test , ground_truth )
    sim = my_model.SIM(dir_stimuli_test , ground_truth )

    img_kld = my_model.image_based_kl_divergence(dir_stimuli_test , ground_truth)

    print('AUC:', auc )
    print('sAUC:', sauc )
    print('NSS:', nss )
    print('image_KLD:', img_kld) # this should be same as our loss function
    print('fixation_KLD:', fix_kld)
    print('CC:', cc )
    print('SIM:', sim )

def main():
    
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--path",  type=str, 
                        help="specify the root path to data")

    args = parser.parse_args()

    compute_saliency_metrics(args.path)
        

if __name__ == "__main__":
    main()