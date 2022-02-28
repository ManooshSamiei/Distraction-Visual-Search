import argparse
import os
import numpy as np
import tensorflow as tf
import config
import data
import model
import utils
import random
import sys
import cv2
import metrics

seed_value = 32
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_random_seed(seed_value)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

def define_paths(current_path):
    """A helper function to define all relevant path elements for the
       locations of data, weights, and the results from either training
       or testing a model.
    Args:
        current_path (str): The path string of this script.
        args (object): A namespace object with values from command line.
    Returns:
        dict: A dictionary with all path elements.
    """

    data_path = current_path 
    results_path = current_path + "results/"
    weights_path = current_path + "weights/"

    history_path = results_path + "history/"
    images_path = results_path + "images/"
    ckpts_path = results_path + "ckpts/"

    best_path = ckpts_path + "best/"
    latest_path = ckpts_path + "latest/"

    paths = {
        "data": data_path,
        "history": history_path,
        "images": images_path,
        "best": best_path,
        "latest": latest_path,
        "weights": weights_path
    }

    return paths


def train_model(dataset, paths, device):
    """The main function for executing network training. It loads the specified
       dataset iterator, saliency model, and helper classes. Training is then
       performed in a new session by iterating over all batches for a number of
       epochs. After validation on an independent set, the model is saved and
       the training history is updated.
    Args:
        dataset (str): Denotes the dataset to be used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """

    iterator = data.get_dataset_iterator("train", dataset, paths["data"])

    next_element, train_init_op, valid_init_op = iterator

    input_images, ground_truths, input_targets  = next_element[:3]
    ground_truths = tf.divide(ground_truths, 255)

    input_plhd = tf.placeholder_with_default(input_images,
                                             (None, None, None, 3),
                                             name="input")

    input_target_img = tf.placeholder_with_default(input_targets,
                                                   (None, None, None, 3),
                                                   name="input_2")

    msinet = model.MSINET()

    feature_map_stimuli = msinet.forward(input_plhd)

    feature_map_target = msinet.forward(input_target_img)

    predicted_maps = msinet.output_stream(feature_map_stimuli, feature_map_target)

    # max_out = tf.math.reduce_min(predicted_maps)

    optimizer, loss = msinet.train(ground_truths, predicted_maps,
                                   config.PARAMS["learning_rate"])

    n_train_data = getattr(data, dataset.upper()).n_train
    n_valid_data = getattr(data, dataset.upper()).n_valid

    n_train_batches = int(np.ceil(n_train_data / config.PARAMS["batch_size"]))
    n_valid_batches = int(np.ceil(n_valid_data / config.PARAMS["batch_size"]))

    history = utils.History(n_train_batches,
                            n_valid_batches,
                            dataset,
                            paths["history"],
                            device)

    progbar = utils.Progbar(n_train_data,
                            n_train_batches,
                            config.PARAMS["batch_size"],
                            config.PARAMS["n_epochs"],
                            history.prior_epochs)


    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        saver = msinet.restore(sess, dataset, paths, device)
        writer = tf.summary.FileWriter('./tflogs', sess.graph)
        print(">> Start training on %s..." % dataset.upper())

        for epoch in range(config.PARAMS["n_epochs"]):
            sess.run(train_init_op)

            for batch in range(n_train_batches):
                # print((sess.run([ predicted_maps])).shape)#, ground_truths, input_targets))
                _, error = sess.run([optimizer, loss])
                # print(sess.run([tf.math.reduce_max(ground_truths)]))
                # print(tf.shape(predicted_maps) )
                # print(sess.run([tf.math.reduce_max(predicted_maps)]))

                history.update_train_step(error)
                progbar.update_train_step(batch)

            sess.run(valid_init_op)

            for batch in range(n_valid_batches):
                error = sess.run(loss)

                history.update_valid_step(error)
                progbar.update_valid_step()

            msinet.save(saver, sess, dataset, paths["latest"], device)

            history.save_history()

            progbar.write_summary(history.get_mean_train_error(),
                                history.get_mean_valid_error())

            if history.valid_history[-1] == min(history.valid_history):
                msinet.save(saver, sess, dataset, paths["best"], device)
                msinet.optimize(sess, dataset, paths["best"], device)

                print("\tBest model!", flush=True)


def test_model(dataset, paths, device):
    """The main function for executing network testing. It loads the specified
       dataset iterator and optimized saliency model. By default, when no model
       checkpoint is found locally, the pretrained weights will be downloaded.
       Testing only works for models trained on the same device as specified in
       the config file.
    Args:
        dataset (str): Denotes the dataset that was used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """

    iterator = data.get_dataset_iterator("test", dataset, paths["data"])

    next_element, init_op = iterator

    # input_images, ground_truths, input_targets = next_element[:2]
    # input_images, input_targets = next_element[:2]
    input_images, ground_truths, input_targets, original, file_path = next_element

    original_shape = (512, 320)
    # file_path = next_element[0][-1]
    graph_def = tf.GraphDef()

    model_name = "model_%s_%s.pb" % (dataset, device)

    if os.path.isfile(paths["best"] + model_name):
        with tf.gfile.Open(paths["best"] + model_name, "rb") as file:
            graph_def.ParseFromString(file.read())

    predicted_maps = tf.import_graph_def(graph_def,
                                         input_map={"input": input_images, "input_2": input_targets},
                                         return_elements=["output:0"])
    # p_map = predicted_maps[0]
    predicted_maps = tf.squeeze(predicted_maps, axis=0)
    input_images = tf.squeeze(input_images, axis=0)
    # print(tf.shape(predicted_maps))
    jpeg , npy = data.postprocess_saliency_map(predicted_maps[0], original_shape)
    
    n_test_data = getattr(data, 'TEST').n_test

    # jpeg = tf.transpose(jpeg , (1,2,0) )
    print(">> Start testing with %s %s model..." % (dataset.upper(), device))

    m_kld_error, m_cc_error, m_sim_error, m_nss_error, m_auc_error = 0,0,0,0,0

    with tf.Session() as sess:
        sess.run(init_op)

        while True:
            try:

                output_file_jpeg, output_file_npy, path = sess.run(
                    [jpeg, npy, file_path])

            except tf.errors.OutOfRangeError:
                break

            # print(path)
            path = path[0][0].decode("utf-8")

            filename = os.path.basename(path)
            filename = os.path.splitext(filename)[0]
            filename_jpeg = filename + ".jpg"
            filename_npy = filename + ".npy"

            os.makedirs(paths["images"], exist_ok=True)

            with open(paths["images"] + filename_jpeg, "wb") as file:
                file.write(output_file_jpeg)

            with open(paths["images"] + filename_npy, "wb") as file:
                np.save(file, output_file_npy)
               

def jet_map(paths, threshold=30, alpha=0.5):

    test_data_path = paths['data'] + 'cocosearch/stimuli/test_targ_bbox/'
    prediction_path = paths['images']
    output_dir = paths['images'] + 'images_jet/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, dirs, files in os.walk(test_data_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
  
              img = cv2.imread(test_data_path + file)
              heatmap = cv2.imread( prediction_path + file)

              heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
              
              # Create mask
              mask = np.where(heatmap<=threshold, 1, 0)
              mask = np.reshape(mask, (img.shape[0] , img.shape[1], 3))

              # Marge images
              marge = img*mask + heatmap_color*(1-mask)
              marge = marge.astype("uint8")

              marge = cv2.addWeighted(img, 1-alpha, marge, alpha,0)
              cv2.imwrite( output_dir + file ,marge)

def main():
    
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    phases_list = ["train", "test"]

    # dataset = 'salicon' #input('Which dataset do you want to use? (enter salicon/mit1003) ')
    dataset = 'cocosearch'

    default_data_path = "./"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--phase", metavar="PHASE", choices=phases_list,  type=str,
                        help="sets the network phase (allowed: train or test)")

    parser.add_argument("--path", default=default_data_path,  type=str, 
                        help="specify the path where training data will be \
                              downloaded to or test data is stored")

    parser.add_argument("--threshold", default=30,  type=int,
                        help="jet map threshold for predicted saliency maps.")

    args = parser.parse_args()

    paths = define_paths(args.path)

    tf.reset_default_graph()

    if args.phase == "train":
        train_model(dataset, paths, config.PARAMS["device"])
    elif args.phase == "test":
        test_model(dataset, paths, config.PARAMS["device"])

    jet_map(paths, args.threshold, alpha=0.5)

        

if __name__ == "__main__":
    main()