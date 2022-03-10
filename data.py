import os
import sys
import numpy as np
import tensorflow as tf
import config
import random 

class COCOSEARCH:

    """This class represents the COCOSearch18 dataset. It consists of 3101
       target-present images. All stimuli are of size 1680x1050 pixels
       and are resized to 512x320 (height by width).

    Attributes:
        n_train: Number of training instances as defined in the dataset.
        n_valid: Number of validation instances as defined in the dataset.

    Returns:
        tuple: A tuple that consists of dataset objects holding the training
               and validation set instances respectively.
    """

    n_train = 2823  # 95*2
    n_valid = 324  # 17*2

    def __init__(self, data_path):

        self._stimuli_size = config.DIMS["image_size_cocosearch"]
        self._target_size = config.DIMS["image_target_size_cocosearch"]

        self._dir_stimuli_train = data_path + "cocosearch/stimuli/train"
        self._dir_stimuli_valid = data_path + "cocosearch/stimuli/valid"

        self._dir_saliency_train = data_path + "cocosearch/saliencymap/train"
        self._dir_saliency_valid = data_path + "cocosearch/saliencymap/valid"

        targ_ind_train = str(random.randint(0, 4))
        targ_ind_valid = str(random.randint(0, 4))

        self._dir_target_train = data_path + "cocosearch/target_" + targ_ind_train + "/train"
        self._dir_target_valid = data_path + "cocosearch/target_" + targ_ind_valid + "/valid"


    def load_data(self):
        
        n_train = 2823  # 95*2
        n_valid = 324  # 17*2

        train_list_x = _get_file_list(self._dir_stimuli_train)
        train_list_y = _get_file_list(self._dir_saliency_train)
        train_list_z = _get_file_list(self._dir_target_train)

        _check_consistency(zip(train_list_x, train_list_y, train_list_z), n_train)

        train_set = _fetch_dataset((train_list_x, train_list_y, train_list_z),
                                    self._stimuli_size, self._target_size, True)

        valid_list_x = _get_file_list(self._dir_stimuli_valid)
        valid_list_y = _get_file_list(self._dir_saliency_valid)
        valid_list_z = _get_file_list(self._dir_target_valid)

        _check_consistency(zip(valid_list_x, valid_list_y, valid_list_z), n_valid)

        valid_set = _fetch_dataset((valid_list_x, valid_list_y, valid_list_z),
                                    self._stimuli_size, self._target_size, False)

        return (train_set, valid_set)


class TEST:

    """This class represents test set instances used for inference through
       a trained network. All stimuli are resized to the preferred spatial
       dimensions of the chosen model. This can, however, lead to cases of
       excessive image padding.
    Returns:
        object: A dataset object that holds all test set instances
                specified under the path variable.
    """
    n_test = 324  # 18

    def __init__(self, dataset, data_path):
        
        self._stimuli_size = config.DIMS["image_size_cocosearch"]
        self._target_size = config.DIMS["image_target_size_cocosearch"]

        targ_ind_test = str(random.randint(0, 4))

        self._dir_stimuli_test = data_path + "cocosearch/stimuli/test"
        self._dir_target_test = data_path + "cocosearch/target_"+ targ_ind_test + "/test"
        self._dir_saliency_test = data_path + "cocosearch/saliencymap/test"
 

    def load_data(self):

        n_test = 324  # 18
        test_list_x = _get_file_list(self._dir_stimuli_test)
        test_list_y = _get_file_list(self._dir_saliency_test)
        test_list_z = _get_file_list(self._dir_target_test)

        _check_consistency(zip(test_list_x, test_list_y, test_list_z), n_test)

        test_set = _fetch_dataset((test_list_x, test_list_y, test_list_z), self._stimuli_size, self._target_size,
                                  False, online=True)

        return test_set


def get_dataset_iterator(phase, dataset, data_path):

    """
    Entry point to make an initializable dataset iterator for either
       training or testing a model by calling the respective dataset class.
    Args:
        phase (str): Holds the current phase, which can be "train" or "test".
        dataset (str): Denotes the dataset to be used during training or the
                       suitable resizing procedure when testing a model.
        data_path (str): Points to the directory where training or testing
                         data instances are stored.
    Returns:
        iterator: An initializable dataset iterator holding the relevant data.
        initializer: An operation required to initialize the correct iterator.
    """

    if phase == "train":

        current_module = sys.modules[__name__]
        class_name = "%s" % dataset.upper()

        dataset_class = getattr(current_module, class_name)(data_path)
        train_set, valid_set = dataset_class.load_data()

        iterator = tf.data.Iterator.from_structure(train_set.output_types,
                                                   train_set.output_shapes)
        next_element = iterator.get_next()

        train_init_op = iterator.make_initializer(train_set)
        valid_init_op = iterator.make_initializer(valid_set)

        return next_element, train_init_op, valid_init_op

    if phase == "test":

        test_class = TEST(dataset, data_path)
        test_set = test_class.load_data()

        iterator = tf.data.Iterator.from_structure(test_set.output_types,
                                                   test_set.output_shapes)
        next_element = iterator.get_next()

        init_op = iterator.make_initializer(test_set)

        return next_element, init_op


def postprocess_saliency_map(saliency_map, target_size):
    """This function resizes and crops a single saliency map to the original
       dimensions of the input image. The output is then encoded as a jpeg
       file suitable for saving to disk.
    Args:
        saliency_map (tensor, float32): 3D tensor that holds the values of a
                                        saliency map in the range from 0 to 1.
        target_size (tensor, int32): 1D tensor that specifies the size to which
                                     the saliency map is resized and cropped.
    Returns:
        tensor, str: A tensor of the saliency map encoded as a jpeg file.
    """

    saliency_map_np = saliency_map * 255.0
    saliency_map = _resize_image(saliency_map_np , target_size, True)


    saliency_map = tf.round(saliency_map)
    saliency_map = tf.cast(saliency_map, tf.uint8)

    saliency_map_jpg = tf.image.encode_jpeg(saliency_map, "grayscale", 100)

    return saliency_map_jpg, saliency_map_np 


def _fetch_dataset(files, stimuli_size, target_size, shuffle, online=False):

    """Here the list of file directories is shuffled (only when training),
       loaded, batched, and prefetched to ensure high GPU utilization.
    Args:
        files (list, str): A list that holds the paths to all file instances.
        target_size (tuple, int): A tuple that specifies the size to which
                                  the data will be reshaped.
        shuffle (bool): Determines whether the dataset will be shuffled or not.
        online (bool, optional): Flag that decides whether the batch size must
                                 be 1 or can take any value. Defaults to False.
    Returns:
        object: A dataset object that contains the batched and prefetched data
                instances along with their shapes and file paths.
    """

    dataset = tf.data.Dataset.from_tensor_slices(files)

    if shuffle:
        dataset = dataset.shuffle(len(files[0]))

    dataset = dataset.map(lambda *files: _parse_function(files, stimuli_size, target_size),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    batch_size = 1 if online else config.PARAMS["batch_size"]

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)

    return dataset


def _parse_function(files, stimuli_size, target_size):
    """This function reads image data dependent on the image type and
       whether it constitutes a stimulus or saliency map. All instances
       are then reshaped and padded to yield the target dimensionality.
    Args:
        files (tuple, str): A tuple with the paths to all file instances.
                            The first element contains the stimuli and, if
                            present, the second one the ground truth maps.
        target_size (tuple, int): A tuple that specifies the size to which
                                  the data will be reshaped.
    Returns:
        list: A list that holds the image instances along with their
              shapes and file paths.
    """

    image_list = []

    for count, filename in enumerate(files):

        image_str = tf.read_file(filename)
        channels = 3 if (count == 0 or count == 2) else 1
        image = tf.cond(tf.image.is_jpeg(image_str),
                        lambda: tf.image.decode_jpeg(image_str,
                                                     channels=channels),
                        lambda: tf.image.decode_png(image_str,
                                                    channels=channels))
        original_size = tf.shape(image)[:2]

        if count == 2:
            image = _resize_image(image, target_size)
        
        elif count ==0 or count==1:

            image = _resize_image(image, stimuli_size)

        image_list.append(image)


    image_list.append(original_size)
    image_list.append(files)

    return image_list


def _resize_image(image, target_size, overfull=False):
    """This resizing procedure preserves the original aspect ratio and might be
       followed by padding or cropping. Depending on whether the target size is
       smaller or larger than the current image size, the area or bicubic
       interpolation method will be utilized.
    Args:
        image (tensor, uint8): A tensor with the values of an image instance.
        target_size (tuple, int): A tuple that specifies the size to which
                                  the data will be resized.
        overfull (bool, optional): Denotes whether the resulting image will be
                                   larger or equal to the specified target
                                   size. This is crucial for the following
                                   padding or cropping. Defaults to False.
    Returns:
        tensor, float32: 4D tensor that holds the values of the resized image.
    .. seealso:: The reasoning for using either area or bicubic interpolation
                 methods is based on the OpenCV documentation recommendations.
                 [https://bit.ly/2XAavw0]
    """

    current_size = tf.shape(image)[:2]

    target_size = tf.cast(tf.round(current_size), tf.int32)

    shrinking = tf.cond(tf.logical_or(current_size[0] > target_size[0],
                                      current_size[1] > target_size[1]),
                        lambda: tf.constant(True),
                        lambda: tf.constant(False))

    image = tf.expand_dims(image, 0)

    image = tf.cond(shrinking,
                    lambda: tf.image.resize_area(image, target_size,
                                                 align_corners=True),
                    lambda: tf.image.resize_bicubic(image, target_size,
                                                    align_corners=True))

    image = tf.clip_by_value(image[0], 0.0, 255.0)

    return image


def _get_file_list(data_path):
    """This function detects all image files within the specified parent
       directory for either training or testing. The path content cannot
       be empty, otherwise an error occurs.
    Args:
        data_path (str): Points to the directory where training or testing
                         data instances are stored.
    Returns:
        list, str: A sorted list that holds the paths to all file instances.
    """

    data_list = []

    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    data_list.append(os.path.join(subdir, file))

    data_list.sort()

    if not data_list:
        raise FileNotFoundError("No data was found")

    return data_list


def _get_random_indices(list_length):
    """A helper function to generate an array of randomly shuffled indices
       to divide the MIT1003 and CAT2000 datasets into training and validation
       instances.
    Args:
        list_length (int): The number of indices that is randomly shuffled.
    Returns:
        array, int: A 1D array that contains the shuffled data indices.
    """

    indices = np.arange(list_length)
    prng = np.random.RandomState(42)
    prng.shuffle(indices)

    return indices


def _check_consistency(zipped_file_lists, n_total_files):
    """A consistency check that makes sure all files could successfully be
       found and stimuli names correspond to the ones of ground truth maps.
    Args:
        zipped_file_lists (tuple, str): A tuple of train and valid path names.
        n_total_files (int): The total number of files expected in the list.
    """

    assert len(list(zipped_file_lists)) == n_total_files, "Files are missing"

    for file_tuple in zipped_file_lists:
        file_names = [os.path.basename(entry) for entry in list(file_tuple)]
        file_names = [os.path.splitext(entry)[0] for entry in file_names]
        file_names = [entry.replace("_fixMap", "") for entry in file_names]
        file_names = [entry.replace("_fixPts", "") for entry in file_names]

        assert len(set(file_names)) == 1, "File name mismatch"

    