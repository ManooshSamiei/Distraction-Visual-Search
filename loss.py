import tensorflow as tf
import numpy as np


def kld(y_true, y_pred, eps=1e-7):
    """This function computes the Kullback-Leibler divergence between ground
      truth saliency maps and their predictions. Values are first divided by
      their sum for each image to yield a distribution that adds to 1.
    Args:
        y_true (tensor, float32): A 4d tensor that holds the ground truth
                                  saliency maps with values between 0 and 255.
        y_pred (tensor, float32): A 4d tensor that holds the predicted saliency
                                  maps with values between 0 and 1.
        eps (scalar, float, optional): A small factor to avoid numerical
                                      instabilities. Defaults to 1e-7.
    Returns:
        tensor, float32: A 0D tensor that holds the averaged error.
    """

    sum_per_image = tf.reduce_sum(y_true, axis=(1, 2, 3), keep_dims=True)
    y_true /= eps + sum_per_image

    sum_per_image = tf.reduce_sum(y_pred, axis=(1, 2, 3), keep_dims=True)
    y_pred /= eps + sum_per_image

    loss = y_true * tf.log(eps + y_true / (eps + y_pred))
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))

    return loss


def cc(y_true, y_pred):
    y_pred_norm = (y_pred - tf.math.reduce_mean(y_pred)) / tf.math.reduce_std(y_pred)
    y_true_norm = (y_true - tf.math.reduce_mean(y_true)) / tf.math.reduce_std(y_true)
    a = y_pred_norm
    b = y_true_norm
    r = tf.reduce_sum(tf.math.multiply(a, b), axis=(1, 2, 3)) \
        / tf.math.sqrt(tf.math.multiply(tf.reduce_sum(tf.math.multiply(a, a), axis=(1, 2, 3)),
                                        tf.reduce_sum(tf.math.multiply(b, b), axis=(1, 2, 3))));
    return r


def similarity(y_true, y_pred):
    # here gt is normalized
    y_pred = y_pred / (tf.reduce_sum(y_pred, axis=(1, 2, 3)))
    y_true = y_true / (tf.reduce_sum(y_true, axis=(1, 2, 3)))
    # print(sess.run(tf.where(tf.greater_equal(y_true, 0.))))
    # a = sess.run(tf.where(tf.greater_equal(y_true, 0.)))
    # a = tf.cast(a,tf.int32)
    # sim = 0.0
    # for i in a:

    sim = tf.reduce_sum(tf.math.minimum(y_true, y_pred))
    return sim


def nss(y_true, y_pred):
    y_true = y_true / 255.0
    y_pred = y_pred / 255.0
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    y_pred_norm = (y_pred - tf.reduce_mean(y_pred)) / tf.math.reduce_std(y_pred)
    # a = (tf.where(tf.math.greater(y_true, 0.5))).eval()
    a = (tf.where(tf.math.equal(y_true, 1.0))).eval()
    # print(a)
    temp = []
    y_pred_norm = y_pred_norm.eval()
    for i in a:
        # temp.append(tf.gather(y_pred_norm,i))
        temp.append(y_pred_norm[i[0], i[1]])
        # print(y_pred_norm[i[0] , i[1]])

    return tf.reduce_mean(temp)


def auc_judd(y_pred, y_true):
    # ground truth is discrete, s_map is continous and normalized
    y_true = np.squeeze(y_true) / y_true.max()
    # print( y_true.max())
    y_pred = np.squeeze(y_pred) / y_pred.max()
    # thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = []
    for i in range(0, y_true.shape[0]):
        for k in range(0, y_true.shape[1]):
            if y_true[i][k] > 0:
                thresholds.append(y_pred[i][k])

    num_fixations = np.sum(y_true)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))
    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map, keep only those pixels with values above threshold
        temp = np.zeros(y_pred.shape)
        temp[y_pred >= thresh] = 1.0
        assert np.max(
            y_true) == 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
        assert np.max(
            y_pred) == 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
        num_overlap = np.where(np.add(temp, y_true) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / ((np.shape(y_true)[0] * np.shape(y_true)[1]) - num_fixations)

        area.append((round(tp, 4), round(fp, 4)))
        # tp_list.append(tp)
        # fp_list.append(fp)

    # tp_list.reverse()
    # fp_list.reverse()
    area.append((1.0, 1.0))
    # tp_list.append(1.0)
    # fp_list.append(1.0)
    # print tp_list
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))
