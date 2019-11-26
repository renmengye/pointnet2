# PointNet++ with Continuous Filter Convolution
# Modified from poiunt2_cls_ssg.py
#
# PointNet++ Model for point clouds classification
from __future__ import division, print_function
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module
cont_filter = tf.load_op_library(
    'tf_ops/cont_filter/tf_cont_filter_conv_so.so')
from sklearn.neighbors import NearestNeighbors

from cont_filter import cont_filter_interp, cont_filter_conv


def get_knn(in_xyz, out_xyz, k=10):
    """
    :param in_xyz:     [B, N, 3]
    :param out_xyz:    [B, M, 3]
    :return            offset [B, M, K, 3] neighbors ID [B, M, K]
    """
    offset = np.zeros(
        [in_xyz.shape[0], out_xyz.shape[1], k, 3], dtype=in_xyz.dtype)
    neighbors = np.zeros(
        [in_xyz.shape[0], out_xyz.shape[1], k], dtype=np.int64)
    for b in range(in_xyz.shape[0]):
        _in_xyz = in_xyz[b]
        _out_xyz = out_xyz[b]
        nbrs = NearestNeighbors(
            n_neighbors=k, algorithm='ball_tree').fit(_in_xyz)
        _, idx = nbrs.kneighbors(_out_xyz)  # [N, K]
        offset[b] = _in_xyz[idx] - _out_xyz[:, np.newaxis, :]
        neighbors[b] = idx
    return offset, neighbors


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(
        tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz
    ANCHOR = 27
    verbose = False

    def _fc(points, w, bn=True):
        N = points.shape[0]
        in_channels = w.shape[0]
        out_channels = w.shape[1]
        points_ = tf.reshape(points, [-1, in_channels])
        points_ = tf.matmul(points_, w)
        points = tf.reshape(points_, [N, -1, out_channels])
        if bn:
            points = tf.contrib.layers.batch_norm(
                points,
                center=True,
                scale=True,
                is_training=is_training,
                decay=bn_decay,
                updates_collections=None)
        return points

    # channels = [64, 128, 256, 512]
    # nlayers = len(channels) - 1
    # Official version, no transformer.
    channels = [64, 64, 64, 128, 1024]
    # (1024, 3)
    # (1024, 64)
    # (1024, 64)
    # (1024, 64)
    # (1024, 128)
    # (1024, 1024)
    nlayers = len(channels) - 1
    K = 10
    CUBE = 0.1
    offset, nn = tf.py_func(lambda x, y: get_knn(x, y, k=K), [l0_xyz, l0_xyz],
                            [tf.float32, tf.int64])
    # points = tf.ones_like(l0_xyz[:, :, 0:1])
    # points = tf.reduce_mean(offset, [2])
    # points = tf.reshape(points, [16, 1024, 3])
    points = l0_xyz
    alpha = cont_filter_interp(offset, CUBE, ANCHOR)
    fc0 = tf.get_variable(
        'w0', [3, channels[0]],
        initializer=tf.initializers.truncated_normal(
            0.0, 2.0 / np.sqrt(float(channels[0]))))
    points = _fc(points, fc0, bn=False)
    points = tf.nn.relu(points)
    for layer in range(nlayers):
        conv_init = tf.initializers.uniform_unit_scaling(1.0)
        conv_w = tf.get_variable(
            'convw{}'.format(layer + 1), [
                ANCHOR,
                channels[layer],
            ],
            initializer=conv_init,
            trainable=True)
        fcw = tf.get_variable(
            'w{}'.format(layer + 1), [channels[layer], channels[layer + 1]],
            initializer=tf.initializers.truncated_normal(
                0.0, 2.0 / np.sqrt(float(channels[layer + 1]))))
        points = cont_filter_conv(points, nn, alpha, conv_w)
        points = _fc(points, fcw, bn=True)
        points = tf.nn.relu(points)

    points = tf.reduce_max(points, [1])  # [B, C]

    # Fully connected layers
    net = tf.reshape(points, [batch_size, channels[-1]])
    net = tf_util.fully_connected(
        net,
        512,
        bn=True,
        is_training=is_training,
        scope='fc1',
        bn_decay=bn_decay)
    net = tf_util.dropout(
        net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(
        net,
        256,
        bn=True,
        is_training=is_training,
        scope='fc2',
        bn_decay=bn_decay)
    net = tf_util.dropout(
        net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        output, _ = get_model(inputs, tf.constant(True))
