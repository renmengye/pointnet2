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
from pointnet_util import pointnet_sa_module, sample_and_group
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
        # print('{} ---> {} dist={}'.format(in_xyz.shape[1], out_xyz.shape[
        #     1], np.mean(np.mean(np.abs(offset[b]), axis=-1), axis=0)))
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
    # npoints = [256, 64, 10]
    # npoints = [512, 128, 10]
    # npoints = [512, 128, 8]
    npoints = [512, 128, 32]
    ANCHOR = 27
    l1_xyz, l1_points, l1_nn, l1_offset = sample_and_group(
        npoint=npoints[0],
        radius=0.2,
        nsample=32,
        xyz=l0_xyz,
        points=None,
        knn=False,
        use_xyz=True)
    l2_xyz, l2_points, l2_nn, l2_offset = sample_and_group(
        npoint=npoints[1],
        radius=0.4,
        nsample=64,
        xyz=l1_xyz,
        points=None,
        knn=False,
        use_xyz=True)
    l3_xyz, l3_points, l3_nn, l3_offset = sample_and_group(
        npoint=npoints[2],
        radius=0.8,
        nsample=64,
        xyz=l2_xyz,
        points=None,
        knn=False,
        use_xyz=True)

    # # (1024, 3)   ->  (1024,  64)  fc-0
    # # (1024, 64)  ->  (1024,  64)  conv-1
    # # (1024, 64)  ->  (1024, 128)  fc-1
    # # (1024, 128) ->  (1024, 256)  conv-2
    # # (1024, 256) ->  (1024, 256)  fc-2
    # k_list = [10, 10, 10]
    # cube_list = [0.2, 0.2, 0.2]
    # channels = [64, 128, 256]
    # xyz_list = [l0_xyz, l0_xyz, l0_xyz, l0_xyz]

    # # (1024, 3)  -> (1024, 64)  fc-0
    # # (1024, 64) -> (512,  64)  conv-1
    # # (512,  64) -> (512, 256)  fc-1
    # # (512, 256) -> (10,  256)  conv-2
    # # (10, 1024) -> (10, 1024)  fc-2
    # k_list = [32, 64, 64]
    # cube_list = [0.1, 0.2, 0.4]
    # channels = [64, 256, 512]
    # xyz_list = [l0_xyz, l1_xyz, l2_xyz, l3_xyz]

    # # (1024, 3)  -> (1024, 64)  fc-0
    # # (1024, 64) -> (512,  64)  conv-1
    # # (512,  64) -> (512, 256)  fc-1
    # # (512, 256) -> (10,  256)  conv-2
    # # (10, 1024) -> (10, 1024)  fc-2
    # k_list = [32, 64, 128]
    # cube_list = [0.2, 0.4, 1.0]
    # channels = [64, 256, 512]
    # xyz_list = [l0_xyz, l1_xyz, l2_xyz, l3_xyz]

    # # (1024, 3)  -> (1024, 64)  fc-0
    # # (1024, 64) -> (512,  64)  conv-1
    # # (512,  64) -> (512, 256)  fc-1
    # # (512, 256) -> (10,  256)  conv-2
    # # (10, 1024) -> (10, 1024)  fc-2
    # k_list = [10, 10, 10]
    # cube_list = [0.1, 0.2, 0.4]
    # channels = [64, 256, 1024]
    # xyz_list = [l0_xyz, l1_xyz, l2_xyz, l3_xyz]

    # k_list = [10, 10, 10, 10, 10]
    # cube_list = [0.2, 0.2, 0.4, 0.4, 0.8]
    # channels = [64, 64, 256, 256, 1024]
    # xyz_list = [l0_xyz, l1_xyz, l1_xyz, l2_xyz, l2_xyz, l3_xyz]

    # # (1024,  3) -> (1024, 32)  fc-0
    # # (1024, 32) -> (1024, 32)  conv-1-1 K=32 l0 -> l0
    # # (1024, 32) -> (1024, 32)  fc-1-1
    # # (1024, 32) -> (512,  32)  conv-1-2 K=32 l0 -> l1
    # # (512,  32) -> (512, 128)  fc-1-2
    # # (512, 128) -> (512, 128)  conv-2-1 K=16 l1 -> l1
    # # (512, 128) -> (512, 128)  fc-2-1
    # # (512, 128) -> (128, 128)  conv-2-2 K=16 l1 -> l2
    # # (128, 128) -> (128, 512)  fc-2-2
    # # (128, 512) -> (128, 512)  conv-3-1 K=8  l2 -> l2
    # # (128, 512) -> (128, 512)  fc-3-1
    # # (128, 512) -> (8,   512)  conv-3-2 K=8  l2 -> l3
    # # (8,   512) -> (8,   512)  fc 3-2
    # k_list = [32, 32, 16, 16, 8, 8]
    # cube_list = [0.2, 0.2, 0.4, 0.4, 0.8, 0.8]
    # channels = [32, 32, 128, 128, 512, 512]
    # xyz_list = [l0_xyz, l0_xyz, l1_xyz, l1_xyz, l2_xyz, l2_xyz, l3_xyz]

    # (1024,  3) -> (1024, 32)  fc-0
    # (1024, 32) -> (1024, 32)  conv-1-1 K=32   l0 -> l0
    # (1024, 32) -> (1024, 32)  fc-1-1
    # (1024, 32) -> (512,  32)  conv-1-2 K=32   l0 -> l1
    # (512,  32) -> (512, 128)  fc-1-2
    # (512, 128) -> (512, 128)  conv-2-1 K=16   l1 -> l1
    # (512, 128) -> (512, 128)  fc-2-1
    # (512, 128) -> (128, 128)  conv-2-2 K=16   l1 -> l2
    # (128, 128) -> (128, 512)  fc-2-2
    # (128, 512) -> (128, 512)  conv-3-1 K=16   l2 -> l2
    # (128, 512) -> (128, 512)  fc-3-1
    # (128, 512) -> (1,   512)  conv-3-2 K=128  l2 -> l3
    # (1,   512) -> (1,   512)  fc 3-2
    # k_list = [32, 32, 16, 16, 16, 128]
    # cube_list = [0.2, 0.2, 0.4, 0.4, 0.8, 1.2]
    # channels = [32, 32, 128, 128, 512, 512]
    # xyz_list = [l0_xyz, l0_xyz, l1_xyz, l1_xyz, l2_xyz, l2_xyz, l3_xyz]

    k_list = [32, 32, 16, 16, 16, 16]
    cube_list = [0.2, 0.2, 0.4, 0.4, 0.8, 0.8]
    channels = [32, 32, 128, 128, 512, 512]
    xyz_list = [l0_xyz, l0_xyz, l1_xyz, l1_xyz, l2_xyz, l2_xyz, l3_xyz]

    # # (1024,  3) -> (1024, 32)  fc-0
    # # (1024, 32) -> (1024, 32)  conv-1-1 K=10 l0 -> l0
    # # (1024, 32) -> (1024, 32)  fc-1-1
    # # (1024, 32) -> (1024, 32)  conv-1-2 K=10 l0 -> l0
    # # (1024, 32) -> (1024, 32)  fc-1-2
    # # (1024, 32) -> (1024, 32)  conv-1-3 K=10 l0 -> l0
    # # (1024, 32) -> (1024, 32)  fc-1-3
    # # (1024, 32) -> (512,  32)  conv-1-4 K=10 l0 -> l1
    # # (512,  32) -> (512, 128)  fc-1-4
    # # (512, 128) -> (512, 128)  conv-2-1 K=10 l1 -> l1
    # # (512, 128) -> (512, 128)  fc-2-1
    # # (512, 128) -> (128, 128)  conv-2-2 K=10 l1 -> l2
    # # (128, 128) -> (128, 512)  fc-2-2
    # # (128, 512) -> (128, 512)  conv-3-1 K=10 l2 -> l2
    # # (128, 512) -> (128, 512)  fc-3-1
    # # (128, 512) -> (10,  512)  conv-3-2 K=10 l2 -> l3
    # # (10,  512) -> (10,  512)  fc 3-2
    # k_list = [10, 10, 10, 10, 10, 10, 10, 10]
    # cube_list = [0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.8, 0.8]
    # channels = [32, 32, 32, 32, 128, 128, 512, 512]
    # xyz_list = [l0_xyz, l0_xyz, l0_xyz, l0_xyz, l1_xyz, l1_xyz, l2_xyz, l2_xyz, l3_xyz]

    nlayers = len(xyz_list) - 1
    if True:
        offset_list = [
            tf.py_func(get_knn, [_xyz, _xyz2, _k], [tf.float32, tf.int64])
            for _xyz, _xyz2, _k in zip(xyz_list[:-1], xyz_list[1:], k_list)
        ]
        nn_list = [_offset[1] for _offset in offset_list]
        offset_list = [_offset[0] for _offset in offset_list]
        offset00, _ = tf.py_func(get_knn,
                                 [xyz_list[0], xyz_list[0],
                                  k_list[0]], [tf.float32, tf.int64])

    else:
        offset_list = [l1_offset, l2_offset, l3_offset]
        nn_list = [l1_nn, l2_nn, l3_nn]

    nn_list = [tf.cast(nn, tf.int64) for nn in nn_list]
    alpha_list = [
        cont_filter_interp(_offset, _cube, ANCHOR)
        for (_offset, _cube) in zip(offset_list, cube_list)
    ]

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

    def _conv_block(points, knn, alpha, c_in, c_out):
        conv_init = tf.initializers.uniform_unit_scaling(1.0)
        conv_w = tf.get_variable(
            'convw', [
                ANCHOR,
                c_in,
            ], initializer=conv_init, trainable=True)
        fc_w = tf.get_variable(
            'fcw', [c_in, c_out],
            initializer=tf.initializers.truncated_normal(
                0.0, 2.0 / np.sqrt(float(c_out))),
            trainable=True)
        points = cont_filter_conv(points, knn, alpha, conv_w)
        points = _fc(points, fc_w, bn=True)
        points = tf.nn.relu(points)
        return points

    c_in = channels[0]
    c_init = 3
    # points = tf.ones_like(l0_xyz[:, :, :1])
    # points = tf.reduce_mean(offset00, [2])  # [B, M, K, D]
    # points = tf.reshape(points, [16, 1024, 3])
    points = l0_xyz
    fc0 = tf.get_variable(
        'w0', [c_init, c_in],
        initializer=tf.initializers.truncated_normal(
            0.0, 2.0 / np.sqrt(float(c_in))))
    points = _fc(points, fc0, bn=False)
    print(points)
    for layer in range(nlayers):
        with tf.variable_scope('l{}'.format(layer)):
            print(layer, len(alpha_list))
            alpha = alpha_list[layer]
            nn = nn_list[layer]
            c_out = channels[layer]
            points = _conv_block(points, nn, alpha, c_in, c_out)
            c_in = c_out

    # Fully connected layers
    l3_points = tf.reduce_max(points, [1])  # [B, C]
    # l3_points = tf.reduce_mean(points, [1])  # [B, C]
    net = tf.reshape(l3_points, [batch_size, channels[-1]])
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
