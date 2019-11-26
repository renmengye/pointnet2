import tensorflow as tf
from tensorflow.python.framework import ops

cont_filter_module = tf.load_op_library(
    'tf_ops/cont_filter/tf_cont_filter_conv_so.so')


@ops.RegisterGradient("ContFilterConv")
def _cont_filter_grad(op, grad):
    inp = op.inputs[0]
    neighbor = op.inputs[1]
    alpha = op.inputs[2]
    base_weight = op.inputs[3]
    grad_inp, grad_base_weight = cont_filter_module.cont_filter_conv_grad(
        inp, neighbor, alpha, base_weight, grad)
    return [grad_inp, None, None, grad_base_weight]


def cont_filter_interp(dist, cube_size, num_anchors):
    """Compute interpolation weights.

    :param offset:       [B, M, K, 3]      Neighbor offsets.
    :param cube_size:    [float]           Size of the cube.
    :param num_anchors:  [int]             Default 27, R.

    :return              [B, R, M, K]      Filter interpolation weight.
    """
    assert num_anchors == 27
    return cont_filter_module.cont_filter_conv_interp(dist, cube_size,
                                                      num_anchors)


def cont_filter_conv(inp, neighbor, alpha, base_weight, transpose=True):
    """Continuous filter convolution.

    B: Batch size
    C: Features
    N: Num input points
    M: Num output points
    R: Num anchors
    K: Num neighbors

    :param inp:          [B, N, C]      Input features.
    :param neighbor:     [B, M, K]      Neighbor ID.
    :param alpha:        [B, R, M, K]   Interpolation weights.
    :param base_weight:  [R, C]         Filter weights.
    :return              [B, M, C]
    """
    if transpose:
        inp = tf.transpose(inp, [0, 2, 1])
    # inp = tf.Print(inp, [inp.name, tf.sqrt(tf.reduce_sum(tf.square(inp)))])
    result = cont_filter_module.cont_filter_conv(inp, neighbor, alpha,
                                                 base_weight)
    if transpose:
        result = tf.transpose(result, [0, 2, 1])
    return result


def get_knn(in_xyz, out_xyz, k=10):
    """
    :param in_xyz:     [B, N, 3]
    :param out_xyz:    [B, M, 3]
    :return            [B, M, K, 3]
    """
    from sklearn.neighbors import NearestNeighbors
    offset = np.zeros(
        [in_xyz.shape[0], out_xyz.shape[1], k, 3], dtype=in_xyz.dtype)
    neighbors = np.zeros(
        [in_xyz.shape[0], out_xyz.shape[1], k], dtype=np.int64)
    for b in range(in_xyz.shape[0]):
        _in_xyz = in_xyz[b]
        _out_xyz = out_xyz[b]
        # print('in xyz', _in_xyz.shape)
        # print('out xyz', _out_xyz.shape)
        nbrs = NearestNeighbors(
            n_neighbors=k, algorithm='ball_tree').fit(_in_xyz)
        _, idx = nbrs.kneighbors(_out_xyz)  # [N, K]
        offset[b] = _in_xyz[idx] - _out_xyz[:, np.newaxis, :]
        neighbors[b] = idx
    return offset, neighbors


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)
    with tf.Session() as sess:
        xyz = tf.constant(np.random.uniform(-1.0, 1.0, [2, 10, 3]), tf.float32)
        xyz2 = tf.constant(np.random.uniform(-1.0, 1.0, [2, 5, 3]), tf.float32)
        offset, neighbor = tf.py_func(get_knn, [xyz, xyz2],
                                      [tf.float32, tf.int64])
        # offset = tf.Print(offset, ['offset', tf.shape(offset)], summarize=100)
        # offset = tf.Print(
        #     offset, ['neighbor', tf.shape(neighbor)], summarize=100)
        alpha = cont_filter_interp(offset, 1.0, 27)
        inp = tf.constant(
            np.random.uniform(-1.0, 1.0, [2, 10, 3]), dtype=tf.float32)
        base_weight = tf.constant(
            np.random.uniform(-1.0, 1.0, [27, 3]), dtype=tf.float32)
        # base_weight = tf.Print(
        #     base_weight, ['base', tf.shape(base_weight)], summarize=100)
        # base_weight = tf.Print(
        #     base_weight, ['alpha', tf.shape(alpha)], summarize=100)
        y = cont_filter_conv(inp, neighbor, alpha, base_weight, transpose=True)
        dx = tf.gradients(y, base_weight)
        print(sess.run(y))
        print(sess.run(dx))
        print('hehe')
        alpha = sess.run(alpha[0, :, 0, :]).T
        print(alpha)
        print(alpha.sum(axis=1))
        t = tf.test.compute_gradient_error([inp, base_weight], [(2, 10, 3),
                                                                (27, 3)], y,
                                           y.eval().shape)
        print(t)
