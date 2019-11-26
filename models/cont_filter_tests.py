import numpy as np
import tensorflow as tf
import unittest

from models.cont_filter import cont_filter_conv, cont_filter_interp


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


class ContFilterTests(unittest.TestCase):
    def test_interp_corner(self):
        with tf.Session() as sess:
            # [B, M, K, D] -> [B, R, M, K]
            offset = np.array([
                [0.0, 0.0, 1.0],  #
                [0.0, 1.0, 0.0],  #
                [1.0, 0.0, 0.0],  #
                [0.0, 0.0, -1.0],  #
                [0.0, -1.0, 0.0],  #
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, -1.0],
                [1.0, -1.0, 0.0],
                [1.0, 0.0, -1.0],
                [0.0, -1.0, 1.0],
                [-1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0],
                [0.0, -1.0, -1.0],
                [-1.0, -1.0, 0.0],
                [-1.0, 0.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0]
            ]).astype(np.float32)
            offset = np.expand_dims(np.expand_dims(offset, 0), 0)
            offset = tf.constant(offset)
            alpha = cont_filter_interp(offset, 1.0, 27)
            alpha_val = sess.run(alpha)
            alpha_val = alpha_val[0, :, 0, :]  # [27, 27]
            alpha_val = alpha_val.T  # [M, R]
            np.testing.assert_allclose(
                alpha_val.sum(axis=1), np.ones(27), rtol=1e-5)
            np.testing.assert_allclose(
                alpha_val.sum(axis=0), np.ones(27), rtol=1e-5)

    def test_interp_decay(self):
        with tf.Session() as sess:
            # [B, M, K, D] -> [B, R, M, K]
            offset = np.array([
                [0.0, 0.0, 1.2],  #
                [0.0, 1.2, 0.0],  #
                [1.2, 0.0, 0.0],  #
                [0.0, 0.0, -1.2],  #
                [0.0, -1.2, 0.0],  #
                [-1.2, 0.0, 0.0],
                [0.0, 1.2, 1.2],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.2],
                [0.0, 1.2, -1.2],
                [1.2, -1.2, 0.0],
                [1.2, 0.0, -1.2],
                [0.0, -1.2, 1.2],
                [-1.2, 1.2, 0.0],
                [-1.2, 0.0, 1.2],
                [0.0, -1.2, -1.2],
                [-1.2, -1.2, 0.0],
                [-1.2, 0.0, -1.2],
                [-1.2, -1.2, -1.2],
                [-1.2, -1.2, 1.2],
                [-1.2, 1.2, -1.2],
                [1.2, -1.2, -1.2],
                [1.2, 1.2, -1.2],
                [1.2, -1.2, 1.2],
                [-1.2, 1.2, 1.2],
                [1.2, 1.2, 1.2],
                [0.0, 0.0, 0.0]
            ]).astype(np.float32)
            decay = np.copy(offset)
            decay[np.where(np.abs(offset) == 1.2)] = 0.8
            decay[np.where(np.abs(offset) == 0.0)] = 1.0
            decay = np.prod(decay, axis=-1)  # [27]
            offset = np.expand_dims(np.expand_dims(offset, 0), 0)
            offset = tf.constant(offset)
            alpha = cont_filter_interp(offset, 1.0, 27)
            alpha_val = sess.run(alpha)
            alpha_val = alpha_val[0, :, 0, :]  # [27, 27]
            alpha_val = alpha_val.T  # [M, R]
            np.testing.assert_allclose(alpha_val.sum(axis=1), decay, rtol=1e-5)

    def test_interp_basic(self):
        with tf.Session() as sess:
            offset = np.array([[-0.3, -0.4, -0.2]]).astype(np.float32)
            offset = np.expand_dims(np.expand_dims(offset, 0), 0)
            offset = tf.constant(offset)
            alpha = cont_filter_interp(offset, 1.0, 27)
            alpha_val = sess.run(alpha)
            alpha_val = alpha_val[0, :, 0, 0]  # [27, 1]
            exp = np.zeros([27])
            exp[0] = 0.3 * 0.4 * 0.2
            exp[1] = 0.7 * 0.4 * 0.2
            exp[3] = 0.6 * 0.3 * 0.2
            exp[4] = 0.6 * 0.7 * 0.2
            exp[9] = 0.3 * 0.4 * 0.8
            exp[10] = 0.7 * 0.4 * 0.8
            exp[12] = 0.6 * 0.3 * 0.8
            exp[13] = 0.6 * 0.7 * 0.8
            np.testing.assert_allclose(alpha_val, exp, rtol=1e-5)

    def test_sum(self):
        np.random.seed(0)
        B = 2
        D = 3
        for K in range(4, 10, 2):
            for N in range(K, K * 2):
                for use_same_inp in [True, False]:
                    with tf.Session() as sess:
                        xyz = tf.constant(
                            np.random.uniform(-1.0, 1.0, [B, N, 3]),
                            tf.float32)
                        if use_same_inp:
                            M = N
                            xyz2 = xyz
                        else:
                            M = N // 2
                            xyz2 = tf.constant(
                                np.random.uniform(-1.0, 1.0, [B, M, 3]),
                                tf.float32)
                        offset, neighbor = tf.py_func(
                            lambda x, y: get_knn(x, y, k=K), [xyz, xyz2],
                            [tf.float32, tf.int64])  # [B, M, K, 3] [B, M, K]
                        inp = tf.constant(
                            np.random.uniform(-1.0, 1.0, [B, N, D]),
                            dtype=tf.float32)
                        base_weight = tf.constant(
                            np.ones([27, D]), dtype=tf.float32)
                        alpha = tf.concat(
                            [
                                tf.ones_like(alpha)[:, :1, :, :],
                                tf.zeros_like(alpha)[:, :-1, :, :]
                            ],
                            axis=1)  # [B, R, M, K]
                        x = inp
                        y = cont_filter_conv(
                            x, neighbor, alpha, base_weight,
                            transpose=True)  # [B, M, C]
                        x_gather = [
                            tf.gather(x[ii], tf.reshape(neighbor[ii], [-1]))
                            for ii in range(B)
                        ]
                        x_gather = tf.reshape(tf.stack(x_gather), [B, M, K, D])
                        y2 = tf.reduce_sum(x_gather, [2])
                        y_val, y2_val, nn_val = sess.run([y, y2, neighbor])
                        np.testing.assert_allclose(y_val, y2_val)

                        dx = tf.gradients(y, x)
                        dx2 = tf.gradients(y2, x)
                        dx_val, dx2_val = sess.run([dx, dx2])
                        np.testing.assert_allclose(dx_val, dx2_val)

    def test_grad(self):
        np.random.seed(0)
        B = 2
        D = 3
        for K in range(4, 10, 2):
            for N in range(K, K * 2):
                for use_same_inp in [True, False]:
                    with tf.Session() as sess:
                        xyz = tf.constant(
                            np.random.uniform(-1.0, 1.0, [B, N, D]),
                            tf.float32)
                        if use_same_inp:
                            xyz2 = xyz
                        else:
                            M = N // 2
                            xyz2 = tf.constant(
                                np.random.uniform(-1.0, 1.0, [B, M, D]),
                                tf.float32)
                        offset, neighbor = tf.py_func(
                            lambda x, y: get_knn(x, y, k=K), [xyz, xyz2],
                            [tf.float32, tf.int64])
                        alpha = cont_filter_interp(offset, 1.0, 27)
                        inp = tf.constant(
                            np.random.uniform(-1.0, 1.0, [B, N, D]),
                            dtype=tf.float32)
                        base_weight = tf.constant(
                            np.random.uniform(-1.0, 1.0, [27, D]),
                            dtype=tf.float32)
                        y = cont_filter_conv(
                            inp, neighbor, alpha, base_weight, transpose=True)
                        dx = tf.gradients(y, base_weight)
                        alpha = sess.run(alpha[0, :, 0, :]).T
                        t = tf.test.compute_gradient_error(
                            [inp, base_weight], [(B, N, D), (27, D)], y,
                            y.eval().shape)
                        assert t < 1e-3, 'gradient check failed {}'.format(t)


if __name__ == '__main__':
    unittest.main()
