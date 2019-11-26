import numpy as np
import tensorflow as tf
import unittest

from utils.pointnet_util import sample_and_group


class GroupingSamplingTests(unittest.TestCase):
    def test_basic(self):
        """Test to see if all points are sampled."""
        N = 1024
        np.random.seed(0)

        with tf.Session() as sess:
            xyz = np.random.uniform(-1.0, 1.0, [1, 1024, 3])
            xyz2, _, nn, offset, new_idx = sample_and_group(
                npoint=N,  # same as inputs
                radius=0.2,
                nsample=32,
                xyz=xyz,
                points=None,
                knn=False,
                use_xyz=True,
                return_new=True)
            new_idx_val = sess.run(new_idx)
            result = [False] * N
            for ii in np.reshape(new_idx_val, [-1]):
                result[ii] = True
            assert all(result)


if __name__ == '__main__':
    unittest.main()
