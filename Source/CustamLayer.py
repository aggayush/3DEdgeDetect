###############################
# Custom 3D canny edge filter layer
###############################

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.keras as tfk

class WeightedBinaryCrossEntropy(tfk.losses.Loss):


class SobelFilter(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super(SobelFilter, self).__init__(**kwargs)
        self.kernelX = tfk.initializers.constant([
            [
                [-1., -2., -1.],
                [-2., -4., -2.],
                [-1., -2., -1.]
            ],
            [
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]
            ],
            [
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.]
            ]
        ])
        self.kernelY = tfk.initializers.constant([
            [
                [-1., -2., -1.],
                [0., 0., 0.],
                [1., 2., 1.]
            ],
            [
                [-2., -4., -2.],
                [0., 0., 0.],
                [2., 4., 2.]
            ],
            [
                [-1., -2., -1.],
                [0., 0., 0.],
                [1., 2., 1.]
            ]
        ])
        self.kernelZ = tfk.initializers.constant([
            [
                [-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., -1.]
            ],
            [
                [-2., 0., 2.],
                [-4., 0., 4.],
                [-2., 0., 2.]
            ],
            [
                [-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]
            ]
        ])
        self.convX = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelX)
        self.convY = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelY)
        self.convZ = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelZ)

    def call(self, input):

        x = self.convX(input)
        y = self.convY(input)
        z = self.convZ(input)

        mag = tf.sqrt(tf.square(x) + tf.square(y) + tf.square(z))

        return mag


