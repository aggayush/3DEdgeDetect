###############################
# Ekomkar
# Custom 3D canny edge filter layer
###############################

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras as tfk
import math


class MergePointCloud(tfk.layers.Layer):
    def __init__(self, outputSize, outputFunction='avg', **kwargs):
        super(MergePointCloud, self).__init__(**kwargs)
        self.outputSize = outputSize
        self.outputFunction = outputFunction

    def create_all_indexes(self, shape):
        # index arrays for each dimension
        voxelX = tf.range(shape[0])
        voxelY = tf.range(shape[1])
        voxelZ = tf.range(shape[2])

        # processing for creating all permutation and combinations
        voxelX = tf.expand_dims(voxelX, 1)
        voxelX = tf.tile(voxelX, [1, shape[1] * shape[2]])
        voxelX = tf.reshape(voxelX, [-1])

        voxelY = tf.expand_dims(voxelY, 0)
        voxelY = tf.expand_dims(voxelY, 2)
        voxelY = tf.tile(voxelY, [shape[0], 1, shape[2]])
        voxelY = tf.reshape(voxelY, [-1])

        voxelZ = tf.tile(voxelZ, [shape[0] * shape[1]])

        return voxelX, voxelY, voxelZ

    def transformGrid(self, voxelGrid):

        voxelGrid = tf.squeeze(voxelGrid, [-1])
        shape = tf.shape(voxelGrid)

        outSize = tf.convert_to_tensor(self.outputSize)
        tp = tf.expand_dims(shape[0], axis=0)
        outSize = tf.concat([tp, outSize], axis=0)
        outSize = tf.cast(outSize, tf.int64)

        voxelX, voxelY, voxelZ = self.create_all_indexes(shape[1:])

        voxelX = tf.cast(voxelX, tf.float64)
        voxelY = tf.cast(voxelY, tf.float64)
        voxelZ = tf.cast(voxelZ, tf.float64)

        # transforming given indexes to new grid indexes
        voxelX = tf.cast(tf.floor(tf.divide(tf.add(tf.multiply(voxelX, 2 * self.outputSize[0]), self.outputSize[0]),
                                            tf.cast(tf.multiply(2, shape[1]), tf.float64))), tf.int32)
        voxelY = tf.cast(tf.floor(tf.divide(tf.add(tf.multiply(voxelY, 2 * self.outputSize[1]), self.outputSize[1]),
                                            tf.cast(tf.multiply(2, shape[2]), tf.float64))), tf.int32)
        voxelZ = tf.cast(tf.floor(tf.divide(tf.add(tf.multiply(voxelZ, 2 * self.outputSize[2]), self.outputSize[2]),
                                            tf.cast(tf.multiply(2, shape[3]), tf.float64))), tf.int32)

        batchAxis = tf.range(shape[0])
        batchAxis = tf.expand_dims(batchAxis, axis=1)
        batchAxis = tf.tile(batchAxis, [1, tf.shape(voxelX)[0]])
        batchAxis = tf.cast(tf.reshape(batchAxis, [-1]), tf.int32)

        voxelX = tf.tile(voxelX, [shape[0]])
        voxelY = tf.tile(voxelY, [shape[0]])
        voxelZ = tf.tile(voxelZ, [shape[0]])

        newGrid = tf.cast(tf.stack([batchAxis, voxelX, voxelY, voxelZ], axis=1), tf.int64)
        newGridValues = tf.reshape(voxelGrid, [-1])

        delta = tf.sparse.SparseTensor(indices=newGrid, values=newGridValues, dense_shape=outSize)
        grid = tf.sparse.to_dense(delta, default_value=0.0, validate_indices=False)

        grid = tf.expand_dims(grid, axis=-1)

        return grid

    def functionMap(self):

        if self.outputFunction == 'max':
            return tf.reduce_max
        elif self.outputFunction == 'avg':
            return tf.reduce_mean
        elif self.outputFunction == 'min':
            return tf.reduce_min
        elif self.outputFunction == 'sum':
            return tf.reduce_sum
        else:
            raise tf.errors.InvalidArgumentError('Invalid function type given for layer MergePointCloud')

    def call(self, input):

        stackedGrid = None
        for grid in input:
            tempGrid = self.transformGrid(grid)
            if stackedGrid is None:
                stackedGrid = tempGrid
            else:
                stackedGrid = tf.concat([stackedGrid, tempGrid], axis=-1)

        fn = self.functionMap()

        finalVoxelGrid = fn(stackedGrid, axis=-1)
        finalVoxelGrid = tf.expand_dims(finalVoxelGrid, axis=-1)

        return finalVoxelGrid


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
        bias = tfk.initializers.Constant([0])
        self.convX = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelX,
                                       bias_initializer=bias, name='Sobel_ConvX')
        self.convY = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelY,
                                       bias_initializer=bias, name='Sobel_ConvY')
        self.convZ = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelZ,
                                       bias_initializer=bias, name='Sobel_ConvZ')

    def call(self, input):
        x = self.convX(input)
        y = self.convY(input)
        z = self.convZ(input)

        mag = tf.sqrt(tf.square(x) + tf.square(y) + tf.square(z))

        return mag


class WeightedLoss(tfk.losses.Loss):

    def __init__(self, pos_weight, weight, num_classes=2,
                 reduction=tfk.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super(WeightedLoss, self).__init__(reduction=reduction,
                                           name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.num_classes = num_classes

    def weighted_binary_cross_entropy(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1, self.num_classes])
        y_pred = tf.reshape(y_pred, [-1, self.num_classes])

        # Manually calculate the weighted cross entropy.
        # Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # where z are labels, x is logits, and q is the weight.
        # Since the values passed are from sigmoid (assuming in this case)
        # sigmoid(x) will be replaced by y_pred

        # qz * -log(sigmoid(x)) 1e-6 is added as an epsilon to stop passing a zero into the log
        x_1 = y_true * self.pos_weight * -tf.math.log(y_pred + 1e-6)

        # (1 - z) * -log(1 - sigmoid(x)). Epsilon is added to prevent passing a zero into the log
        x_2 = (1 - y_true) * -tf.math.log(1 - y_pred + 1e-6)

        return tf.add(x_1, x_2) * self.weight

    def softmax_cross_entropy(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1, self.num_classes])
        y_pred = tf.reshape(y_pred, [-1, self.num_classes])

        return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

    def weighted_mean_square_error(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1,])
        y_pred = tf.reshape(y_pred, [-1,])

        pos_idx = tf.where(y_true == 1.0)
        neg_idx = tf.where(y_true == 0.0)

        pos_idx = tf.reshape(pos_idx, [-1,])
        neg_idx = tf.reshape(neg_idx, [-1,])

        pos_mse = tf.reduce_mean(tf.multiply(tf.square(tf.gather(y_true, pos_idx) - tf.gather(y_pred, pos_idx)), self.pos_weight))
        neg_mse = tf.reduce_mean(tf.square(tf.gather(y_true, neg_idx) - tf.gather(y_pred, neg_idx))) * self.weight

        return tf.add(pos_mse, neg_mse)

    def call(self, y_true, y_pred):
        # return self.weighted_binary_cross_entropy(y_true, y_pred)
        # return self.softmax_cross_entropy(y_true, y_pred)
        return self.weighted_mean_square_error(y_true, y_pred)
