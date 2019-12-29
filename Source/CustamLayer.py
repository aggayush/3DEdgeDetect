###############################
# Ekomkar
# Custom 3D canny edge filter layer
###############################

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np


class MergePointCloud(tfk.layers.Layer):
    def __init__(self, outputSize, outputFunction='avg', normalize=True, **kwargs):
        super(MergePointCloud, self).__init__(**kwargs)
        self.outputSize = outputSize
        self.outputFunction = outputFunction
        self.norm = normalize

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

        # voxelGrid = tf.squeeze(voxelGrid, [-1])
        shape = tf.shape(voxelGrid)

        outSize = tf.convert_to_tensor(self.outputSize)
        tp = tf.expand_dims(shape[0], axis=0)
        fp = tf.expand_dims(shape[-1], axis=0)
        outSize = tf.concat([tp, outSize, fp], axis=0)
        outSize = tf.cast(outSize, tf.int64)

        voxelX, voxelY, voxelZ = self.create_all_indexes(shape[1:-1])

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

        feat = tf.range(shape[-1])
        feat = tf.tile(feat, [tf.shape(voxelX)[0]])

        batchAxis = tf.expand_dims(batchAxis, axis=1)
        batchAxis = tf.tile(batchAxis, [1, shape[-1]])
        batchAxis = tf.reshape(batchAxis, [-1])
        voxelX = tf.expand_dims(voxelX, axis=1)
        voxelX = tf.tile(voxelX, [1, shape[-1]])
        voxelX = tf.reshape(voxelX, [-1])
        voxelY = tf.expand_dims(voxelY, axis=1)
        voxelY = tf.tile(voxelY, [1, shape[-1]])
        voxelY = tf.reshape(voxelY, [-1])
        voxelZ = tf.expand_dims(voxelZ, axis=1)
        voxelZ = tf.tile(voxelZ, [1, shape[-1]])
        voxelZ = tf.reshape(voxelZ, [-1])

        newGrid = tf.cast(tf.stack([batchAxis, voxelX, voxelY, voxelZ, feat], axis=1), tf.int64)
        newGridValues = tf.reshape(voxelGrid, [-1])

        delta = tf.sparse.SparseTensor(indices=newGrid, values=newGridValues, dense_shape=outSize)
        grid = tf.sparse.to_dense(delta, default_value=0.0, validate_indices=False)

        grid = tf.expand_dims(grid, axis=-1)

        del batchAxis
        del voxelX
        del voxelY
        del voxelZ
        del feat
        del shape
        del outSize
        del newGrid
        del newGridValues
        del delta

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

    def normalize(self, grid):
        grid_shape = tf.shape(grid)
        reshaped_grid = tf.reshape(grid, (grid_shape[0], -1))
        max = tf.expand_dims(
            tf.tile(
                tf.expand_dims(
                    tf.tile(
                        tf.expand_dims(
                            tf.tile(
                                tf.reduce_max(reshaped_grid, axis=1, keepdims=True),
                                [1, grid_shape[1]]),
                            axis=-1),
                        [1, 1, grid_shape[2]]),
                    axis=-1),
                [1, 1, 1, grid_shape[3]]),
            axis=-1)
        min = tf.expand_dims(
            tf.tile(
                tf.expand_dims(
                    tf.tile(
                        tf.expand_dims(
                            tf.tile(
                                tf.reduce_min(reshaped_grid, axis=1, keepdims=True),
                                [1, grid_shape[1]]),
                            axis=-1),
                        [1, 1, grid_shape[2]]),
                    axis=-1),
                [1, 1, 1, grid_shape[3]]),
            axis=-1)
        diff = 2.0/(max-min)
        grid = diff * grid - 1.0


        del reshaped_grid
        del min
        del max
        del diff

        return grid

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
        if self.norm:
            finalVoxelGrid = self.normalize(finalVoxelGrid)

        del stackedGrid

        return finalVoxelGrid


class SobelFilter(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super(SobelFilter, self).__init__(**kwargs)
        self.kernelX = tfk.initializers.constant([
            [
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.]
            ],
            [
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]
            ],
            [
                [-1., -2., -1.],
                [-2., -4., -2.],
                [-1., -2., -1.]
            ]
        ])
        self.kernelY = tfk.initializers.constant([
            [
                [1., 2., 1.],
                [0., 0., 0.],
                [-1., -2., -1.]
            ],
            [
                [2., 4., 2.],
                [0., 0., 0.],
                [-2., -4., -2.]
            ],
            [
                [1., 2., 1.],
                [0., 0., 0.],
                [-1., -2., -1.]
            ]
        ])
        self.kernelZ = tfk.initializers.constant([
            [
                [1., 0., -1.],
                [2., 0., -2.],
                [1., 0., -1.]
            ],
            [
                [2., 0., -2.],
                [4., 0., -4.],
                [2., 0., -2.]
            ],
            [
                [1., 0., -1.],
                [2., 0., -2.],
                [1., 0., -1.]
            ]
        ])
        self.convX = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelX,
                                       use_bias=False, name='Sobel_ConvX')
        self.convY = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelY,
                                       use_bias=False, name='Sobel_ConvY')
        self.convZ = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelZ,
                                       use_bias=False, name='Sobel_ConvZ')

    def call(self, input):
        x = tf.divide(self.convX(input), 16.0)
        y = tf.divide(self.convY(input), 16.0)
        z = tf.divide(self.convZ(input), 16.0)

        ang_xy = tf.atan(y/(x+1e-5))
        ang_xz = tf.atan(z/(x+1e-5))
        ang_yz = tf.atan(z/(y+1e-5))

        mag = tf.sqrt(tf.add_n([tf.square(x), tf.square(y), tf.square(z)]))

        # return tf.concat([mag, ang_xy, ang_xz, ang_yz], axis=-1)
        return mag


class GMMClusteringLayer(tfk.layers.Layer):

    def __init__(self, nClusters=2, means=None, variance=None, weights=None, **kwargs):
        super(GMMClusteringLayer, self).__init__(**kwargs)
        self.nClusters = nClusters
        if weights is None:
            self.initWeights = tfk.initializers.constant(np.ones(shape=[self.nClusters], dtype=float)* (1./self.nClusters))

        if means is None:
            self.initMeans = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        if variance is None:
            self.initVariance = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

    def build(self, input_shape):
        val = np.log(2 * np.pi) * input_shape[-1]
        self.ln2piD = tf.constant(val, dtype=tf.float32)
        self.means = self.add_weight(shape=(self.nClusters, input_shape[-1]), initializer=self.initMeans, name='gmm_mean')
        self.variance = self.add_weight(shape=(self.nClusters, input_shape[-1]), initializer=self.initVariance, name='gmm_variance')
        self.clusWeights = self.add_weight(shape=(self.nClusters,), initializer=self.initWeights)
        self.built = True

    def call(self, inputs):
        distances = tf.math.squared_difference(tf.expand_dims(inputs, -2), self.means)
        dist_times_inv_var = tf.reduce_sum(distances / self.variance, -1)
        log_coefficients = tf.add(self.ln2piD, tf.reduce_sum(tf.math.log(self.variance), 1))
        log_components = -0.5 * (log_coefficients + dist_times_inv_var)
        log_weighted = log_components + tf.math.log(self.clusWeights)
        log_shift = tf.expand_dims(tf.reduce_max(log_weighted, -1), -1)
        exp_log_shifted = tf.exp(log_weighted - log_shift)
        exp_log_shifted_sum = tf.reduce_sum(exp_log_shifted, -1)
        gamma = exp_log_shifted / tf.expand_dims(exp_log_shifted_sum, -1)

        return gamma


class KMeansClusteringLayer(tfk.layers.Layer):
    def __init__(self, nClusters=2, weights=None, alpha=1.0, **kwargs):
        super(KMeansClusteringLayer, self).__init__(**kwargs)
        self.nClusters = nClusters
        self.initWeights = weights
        self.alpha = alpha

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.clusters = self.add_weight(shape=(self.nClusters, feat_dim), initializer=tfk.initializers.glorot_uniform, name='Clusters')
        if self.initWeights is not None:
            self.set_weights(self.initWeights)
            del self.initWeights
        self.built = True

    def call(self, inputs):
        dist = tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(inputs, axis=-2), self.clusters)), axis=-1)
        pdf = 1.0 / (1.0 + dist/self.alpha)
        pdf = tf.pow(pdf, ((self.alpha + 1.0)/2.0))
        pdf = tf.divide(pdf, tf.expand_dims(tf.reduce_sum(pdf, axis=-1), axis=-1))

        return pdf


class MinPooling3D(tfk.layers.Layer):

    def __init__(self, pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', **kwargs):
        super(MinPooling3D, self).__init__(**kwargs)
        self.poolSize = pool_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.maxPool = tfk.layers.MaxPool3D(pool_size=self.poolSize,
                                            strides=self.strides,
                                            padding=self.padding)
        self.build = True

    def call(self, inputs):
        result = tf.multiply(inputs, -1.0)
        result = self.maxPool(result)
        result = tf.multiply(result, -1.0)

        return result


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

        pos_idx = tf.reshape(pos_idx, [-1, ])
        neg_idx = tf.reshape(neg_idx, [-1, ])

        pos_mse = tf.reduce_mean(tf.multiply(tf.square(tf.gather(y_true, pos_idx) - tf.gather(y_pred, pos_idx)), self.pos_weight))
        neg_mse = tf.reduce_mean(tf.square(tf.gather(y_true, neg_idx) - tf.gather(y_pred, neg_idx))) * self.weight

        return tf.add(pos_mse, neg_mse)

    def call(self, y_true, y_pred):
        # return self.weighted_binary_cross_entropy(y_true, y_pred)
        # return self.softmax_cross_entropy(y_true, y_pred)
        return self.weighted_mean_square_error(y_true, y_pred)
