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
        self.digits = [int(math.log10(self.outputSize[0]))+1, int(math.log10(self.outputSize[1]))+1, int(math.log10(self.outputSize[2]))+1]
        idxX, idxY, idxZ = self.create_all_indexes(self.outputSize)
        self.outputIndexs = tf.stack([idxX, idxY, idxZ], axis=1)
        self.outputIndexHashes = tf.add(tf.add(tf.multiply(idxX, tf.pow(10, self.digits[1] + self.digits[2])), tf.multiply(idxY, tf.pow(10, self.digits[2]))), idxZ)

    def create_all_indexes(self, shape):
        # index arrays for each dimension
        voxelX = tf.range(shape[0])
        voxelY = tf.range(shape[1])
        voxelZ = tf.range(shape[2])

        # processing for creating all permutation and combinations
        voxelX = tf.expand_dims(voxelX, 1)
        voxelX = tf.tile(voxelX, [1, shape[1]*shape[2]])
        voxelX = tf.reshape(voxelX, [-1])

        voxelY = tf.expand_dims(voxelY, 0)
        voxelY = tf.expand_dims(voxelY, 2)
        voxelY = tf.tile(voxelY, [shape[0], 1, shape[2]])
        voxelY = tf.reshape(voxelY, [-1])

        voxelZ = tf.tile(voxelZ, [shape[0]*shape[1]])

        return voxelX, voxelY, voxelZ

    def voxel_to_point(self, voxelGrid):
        voxelGrid = tf.squeeze(voxelGrid, [-1])
        shape = tf.shape(voxelGrid)

        voxelX, voxelY, voxelZ = self.create_all_indexes(shape[1:])

        voxelX = tf.cast(voxelX, tf.float64)
        voxelY = tf.cast(voxelY, tf.float64)
        voxelZ = tf.cast(voxelZ, tf.float64)

        # converting index to points
        pointCloudX = tf.add(tf.divide(tf.subtract(voxelX, shape[1]/2), shape[1]/2), 1/shape[1])
        pointCloudY = tf.add(tf.divide(tf.subtract(voxelY, shape[2]/2), shape[2]/2), 1/shape[2])
        pointCloudZ = tf.add(tf.divide(tf.subtract(voxelZ, shape[3]/2), shape[3]/2), 1/shape[3])

        pointCloud = tf.stack([pointCloudX, pointCloudY, pointCloudZ], axis=1)
        # pointCloud = tf.expand_dims(pointCloud, 0)
        # pointCloud = tf.tile(pointCloud, [shape[0], 1, 1])
        pointCloudValues = tf.reshape(voxelGrid, [shape[0], -1])

        return pointCloud, pointCloudValues

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

    def point_to_voxel(self, pointCloud, pointCloudValues):

        fn = self.functionMap()
        voxelGrid = None
        gridX = tf.cast(tf.floor(tf.add(tf.multiply(pointCloud[:, 0], (self.outputSize[0] / 2)), self.outputSize[0] / 2)), tf.int32)
        gridY = tf.cast(tf.floor(tf.add(tf.multiply(pointCloud[:, 1], (self.outputSize[1] / 2)), self.outputSize[1] / 2)), tf.int32)
        gridZ = tf.cast(tf.floor(tf.add(tf.multiply(pointCloud[:, 2], (self.outputSize[2] / 2)), self.outputSize[2] / 2)), tf.int32)

        indexHashes = tf.add(tf.add(tf.multiply(gridX, tf.pow(10, self.digits[1] + self.digits[2])), tf.multiply(gridY, tf.pow(10, self.digits[2]))), gridZ)
        sparseValues = None

        for uniqueHash in tf.unstack(self.outputIndexHashes):
            idx = tf.where(tf.equal(indexHashes, uniqueHash))
            idx = tf.reshape(idx, [-1])
            temp = fn(tf.gather(pointCloudValues, idx, axis=1), axis=1)
            temp = tf.expand_dims(temp, axis=-1)
            if sparseValues is None:
                sparseValues = temp
            else:
                sparseValues = tf.concat([sparseValues, temp], axis=1)

        indexes = tf.cast(self.outputIndexs, tf.int64)
        indexes = tf.tile(indexes, [2, 1])
        batchAxis = tf.range(tf.shape(sparseValues)[0])
        batchAxis = tf.expand_dims(batchAxis, axis=1)
        batchAxis = tf.tile(batchAxis, [1, tf.shape(self.outputIndexHashes)[0]])
        batchAxis = tf.cast(tf.reshape(batchAxis, [-1]), tf.int64)
        batchAxis = tf.expand_dims(batchAxis, axis=1)
        indexes = tf.concat([batchAxis, indexes], axis=1)
        sparseValues = tf.reshape(sparseValues, [-1])
        outSize = tf.convert_to_tensor(self.outputSize)
        tp = tf.shape(pointCloudValues)[0]
        tp = tf.expand_dims(tp, axis=0)
        outSize = tf.concat([tp, outSize], axis=0)
        outSize = tf.cast(outSize, tf.int64)
        delta = tf.sparse.SparseTensor(indices=indexes, values=sparseValues, dense_shape=outSize)
        voxelGrid = tf.sparse.to_dense(delta, default_value=0.0, validate_indices=False)

        voxelGrid = tf.expand_dims(voxelGrid, -1)

        return voxelGrid

    def call(self, input):

        stackedPoints = None
        stackedPointValues = None
        for grid in input:
            pointCloud, pointCloudValues = self.voxel_to_point(grid)
            if stackedPoints is None:
                stackedPoints = pointCloud
                stackedPointValues = pointCloudValues
            else:
                stackedPoints = tf.concat([stackedPoints, pointCloud], axis=0)
                stackedPointValues = tf.concat([stackedPointValues, pointCloudValues], axis=1)

        finalVoxelGrid = self.point_to_voxel(stackedPoints, stackedPointValues)

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
        self.convX = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelX)
        self.convY = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelY)
        self.convZ = tfk.layers.Conv3D(1, (3, 3, 3), padding='same', kernel_initializer=self.kernelZ)

    def call(self, input):

        x = self.convX(input)
        y = self.convY(input)
        z = self.convZ(input)

        mag = tf.sqrt(tf.square(x) + tf.square(y) + tf.square(z))

        return mag


