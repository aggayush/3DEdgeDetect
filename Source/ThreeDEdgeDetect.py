################################
# Main class file to execute the code
################################\

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import tensorflow.keras as tfk
from Source.CustamLayer import SobelFilter
from Source.utilities import path_reader, visualize


class ThreeDEdgeDetector:
    VOXEL_GRID_X = 256
    VOXEL_GRID_Y = 256
    VOXEL_GRID_Z = 256
    DATA_DEFAULTS = [[0.], [0.], [0.]]

    def __init__(self,args=None):
        if args is None:
            self.trainDataPath = ""
            self.testDataPath = ""
            self.outputPath = ""
            self.modelPath = ""
            self.isTrain = True
            self.isStreamed = False
            self.batchSize = 1
            self.shuffleBufferSize = 1000
            self.activation = 'relu'
            self.dropoutRate = 0.01
        else:
            self.trainDataPath = args.trainDataPath
            self.testDataPath = args.testDataPath
            self.outputPath = args.outputPath
            self.modelPath = args.modelPath
            self.isTrain = args.isTrain
            self.isStreamed = args.isStreamed
            self.batchSize = args.batchSize
            self.shuffleBufferSize = args.shuffleBufferSize
            self.activation = args.activation
            self.dropoutRate = args.dropoutRate
        self.prefetchBufferSize=10
        self.trainDataset = None
        self.testDataset = None
        self.model = None

    def process_data_files(self, filePath):

        grid = tf.zeros([self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z],
                        tf.bool)

        rawData = tf.io.read_file(filePath)
        records = tf.strings.split(rawData, sep='\r\n')
        data = tf.io.decode_csv(records,
                               self.DATA_DEFAULTS,
                               field_delim=' ')
        # Normalization to unit sphere
        data = tf.transpose(data)
        centroid = tf.math.reduce_mean(data, axis=0)
        data = data - centroid
        furthestDistanceFromCentre = tf.math.reduce_max(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.abs(data)), axis=-1)))
        data = tf.math.divide(data, furthestDistanceFromCentre)

        #Voxelization of data
        gridX = tf.floor(tf.add(tf.multiply(data[:, 0], (self.VOXEL_GRID_X/2)), self.VOXEL_GRID_X/2))
        gridY = tf.floor(tf.add(tf.multiply(data[:, 1], (self.VOXEL_GRID_Y/2)), self.VOXEL_GRID_Y/2))
        gridZ = tf.floor(tf.add(tf.multiply(data[:, 2], (self.VOXEL_GRID_Z/2)), self.VOXEL_GRID_Z/2))

        indexes = tf.stack([gridX, gridY, gridZ], axis=1)
        indexes = tf.cast(indexes, tf.int64)
        shape = tf.shape(indexes)
        sparseValues = tf.ones([shape[0]], tf.bool)
        delta = tf.sparse.SparseTensor(indices=indexes, values=sparseValues, dense_shape=[self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z])
        grid = tf.math.logical_or(grid, tf.sparse.to_dense(delta, default_value=False, validate_indices=False))

        return grid, centroid, furthestDistanceFromCentre

    def read_data(self):

        trainPath = os.path.abspath(self.trainDataPath)
        testPath = os.path.abspath(self.testDataPath)

        # # Display the files in path
        # trainFiles = path_reader(trainPath)
        # testFiles = path_reader(testPath)
        # print(trainFiles)
        # print(testFiles)

        trainFilesDs = tf.data.Dataset.list_files(os.path.join(trainPath, '*.txt'))
        testFilesDs = tf.data.Dataset.list_files(os.path.join(testPath, '*.txt'))

        self.trainDataset = trainFilesDs.map(lambda filePath: self.process_data_files(filePath))
        self.trainDataset = self.trainDataset.shuffle(self.shuffleBufferSize)
        self.trainDataset = self.trainDataset.batch(self.batchSize)
        self.trainDataset = self.trainDataset.prefetch(self.prefetchBufferSize)

        self.testDataset = testFilesDs.map(lambda filePath: self.process_data_files(filePath))
        self.testDataset = self.testDataset.shuffle(self.shuffleBufferSize)
        self.testDataset = self.testDataset.batch(self.batchSize)
        self.testDataset = self.testDataset.prefetch(self.prefetchBufferSize)

        # # to see filename and visualize data
        # for f in trainFilesDs.take(1):
        #     print(f.numpy())
        #     visualize(f.numpy().decode("utf-8"))
        # for f in testFilesDs.take(1):
        #     print(f.numpy())
        #     visualize(f.numpy().decode("utf-8"))

    def create_model(self):

        self.model = tfk.models.Sequential()

        self.model.add(tfk.layers.Conv3D(32,
                                        (3, 3, 3),
                                        strides=(2, 2, 2),
                                        padding='same',
                                        activation=self.activation,
                                        input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                        kernel_initializer=tf.initializers.glorot_normal))
        self.model.add(tfk.layers.BatchNormalization())
        self.model.add(tfk.layers.Dropout(self.dropoutRate))
        self.model.add(tfk.layers.MaxPool3D((2, 2, 2),
                                            padding='same'))
        self.model.add(tfk.layers.Conv3D(1,
                                        (3, 3, 3),
                                        strides=(1, 1, 1),
                                        padding='same',
                                        activation=self.activation,
                                        input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                        kernel_initializer=tf.initializers.glorot_normal))
        self.model.add(tfk.layers.BatchNormalization())
        self.model.add(SobelFilter(name='output_1', trainable=False))
        self.model.add(tfk.layers.BatchNormalization())

        self.model.summary()

    def run(self):

        # try:
        #     self.read_data()
        # except:
        #     print('Data read Error')
        #     exit(1)

        self.create_model()



