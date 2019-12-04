################################
# Main class file to execute the code
################################\

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import tensorflow.keras as tfk
import math
import glob
from Source.CustamLayer import SobelFilter
from Source.utilities import path_reader, visualize


class ThreeDEdgeDetector:
    VOXEL_GRID_X = 256
    VOXEL_GRID_Y = 256
    VOXEL_GRID_Z = 256
    DATA_DEFAULTS = [[0.], [0.], [0.]]

    def __init__(self,args=None):
        if args is None:
            self.trainDataPath = './Data/Train/'
            self.testDataPath = './Data/Test/'
            self.outputPath = './Output/'
            self.modelPath = './Model/'
            self.modelName = 'depth_model.h5'
            self.isTrain = False
            self.isStreamed = False
            self.usePreTrained = False
            self.batchSize = 1
            self.shuffleBufferSize = 1000
            self.activation = 'relu'
            self.dropoutRate = 0.01
            self.learningRate = 0.001
            self.epochs = 20
            self.trainValRatio = 0.2
        else:
            self.trainDataPath = args.trainDataPath
            self.testDataPath = args.testDataPath
            self.outputPath = args.outputPath
            self.modelPath = args.modelPath
            self.modelName = args.modelName
            self.isTrain = args.isTrain
            self.isStreamed = args.isStreamed
            self.usePreTrained = args.usePreTrained
            self.batchSize = args.batchSize
            self.shuffleBufferSize = args.shuffleBufferSize
            self.activation = args.activation
            self.dropoutRate = args.dropoutRate
            self.learningRate = args.learningRate
            self.epochs = args.epochs
            self.trainValRatio = args.trainValRatio
        self.prefetchBufferSize=10
        self.trainDataset = None
        self.testDataset = None
        self.model = None

    def process_data_files(self, filePath):

        grid = tf.zeros([self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z],
                        tf.float32)

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
        sparseValues = tf.ones([shape[0]], tf.float32)
        delta = tf.sparse.SparseTensor(indices=indexes, values=sparseValues, dense_shape=[self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z])
        grid = tf.math.add(grid, tf.sparse.to_dense(delta, default_value=False, validate_indices=False))
        grid = tf.expand_dims(grid, -1)

        return grid, centroid, furthestDistanceFromCentre

    def read_data(self):

        trainPath = os.path.abspath(self.trainDataPath)
        testPath = os.path.abspath(self.testDataPath)

        # # Display the files in path
        # trainFiles = path_reader(trainPath)
        # testFiles = path_reader(testPath)
        # print(trainFiles)
        # print(testFiles)

        trainPath = os.path.join(trainPath, '*.txt')
        testPath = os.path.join(testPath, '*.txt')
        numElements = len(glob.glob(trainPath))
        trainFilesDs = tf.data.Dataset.list_files(trainPath)
        testFilesDs = tf.data.Dataset.list_files(testPath)

        self.trainDataset = trainFilesDs.map(lambda filePath: self.process_data_files(filePath))
        self.trainDataset = self.trainDataset.shuffle(self.shuffleBufferSize)
        self.valDataset = self.trainDataset.take(math.ceil(self.trainValRatio*numElements))

        self.trainDataset = self.trainDataset.skip(math.ceil(self.trainValRatio*numElements))
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

    def load_weights(self):

        file = os.path.join(self.modelPath, self.modelName)
        if os.path.exists(file):
            self.model.load_weights(file)
        else:
            print('Pre-trained Model not exists. using default weights!!')

    def create_model(self):

        input = tfk.Input(shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1))

        layers1 = tfk.layers.Conv3D(64,
                                   (3, 3, 3),
                                   padding='same',
                                   activation=self.activation,
                                   input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                   kernel_initializer=tf.initializers.glorot_normal)(input)
        layers1 = tfk.layers.BatchNormalization()(layers1)
        layers1 = tfk.layers.Dropout(self.dropoutRate)(layers1)
        layers1 = tfk.layers.Conv3D(1,
                                   (3, 3, 3),
                                   strides=(1, 1, 1),
                                   padding='same',
                                   activation=self.activation,
                                   input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                   kernel_initializer=tf.initializers.glorot_normal)(layers1)
        layers1 = tfk.layers.BatchNormalization()(layers1)
        layers1 = tfk.layers.Dropout(self.dropoutRate)(layers1)
        edge1 = SobelFilter(name='edge_1', trainable=False)(layers1)
        out1 = tfk.layers.Activation(activation='softmax')(edge1)

        layers2 = tfk.layers.Conv3D(64,
                                    (3, 3, 3),
                                    strides=(2, 2, 2),
                                    padding='same',
                                    activation=self.activation,
                                    input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                    kernel_initializer=tf.initializers.glorot_normal)(edge1)
        layers2 = tfk.layers.BatchNormalization()(layers2)
        layers2 = tfk.layers.Dropout(self.dropoutRate)(layers2)
        layers2 = tfk.layers.Conv3D(1,
                                    (3, 3, 3),
                                    strides=(1, 1, 1),
                                    padding='same',
                                    activation=self.activation,
                                    input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                    kernel_initializer=tf.initializers.glorot_normal)(layers2)
        layers2 = tfk.layers.BatchNormalization()(layers2)
        layers2 = tfk.layers.Dropout(self.dropoutRate)(layers2)
        edge2 = SobelFilter(name='edge_2', trainable=False)(layers2)
        out2 = tfk.layers.Activation(activation='softmax')(edge2)

        layers3 = tfk.layers.Conv3D(64,
                                    (3, 3, 3),
                                    strides=(2, 2, 2),
                                    padding='same',
                                    activation=self.activation,
                                    input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                    kernel_initializer=tf.initializers.glorot_normal)(edge2)
        layers3 = tfk.layers.BatchNormalization()(layers3)
        layers3 = tfk.layers.Dropout(self.dropoutRate)(layers3)
        layers3 = tfk.layers.Conv3D(1,
                                    (3, 3, 3),
                                    strides=(1, 1, 1),
                                    padding='same',
                                    activation=self.activation,
                                    input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                    kernel_initializer=tf.initializers.glorot_normal)(layers3)
        layers3 = tfk.layers.BatchNormalization()(layers3)
        layers3 = tfk.layers.Dropout(self.dropoutRate)(layers3)
        edge3 = SobelFilter(name='edge_3', trainable=False)(layers3)
        out3 = tfk.layers.Activation(activation='softmax')(edge3)
        outt = tfk.layers.Concatenate()([out1, out2, out3])

        self.model = tfk.Model(inputs=input, outputs=[outt, out1, out2, out3])

        opt = tfk.optimizers.Adam(learning_rate=self.learningRate)
        loss = tfk.losses.BinaryCrossentropy()

        self.model.compile(optimizer=opt,
                           loss=loss,
                           metrics=['accuracy', tfk.metrics.MeanIoU(num_classes=2)])

        self.model.summary()

    def train(self):

        self.model.fit()

    def run(self):

        try:
            self.read_data()
        except Exception as e:
            print('Data read Error')
            print()
            print(str(e))
            exit(1)

        self.create_model()

        if self.usePreTrained:
            self.load_weights()

        # if self.isTrain:
        #     self.train()
        # else:
        #     self.predict()














