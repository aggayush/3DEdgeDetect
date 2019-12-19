################################
# Main class file to execute the code
################################\

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import tensorflow.keras as tfk
import math
import glob
import numpy as np
from Source.CustamLayer import SobelFilter, MergePointCloud, WeightedBinaryCrossEntropy
from tensorflow.python.framework.ops import disable_eager_execution
from Source.utilities import path_reader, visualize_from_file

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# disable_eager_execution()


class ThreeDEdgeDetector:
    VOXEL_GRID_X = 32
    VOXEL_GRID_Y = 32
    VOXEL_GRID_Z = 32
    DATA_DEFAULTS = [[0.], [0.], [0.]]

    def __init__(self, args=None):
        if args is None:
            self.trainDataPath = './Data/Train/'
            self.testDataPath = './Data/Test/'
            self.outputPath = './Output/'
            self.modelPath = './Model/'
            self.logDir = './log/'
            self.modelName = 'depth_model.h5'
            self.isTrain = False
            self.isStreamed = False
            self.usePreTrained = False
            self.batchSize = 2
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
            self.logDir = args.logDir
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
        self.prefetchBufferSize = 10
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
        furthestDistanceFromCentre = tf.math.reduce_max(
            tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.abs(data)), axis=-1)))
        data = tf.math.divide(data, furthestDistanceFromCentre)

        # Voxelization of data
        gridX = tf.floor(tf.add(tf.multiply(data[:, 0], (self.VOXEL_GRID_X / 2)), self.VOXEL_GRID_X / 2))
        gridY = tf.floor(tf.add(tf.multiply(data[:, 1], (self.VOXEL_GRID_Y / 2)), self.VOXEL_GRID_Y / 2))
        gridZ = tf.floor(tf.add(tf.multiply(data[:, 2], (self.VOXEL_GRID_Z / 2)), self.VOXEL_GRID_Z / 2))

        indexes = tf.stack([gridX, gridY, gridZ], axis=1)
        indexes = tf.cast(indexes, tf.int64)
        shape = tf.shape(indexes)
        sparseValues = tf.ones([shape[0]], tf.float32)
        delta = tf.sparse.SparseTensor(indices=indexes, values=sparseValues,
                                       dense_shape=[self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z])
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
        self.valDataset = self.trainDataset.take(math.ceil(self.trainValRatio * numElements))

        self.trainDataset = self.trainDataset.skip(math.ceil(self.trainValRatio * numElements))
        self.trainDataset = self.trainDataset.batch(self.batchSize)
        self.trainDataset = self.trainDataset.prefetch(self.prefetchBufferSize)

        self.valDataset = self.valDataset.batch(1)
        self.valDataset = self.valDataset.prefetch(self.prefetchBufferSize)

        self.testDataset = testFilesDs.map(lambda filePath: self.process_data_files(filePath))
        self.testDataset = self.testDataset.shuffle(self.shuffleBufferSize)
        self.testDataset = self.testDataset.batch(1)
        self.testDataset = self.testDataset.prefetch(self.prefetchBufferSize)

        # # to see filename and visualize data
        # for f in trainFilesDs.take(1):
        #     print(f.numpy())
        #     visualize_from_file(f.numpy().decode("utf-8"))
        # for f in testFilesDs.take(1):
        #     print(f.numpy())
        #     visualize_from_file(f.numpy().decode("utf-8"))

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
                                    strides=(1, 1, 1),
                                    activation=self.activation,
                                    input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                    kernel_initializer=tf.initializers.glorot_normal,
                                    name='Conv1')(input)
        # layers1 = tfk.layers.BatchNormalization()(layers1)
        layers1 = tfk.layers.Dropout(self.dropoutRate)(layers1)
        layers1 = tfk.layers.Conv3D(1,
                                    (3, 3, 3),
                                    strides=(2, 2, 2),
                                    padding='same',
                                    activation=self.activation,
                                    name='Conv2')(layers1)
        # layers1 = tfk.layers.BatchNormalization()(layers1)
        layers1 = tfk.layers.Dropout(self.dropoutRate)(layers1)
        edge1 = SobelFilter(name='edge_1', trainable=False)(layers1)
        layers1 = tfk.layers.Activation(activation=self.activation)(edge1)

        layers2 = tfk.layers.Conv3D(64,
                                    (3, 3, 3),
                                    strides=(1, 1, 1),
                                    padding='same',
                                    activation=self.activation,
                                    kernel_initializer=tf.initializers.glorot_normal,
                                    name='Conv3')(layers1)
        # layers2 = tfk.layers.BatchNormalization()(layers2)
        layers2 = tfk.layers.Dropout(self.dropoutRate)(layers2)
        layers2 = tfk.layers.Conv3D(1,
                                    (3, 3, 3),
                                    strides=(2, 2, 2),
                                    padding='same',
                                    activation=self.activation,
                                    kernel_initializer=tf.initializers.glorot_normal,
                                    name='Conv4')(layers2)
        # layers2 = tfk.layers.BatchNormalization()(layers2)
        layers2 = tfk.layers.Dropout(self.dropoutRate)(layers2)
        edge2 = SobelFilter(name='edge_2', trainable=False)(layers2)
        layers2 = tfk.layers.Activation(activation=self.activation)(edge2)

        layers3 = tfk.layers.Conv3D(64,
                                    (3, 3, 3),
                                    strides=(1, 1, 1),
                                    padding='same',
                                    activation=self.activation,
                                    kernel_initializer=tf.initializers.glorot_normal,
                                    name='Conv5')(layers2)
        # layers3 = tfk.layers.BatchNormalization()(layers3)
        layers3 = tfk.layers.Dropout(self.dropoutRate)(layers3)
        layers3 = tfk.layers.Conv3D(1,
                                    (3, 3, 3),
                                    strides=(2, 2, 2),
                                    padding='same',
                                    activation=self.activation,
                                    kernel_initializer=tf.initializers.glorot_normal,
                                    name='Conv6')(layers3)
        # layers3 = tfk.layers.BatchNormalization()(layers3)
        layers3 = tfk.layers.Dropout(self.dropoutRate)(layers3)
        edge3 = SobelFilter(name='edge_3', trainable=False)(layers3)
        layers4 = MergePointCloud([self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z],
                                  'avg',
                                  name='output_1',
                                  trainable=False)([edge1, edge2, edge3])
        # layers4 = tfk.layers.Conv3D(2,
        #                             (3, 3, 3),
        #                             strides=(1, 1, 1),
        #                             padding='same',
        #                             kernel_initializer=tf.initializers.glorot_normal,
        #                             name='Conv7')(layers4)
        outFinal = tfk.layers.Activation(activation='softmax')(layers4)

        self.model = tfk.Model(inputs=input, outputs=outFinal)

        self.model.summary()

    def train(self):

        opt = tfk.optimizers.Adadelta(learning_rate=self.learningRate)
        # loss = tfk.losses.BinaryCrossentropy()
        loss = WeightedBinaryCrossEntropy(0.8, 2)

        def calc_loss(inp, tar):
            tar_ = self.model(inp)

            # return loss(y_true=tar, y_pred=tar_)
            return loss(tar, tar_)

        def grad(inp, tar):
            with tf.GradientTape() as tape:
                loss_value = calc_loss(inp, tar)
            return tape.gradient(loss_value, self.model.trainable_variables)

        for epoch in range(self.epochs):
            epochTrainLossAvg = tfk.metrics.Mean()
            epochTrainAccuracy = tfk.metrics.BinaryAccuracy()
            epochTrainMIOU = tfk.metrics.MeanIoU(num_classes=2)

            epochValLossAvg = tfk.metrics.Mean()
            epochValAccuracy = tfk.metrics.BinaryAccuracy()
            epochValMIOU = tfk.metrics.MeanIoU(num_classes=2)

            for x, c, f in self.trainDataset:
                y = x
                # y = tf.cast(y, tf.bool)
                # temp1 = tf.logical_not(y)
                # y = tf.concat([temp1, y], axis=-1)
                # y = tf.cast(y, tf.float32)

                pred = self.model(x)
                lossVal = calc_loss(x, y)
                epochTrainLossAvg.update_state(lossVal)
                epochTrainAccuracy.update_state(y, pred)
                epochTrainMIOU.update_state(y, pred)

                grads = grad(x, y)
                opt.apply_gradients(zip(grads, self.model.trainable_variables))

            for x, c, f in self.valDataset:
                y = x
                # y = tf.cast(y, tf.bool)
                # temp1 = tf.logical_not(y)
                # y = tf.concat([temp1, y], axis=-1)
                # y = tf.cast(y, tf.float32)

                lossVal = calc_loss(x, y)
                pred = self.model(x)
                epochValLossAvg.update_state(lossVal)
                epochValAccuracy.update_state(y, pred)
                epochValMIOU.update_state(y, pred)

            print("Epoch {:03d}: Train_Loss: {:.3f}, Train_Accuracy: {:.3%}, Train_MIOU: {:.3f}, "
                  "Val_Loss: {:.3f}, Val_Accuracy: {:.3%}, Train_MIOU: {:.3f}".format(epoch,
                                                                                      epochTrainLossAvg.result(),
                                                                                      epochTrainAccuracy.result(),
                                                                                      epochTrainMIOU.result(),
                                                                                      epochValLossAvg.result(),
                                                                                      epochValAccuracy.result(),
                                                                                      epochValMIOU.result()))

    def test(self):

        for x, c, f in self.testDataset:
            pred = self.model(x)
            pred = pred.numpy()
            pred = pred[0, :, :, :, 0]
            indexX, indexY, indexZ = np.where(pred > 0.2)
            pointsX = indexX - self.VOXEL_GRID_X / 2

    def run(self):

        # disable_eager_execution()
        try:
            self.read_data()
        except Exception as e:
            print('Data read Error')
            print()
            print(e)
            exit(1)

        self.create_model()

        # data = next(iter(self.trainDataset))
        # out = self.model(data[0])
        # out = out.numpy()

        if self.usePreTrained:
            self.load_weights()

        if self.isTrain:
            self.train()
            self.test()
        else:
            self.predict()
