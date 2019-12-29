################################
# Autoencoder for learning features from 3D point cloud
################################\

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import tensorflow.keras as tfk
import math
import glob
import numpy as np
from Source.CustamLayer import WeightedLoss, MergePointCloud
from Source.utilities import path_reader, visualize_from_file, visualize_point_cloud

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class AutoEncoder:
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
            self.modelName = 'encoder_decoder.h5'
            self.isTrain = False
            self.isStreamed = False
            self.usePreTrained = True
            self.batchSize = 40
            self.shuffleBufferSize = 1000
            self.activation = 'relu'
            self.dropoutRate = 0.01
            self.learningRate = 0.001
            self.epochs = 300
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
        self.prefetchBufferSize = 100
        self.trainDataset = None
        self.testDataset = None
        self.model = None

    def process_data_files(self, filePath):

        grid = tf.zeros([self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z],
                        tf.float32)

        rawData = tf.io.read_file(filePath)
        records = tf.strings.split(rawData, sep='\r\n')
        records = records[:-1]
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
        grid = tf.math.add(grid, tf.sparse.to_dense(delta, default_value=0.0, validate_indices=False))
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

        trainPath = os.path.join(trainPath, '*.xyz')
        testPath = os.path.join(testPath, '*.xyz')
        trainFilesDs = tf.data.Dataset.list_files(trainPath, shuffle=False)
        testFilesDs = tf.data.Dataset.list_files(testPath, shuffle=False)

        self.trainDataset = trainFilesDs.map(lambda filePath: self.process_data_files(filePath))
        self.trainDataset = self.trainDataset.shuffle(self.shuffleBufferSize)

        self.trainDataset = self.trainDataset.batch(self.batchSize)
        self.trainDataset = self.trainDataset.prefetch(self.prefetchBufferSize)

        self.testDataset = testFilesDs.map(lambda filePath: self.process_data_files(filePath))
        # self.testDataset = self.testDataset.shuffle(self.shuffleBufferSize)
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
            self.model.load_weights(file, by_name=True)
        else:
            print('Pre-trained Model not exists. using default weights!!')

    def save_weights(self, filename=None):
        if filename is None:
            filename = self.modelName

        file = os.path.join(self.modelPath, filename)
        self.model.save_weights(file)

    def create_model(self):

        input = tfk.Input(shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1))

        layers = tfk.layers.Conv3D(5,
                                   (3, 3, 3),
                                   padding='same',
                                   activation=self.activation,
                                   input_shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1),
                                   name='Encoder_Conv1')(input)
        branch1 = tfk.layers.Conv3D(1,
                                    (1, 1, 1),
                                    strides=(1, 1, 1),
                                    padding='same',
                                    name='Branch1_Conv1')(layers)
        layers = tfk.layers.Conv3D(5,
                                   (3, 3, 3),
                                   strides=(2, 2, 2),
                                   padding='same',
                                   activation=self.activation,
                                   name='Encoder_Conv2')(layers)
        branch2 = tfk.layers.Conv3D(1,
                                    (1, 1, 1),
                                    strides=(1, 1, 1),
                                    padding='same',
                                    name='Branch2_Conv1')(layers)
        layers = tfk.layers.Conv3D(5,
                                   (3, 3, 3),
                                   strides=(2, 2, 2),
                                   padding='same',
                                   activation=self.activation,
                                   name='Encoder_Conv3')(layers)
        branch3 = tfk.layers.Conv3D(1,
                                    (1, 1, 1),
                                    strides=(1, 1, 1),
                                    padding='same',
                                    name='Branch3_Conv1')(layers)
        layers = tfk.layers.Conv3DTranspose(5,
                                            (3, 3, 3),
                                            strides=(2, 2, 2),
                                            padding='same',
                                            activation=self.activation,
                                            name='DeConv3')(layers)
        layers = tfk.layers.BatchNormalization(name='BatchNorm3')(layers)
        layers = tfk.layers.Conv3DTranspose(5,
                                            (3, 3, 3),
                                            strides=(2, 2, 2),
                                            padding='same',
                                            activation=self.activation,
                                            name='DeConv2')(layers)
        layers = tfk.layers.BatchNormalization(name='BatchNorm2')(layers)
        outFinal = tfk.layers.Conv3D(1,
                                     (3, 3, 3),
                                     padding='same',
                                     activation='sigmoid',
                                     name='FinalOut')(layers)
        outFinal_branch = MergePointCloud([self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z],
                                  'sum',
                                  trainable=True)([branch1, branch2, branch3])
        outFinal_branch = tfk.layers.Activation('sigmoid')(outFinal_branch)

        self.model = tfk.Model(inputs=input, outputs=[outFinal, outFinal_branch])

        self.model.summary()

    def train(self):

        opt = tfk.optimizers.Adam(learning_rate=self.learningRate, decay=self.learningRate/(self.epochs*75), clipvalue=10.0)
        loss = WeightedLoss(4, 2, 1)

        def calc_loss(inp, tar):
            tar_ = self.model(inp)

            return loss(tar[0], tar_[0]) + loss(tar[1], tar_[1])

        def grad(inp, tar):
            with tf.GradientTape() as tape:
                loss_value = calc_loss(inp, tar)
            return tape.gradient(loss_value, self.model.trainable_variables)

        for epoch in range(self.epochs):
            epochTrainLossAvg = tfk.metrics.Mean()

            epochValLossAvg = tfk.metrics.Mean()
            iteration = 0
            for x, c, f in self.trainDataset:
                y = [x, x]

                grads = grad(x, y)
                opt.apply_gradients(zip(grads, self.model.trainable_variables))

                # pred = self.model(x)
                lossVal = calc_loss(x, y)
                epochTrainLossAvg.update_state(lossVal)
                print("Iter {:03d}: Train_Loss: {:.8f} ".format(iteration, epochTrainLossAvg.result()))
                iteration = iteration + 1

            for x, c, f in self.testDataset:
                y = [x, x]
                lossVal = calc_loss(x, y)
                epochValLossAvg.update_state(lossVal)

            print("Epoch {:03d}: Train_Loss: {:.8f}, "
                  "Val_Loss: {:.8f}".format(epoch,
                                            epochTrainLossAvg.result(),
                                            epochValLossAvg.result()))

            self.save_weights()

    def predict(self):

        for x, c, f in self.testDataset:
            pred = x.numpy()
            pred = pred[0, :, :, :, 0]
            indexX, indexY, indexZ = np.where(pred > 0)
            pointsX = (indexX - (self.VOXEL_GRID_X / 2)) / (self.VOXEL_GRID_X / 2) + (1 / self.VOXEL_GRID_X)
            pointsY = (indexY - (self.VOXEL_GRID_Y / 2)) / (self.VOXEL_GRID_Y / 2) + (1 / self.VOXEL_GRID_Y)
            pointsZ = (indexZ - (self.VOXEL_GRID_Z / 2)) / (self.VOXEL_GRID_Z / 2) + (1 / self.VOXEL_GRID_Z)
            pointCloud = np.stack([pointsX, pointsY, pointsZ], axis=1)
            visualize_point_cloud(pointCloud)
            preds = self.model(x)
            pred = preds[0].numpy()
            pred = pred[0, :, :, :, 0]
            indexX, indexY, indexZ = np.where(pred > 0.1)
            pointsX = (indexX - (self.VOXEL_GRID_X / 2)) / (self.VOXEL_GRID_X / 2) + (1 / self.VOXEL_GRID_X)
            pointsY = (indexY - (self.VOXEL_GRID_Y / 2)) / (self.VOXEL_GRID_Y / 2) + (1 / self.VOXEL_GRID_Y)
            pointsZ = (indexZ - (self.VOXEL_GRID_Z / 2)) / (self.VOXEL_GRID_Z / 2) + (1 / self.VOXEL_GRID_Z)
            pointCloud = np.stack([pointsX, pointsY, pointsZ], axis=1)
            visualize_point_cloud(pointCloud)
            pred = preds[1].numpy()
            pred = pred[0, :, :, :, 0]
            indexX, indexY, indexZ = np.where(pred < -1)
            siz = 8
            pointsX = (indexX - (siz / 2)) / (siz / 2) + (1 / siz)
            pointsY = (indexY - (siz / 2)) / (siz / 2) + (1 / siz)
            pointsZ = (indexZ - (siz / 2)) / (siz / 2) + (1 / siz)
            pointCloud = np.stack([pointsX, pointsY, pointsZ], axis=1)
            visualize_point_cloud(pointCloud)

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
        else:
            self.predict()
