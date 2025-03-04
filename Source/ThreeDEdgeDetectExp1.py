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
import time
from Source.CustamLayer import SobelFilter, MergePointCloud, WeightedLoss, KMeansClusteringLayer, MinPooling3D, \
    GMMClusteringLayer
from tensorflow.python.framework.ops import disable_eager_execution
from Source.utilities import path_reader, visualize_from_file, visualize_point_cloud, visualize_histogram, \
    save_point_cloud, create_color_from_val

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# disable_eager_execution()


class ThreeDEdgeDetectorExp1:
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
            self.modelName = 'edge_model_exp1_mnist.h5'
            self.encoderModelName = 'encoder_decoder.h5'
            self.isTrain = False
            self.isStreamed = False
            self.usePreTrained = True
            self.batchSize = 40
            self.shuffleBufferSize = 1000
            self.activation = 'relu'
            self.dropoutRate = 0.01
            self.learningRate = 0.01
            self.epochs = 50
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
        self.prefetchBufferSize = 100
        self.trainDataset = None
        self.testDataset = None
        self.model = None

    # @tf.function
    # def augmentation(self, data):
    #     with tf.device('/GPU:0'):
    #         x_cord, y_cord, z_cord = tf.split(data, [1, 1, 1], axis=1)
    #
    #         #rotation in X
    #         rot_x = tf.random.uniform(shape=[], minval=-1.0*np.pi, maxval=np.pi, dtype=tf.float32)
    #         y_cord_n = y_cord * tf.cos(rot_x) + z_cord * tf.sin(rot_x)
    #         z_cord_n = y_cord * -1.0 * tf.sin(rot_x) + z_cord * tf.cos(rot_x)
    #         y_cord = y_cord_n
    #         z_cord = z_cord_n
    #
    #         del y_cord_n
    #         del z_cord_n
    #         del rot_x
    #
    #         # rotation in Y
    #         rot_y = tf.random.uniform(shape=[], minval=-1.0*np.pi, maxval=np.pi, dtype=tf.float32)
    #
    #         x_cord_n = x_cord * tf.cos(rot_y) + z_cord * -1.0 * tf.sin(rot_y)
    #         z_cord_n = x_cord * tf.sin(rot_y) + z_cord * tf.cos(rot_y)
    #         x_cord = x_cord_n
    #         z_cord = z_cord_n
    #
    #         del x_cord_n
    #         del z_cord_n
    #         del rot_y
    #
    #         # rotation in Z
    #         rot_z = tf.random.uniform(shape=[], minval=-1.0*np.pi, maxval=np.pi, dtype=tf.float32)
    #
    #         x_cord_n = x_cord * tf.cos(rot_z) + y_cord * tf.sin(rot_z)
    #         y_cord_n = x_cord * -1.0 * tf.sin(rot_z) + y_cord * tf.cos(rot_z)
    #         x_cord = x_cord_n
    #         y_cord = y_cord_n
    #
    #         del x_cord_n
    #         del y_cord_n
    #         del rot_z
    #
    #         # translation in XYZ
    #         tr_x = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5, dtype=tf.float32)
    #         tr_y = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5, dtype=tf.float32)
    #         tr_z = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5, dtype=tf.float32)
    #
    #         x_cord = x_cord + tr_x
    #         y_cord = y_cord + tr_y
    #         z_cord = z_cord + tr_z
    #
    #         del tr_x
    #         del tr_y
    #         del tr_z
    #
    #         # scaling in XYZ
    #         sc = tf.random.uniform(shape=[], minval=0.5, maxval=2.0, dtype=tf.float32)
    #         # sc_y = tf.random.uniform(shape=[], minval=0.5, maxval=2.0, dtype=tf.float32)
    #         # sc_z = tf.random.uniform(shape=[], minval=0.5, maxval=2.0, dtype=tf.float32)
    #
    #         x_cord = x_cord * sc
    #         y_cord = y_cord * sc
    #         z_cord = z_cord * sc
    #
    #         del sc
    #
    #         # # shearing in XYZ
    #         # sh_yx = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5, dtype=tf.float32)
    #         # sh_xy = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5, dtype=tf.float32)
    #         # sh_zx = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5, dtype=tf.float32)
    #         # sh_xz = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5, dtype=tf.float32)
    #         # sh_yz = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5, dtype=tf.float32)
    #         # sh_zy = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5, dtype=tf.float32)
    #         #
    #         # x_cord_n = x_cord +
    #
    #         data = tf.concat([x_cord, y_cord, z_cord], axis=1)
    #
    #         del x_cord
    #         del y_cord
    #         del z_cord
    #
    #     return data

    @tf.function
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
        # data = self.augmentation(data)
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
        # self.trainDataset = self.trainDataset.shuffle(self.shuffleBufferSize)
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
        encoderFile = os.path.join(self.modelPath, self.encoderModelName)
        if os.path.exists(file):
            self.model.load_weights(file, by_name=True)
            print("Loaded complete pre-trained model")
        elif os.path.exists(encoderFile):
            self.model.load_weights(encoderFile, by_name=True)
            print("Loaded encoder model only")
        else:
            print('Pre-trained Model not exists. using default weights!!')

    def save_weights(self, filename=None):

        if filename is None:
            filename = self.modelName
        file = os.path.join(self.modelPath, filename)

        self.model.save_weights(file)

    def create_model(self):

        input = tfk.Input(shape=(self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z, 1), batch_size=self.batchSize)

        branch1 = SobelFilter(name='Branch1_Sobel1', trainable=False)(input)
        outFinal_clus = tfk.layers.Activation('relu')(branch1)
        # outFinal_clus = tfk.layers.BatchNormalization()(outFinal_clus)
        outFinal_clus = KMeansClusteringLayer(nClusters=2, name='kmm_clus')(outFinal_clus)

        self.model = tfk.Model(inputs=input, outputs=outFinal_clus)
        # self.model = tfk.Model(inputs=input, outputs=outFinal_clus)

        self.model.summary()

    def train(self):

        opt = tfk.optimizers.Adam(learning_rate=self.learningRate, decay=self.learningRate / (self.epochs * 75),
                                  clipvalue=10.0)
        loss_clus = tfk.losses.KLDivergence()
        loss_enc = WeightedLoss(4, 2, 1)

        def calc_loss(inp, tar):
            tar_ = self.model(inp)

            return loss_clus(tar, tar_)

        def grad(inp, tar):
            with tf.GradientTape() as tape:
                loss_value = calc_loss(inp, tar)
            return tape.gradient(loss_value, self.model.trainable_variables)

        def update_target_distribution(q):
            freq = tf.reduce_sum(tf.reshape(q, (-1, 2)), axis=0)
            weight = tf.divide(q, freq)
            weight = tf.divide(weight, tf.expand_dims(tf.reduce_sum(weight, axis=-1), axis=-1))

            return weight

        update_epoch = 1

        for epoch in range(self.epochs):
            epochTrainLossAvg = tfk.metrics.Mean()
            epochValLossAvg = tfk.metrics.Mean()

            if epoch % update_epoch == 0:
                whole_pred = None
                # updating the target distribution
                for x, c, f in self.trainDataset:
                    p = self.model(x)
                    if whole_pred is None:
                        whole_pred = p
                    else:
                        whole_pred = tf.concat([whole_pred, p], axis=0)

                y = update_target_distribution(whole_pred)

                predDataset = tf.data.Dataset.from_tensor_slices(y)
                predDataset = predDataset.batch(self.batchSize)
                predDataset = predDataset.prefetch(self.prefetchBufferSize)

                finalDataset = tf.data.Dataset.zip((self.trainDataset, predDataset))

                del predDataset
                del y
                del whole_pred
                del p

            iteration = 0
            for x, y in finalDataset:
                grads = grad(x[0], y)
                opt.apply_gradients(zip(grads, self.model.trainable_variables))

                lossVal = calc_loss(x[0], y)
                epochTrainLossAvg.update_state(lossVal)

                print("iter {:04d}: Train_loss: {:.8f}".format(iteration, lossVal))
                iteration = iteration + 1

            # for x, c, f in self.testDataset:
            #     lossVal = calc_loss(x, y)
            #     epochValLossAvg.update_state(lossVal)

            # self.predict()

            print("Epoch {:03d}: Train_Loss: {:.8f}".format(epoch,
                                                            epochTrainLossAvg.result()))

            self.save_weights(str(epoch) + '.h5')

    def predict(self):

        i = 0
        # diff = 0
        # start_time = time.time()
        for x, c, f in self.testDataset:
            # print(i)
            inp = x.numpy()
            inp = inp[0, :, :, :, 0]
            indexX, indexY, indexZ = np.where(inp > 0)
            pointsX = (indexX - (self.VOXEL_GRID_X / 2)) / (self.VOXEL_GRID_X / 2) + (1 / self.VOXEL_GRID_X)
            pointsY = (indexY - (self.VOXEL_GRID_Y / 2)) / (self.VOXEL_GRID_Y / 2) + (1 / self.VOXEL_GRID_Y)
            pointsZ = (indexZ - (self.VOXEL_GRID_Z / 2)) / (self.VOXEL_GRID_Z / 2) + (1 / self.VOXEL_GRID_Z)
            pointCloud = np.stack([pointsX, pointsY, pointsZ], axis=1)
            visualize_point_cloud(pointCloud)
            # save_point_cloud(pointCloud, self.outputPath, str(i) + "_orig")
            # stime = time.time()
            preds = self.model(x)
            # # diff = diff + (time.time() - stime )
            # pred1 = preds[1]
            # pred1 = pred1.numpy()
            # pred1 = pred1[0, :, :, :, 0]
            # indexX, indexY, indexZ = np.where(pred1 > 0.5)
            # siz = 32
            # pointsX = (indexX - (siz / 2)) / (siz / 2) + (1 / siz)
            # pointsY = (indexY - (siz / 2)) / (siz / 2) + (1 / siz)
            # pointsZ = (indexZ - (siz / 2)) / (siz / 2) + (1 / siz)
            # pointCloud = np.stack([pointsX, pointsY, pointsZ], axis=1)
            # visualize_point_cloud(pointCloud)
            pred = preds
            pred = pred.numpy()
            pred = pred[0, :, :, :, :]
            # visualize_histogram(pred)
            # pred = 1 - pred
            pred_idx = np.argmax(pred, axis=-1)
            pred_val = np.max(pred, axis=-1)
            indexX, indexY, indexZ = np.where(pred_idx == 1)
            idx = np.where(pred_val[indexX, indexY, indexZ] < 0.5)
            indexX = np.delete(indexX, idx)
            indexY = np.delete(indexY, idx)
            indexZ = np.delete(indexZ, idx)
            pointsX = (indexX - (self.VOXEL_GRID_X / 2)) / (self.VOXEL_GRID_X / 2) + (1 / self.VOXEL_GRID_X)
            pointsY = (indexY - (self.VOXEL_GRID_Y / 2)) / (self.VOXEL_GRID_Y / 2) + (1 / self.VOXEL_GRID_Y)
            pointsZ = (indexZ - (self.VOXEL_GRID_Z / 2)) / (self.VOXEL_GRID_Z / 2) + (1 / self.VOXEL_GRID_Z)
            pointCloud = np.stack([pointsX, pointsY, pointsZ], axis=1)
            colors = create_color_from_val(pred_val[indexX, indexY, indexZ])
            visualize_point_cloud(pointCloud)
            # save_point_cloud(pointCloud, self.outputPath, str(i) + "_predg5")
            idx = np.where(pred_val[indexX, indexY, indexZ] > 0.7)
            indexX = np.delete(indexX, idx)
            indexY = np.delete(indexY, idx)
            indexZ = np.delete(indexZ, idx)
            pointsX = (indexX - (self.VOXEL_GRID_X / 2)) / (self.VOXEL_GRID_X / 2) + (1 / self.VOXEL_GRID_X)
            pointsY = (indexY - (self.VOXEL_GRID_Y / 2)) / (self.VOXEL_GRID_Y / 2) + (1 / self.VOXEL_GRID_Y)
            pointsZ = (indexZ - (self.VOXEL_GRID_Z / 2)) / (self.VOXEL_GRID_Z / 2) + (1 / self.VOXEL_GRID_Z)
            pointCloud = np.stack([pointsX, pointsY, pointsZ], axis=1)
            visualize_point_cloud(pointCloud)
            # save_point_cloud(pointCloud, self.outputPath, str(i) + "_predg5l55")
            i = i + 1

        # end_time = time.time()
        # print("--- {:.10f} seconds with data load time ---".format((end_time - start_time)/1000))
        # print("--- {:.10f} seconds without data load time ---".format(diff/1000))

    def run(self):

        try:
            self.read_data()
        except Exception as e:
            print('Data read Error')
            print()
            print(e)
            exit(1)

        self.create_model()

        if self.usePreTrained:
            self.load_weights()

        # dat = next(iter(self.testDataset))
        # sob = SobelFilter()
        # out = sob(dat[0])
        # out = out.numpy()
        # out = out[0,:,:,:,:]
        # visualize_histogram(out)

        if self.isTrain:
            self.train()
        else:
            self.predict()
