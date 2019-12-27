######################################
## utlities function for the code
########################################

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from os import listdir
from os.path import isfile, join
import open3d
import numpy as np
import matplotlib.pyplot as plt


def arg_creator():
    parser = argparse.ArgumentParser(description='Arguments for the edge detector')
    parser.add_argument('--train-data-path',
                        default='./Data/Train/',
                        dest='trainDataPath',
                        help='Path to the training data files')
    parser.add_argument('--test-data-path',
                        default='./Data/Test/',
                        dest='testDataPath',
                        help='Path to the testing data files')
    parser.add_argument('--output-path',
                        default='./Output/',
                        dest='outputPath',
                        help='Path to the output folder')
    parser.add_argument('--model-path',
                        default='./Model/',
                        dest='modelPath',
                        help='Path to the model save folder (same for loading pretrained model)')
    parser.add_argument('--log-dir',
                        default='./log/',
                        dest='logDir',
                        help='Log Directory path')
    parser.add_argument('--model-name',
                        default='encoder_decoder.h5',
                        dest='modelName',
                        help='Model filename to load into memory')
    parser.add_argument('--is-train',
                        action='store_true',
                        default=True,
                        dest='isTrain',
                        help='To run training or testing')
    parser.add_argument('--is-streamed',
                        action='store_true',
                        default=False,
                        dest='isStreamed',
                        help='if data is streamed input or to be read from file')
    parser.add_argument('--batch-size',
                        default=20,
                        dest='batchSize',
                        help='batch Size to process')
    parser.add_argument('--shuffle-buffer-size',
                        default=1000,
                        dest='shuffleBufferSize',
                        help='Data shuffle buffer size')
    parser.add_argument('--activation',
                        default='relu',
                        dest='activation',
                        help='Activation function to use')
    parser.add_argument('--dropout-rate',
                        default=0.01,
                        dest='dropoutRate',
                        help='Dropout rate for the dropout layers')
    parser.add_argument('--use-pretrained',
                        default=True,
                        action='store_true',
                        dest='usePreTrained',
                        help='To use pretrained weights or not')
    parser.add_argument('--learning-rate',
                        default=0.001,
                        dest='learningRate',
                        help='Learning rate of the network')
    parser.add_argument('--epochs',
                        default=30,
                        dest='epochs',
                        help='No of epochs to run the system for')
    parser.add_argument('--train-val-ratio',
                        default=0.2,
                        dest='trainValRatio',
                        help='Ratio of split of train and validation data')

    return parser


def path_reader(path):

    if path is None:
        return None

    fileList = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    return fileList


def visualize_from_file(filePath):

    points = []
    with open(filePath, "r") as file:
        for line in file:
            tokens = line.split()
            points.append(list(map(float, tokens)))
    visualize_point_cloud(points)


def visualize_point_cloud(points):
    pcd = open3d.geometry.PointCloud()
    cord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[-1, -1, -1])
    pcd.points = open3d.utility.Vector3dVector(np.array(points))
    open3d.visualization.draw_geometries([pcd, cord])

def visualize_histogram(grid):
    grid = np.squeeze(grid, -1)
    indexX, indexY, indexZ = np.where(grid > 4)
    pointsX = (indexX - (32 / 2)) / (32 / 2) + (1 / 32)
    pointsY = (indexY - (32 / 2)) / (32 / 2) + (1 / 32)
    pointsZ = (indexZ - (32 / 2)) / (32 / 2) + (1 / 32)
    pointCloud = np.stack([pointsX, pointsY, pointsZ], axis=1)
    visualize_point_cloud(pointCloud)
    grid = np.reshape(grid, [-1,])
    _ = plt.hist(grid, bins='auto')
    plt.show()

