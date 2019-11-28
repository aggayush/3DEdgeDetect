######################################
## utlities function for the code
########################################

import argparse
from os import listdir
from os.path import isfile, join
import open3d
import numpy as np


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
                        default=1,
                        dest='batchSize',
                        help='batch Size to process')
    parser.add_argument('--shuffle-buffer-size',
                        default=1000,
                        dest='shuffleBufferSize',
                        help='Data shuffle buffer size')

    return parser


def path_reader(path):

    if path is None:
        return None

    fileList = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    return fileList

def visualize(filePath):

    points = []
    with open(filePath, "r") as file:
        for line in file:
            tokens = line.split()
            points.append(list(map(float, tokens)))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(points))
    open3d.visualization.draw_geometries([pcd])