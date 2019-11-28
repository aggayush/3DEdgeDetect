################################
# Main class file to execute the code
################################

import open3d
import tensorflow as tf
import numpy as np
from Source.utilities import path_reader, visualize


class ThreeDEdgeDetector:
    VOXEL_GRID_X = 256
    VOXEL_GRID_Y = 256
    VOXEL_GRID_Z = 256

    def __init__(self,args=None):
        if args is None:
            self.trainDataPath = ""
            self.testDataPath = ""
            self.outputPath = ""
            self.modelPath = ""
            self.isTrain = True
            self.isStreamed = False
        else:
            self.trainDataPath = args.trainDataPath
            self.testDataPath = args.testDataPath
            self.outputPath = args.outputPath
            self.modelPath = args.modelPath
            self.isTrain = args.isTrain
            self.isStreamed = args.isStreamed

        self.dataSize = 0

    # function to create the voxel grid and return
    def preprocessing(self, data):
        grid = np.zeros((self.VOXEL_GRID_X, self.VOXEL_GRID_Y, self.VOXEL_GRID_Z), dtype=np.bool)

        #Normalizing in unit sphere
        centroid = np.mean(data, axis=0)
        data = data - centroid
        furthestDistanceFromCentre = np.max(np.sqrt(np.sum(abs(data)**2, axis=-1)))
        data = data/furthestDistanceFromCentre

        #visualize(data)

        #Voxelization of data
        gridX = np.floor(data[:, 0]*(self.VOXEL_GRID_X/2) + self.VOXEL_GRID_X/2)
        gridY = np.floor(data[:, 1]*(self.VOXEL_GRID_X/2) + self.VOXEL_GRID_X/2)
        gridZ = np.floor(data[:, 2]*(self.VOXEL_GRID_X/2) + self.VOXEL_GRID_X/2)

        indexes = (tuple(gridX.astype(int)), tuple(gridY.astype(int)), tuple(gridZ.astype(int)))

        grid[indexes] = True

        # visualization
        # gridX = (gridX - self.VOXEL_GRID_X/2)/(self.VOXEL_GRID_X/2)
        # gridY = (gridY - self.VOXEL_GRID_Y/2)/(self.VOXEL_GRID_Y/2)
        # gridZ = (gridZ - self.VOXEL_GRID_X/2)/(self.VOXEL_GRID_Z/2)
        # points = np.stack((gridX, gridY, gridZ), axis=1)
        # points = np.unique(points, axis=0)
        # visualize(points)

        return grid

    def prepare_data(self):

        if self.isTrain:
            path = self.trainDataPath
        elif self.isStreamed:
            path = None
        else:
            path = self.testDataPath

        fileList = path_reader(path)

        self.dataSize = len(fileList)
        data = open3d.io.read_point_cloud(fileList[0])
        data = np.asarray(data.points)
        data = self.preprocessing(data)

    def run(self):

        self.prepare_data()

