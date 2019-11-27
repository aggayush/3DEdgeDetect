################################
# Main class file to execute the code
################################


import open3d
import tensorflow as tf
from Source.utilities import path_reader


class ThreeDEdgeDetector:

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

    def prepare_data(self):

        if self.isTrain:
            path = self.trainDataPath
        elif self.isStreamed:
            path = None
        else:
            path = self.testDataPath

        fileList = path_reader(path)

        print(fileList)


     def run(self):

         self.prepare_data()
        
