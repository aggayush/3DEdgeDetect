######################################
## utlities function for the code
########################################

import argparse
from os import listdir
from os.path import isfile, join


def arg_creator():
    parser = argparse.ArgumentParser(description='Arguments for the edge detector')
    parser.add_argument(name='--train-data-path',
                        default='./Data/Train/',
                        dest='trainDataPath',
                        help='Path to the training data files')
    parser.add_argument(name='--test-data-path',
                        default='./Data/Test/',
                        dest='testDataPath',
                        help='Path to the testing data files')
    parser.add_argument(name='--output-path',
                        default='./Output/',
                        dest='outputPath',
                        help='Path to the output folder')
    parser.add_argument(name='--model-path',
                        default='./Model/',
                        dest='modelPath',
                        help='Path to the model save folder (same for loading pretrained model)')
    parser.add_argument(name='--is-train',
                        action='store_true',
                        dest='isTrain',
                        help='To run training or testing')
    parser.add_argument(name='--is-streamed',
                        action='store_true',
                        dest='isStreamed',
                        help='if data is streamed input or to be read from file')

    return parser


def path_reader(path):

    if path is None:
        return None

    fileList = [f for f in listdir(path) if isfile(join(path,f))]

    return fileList