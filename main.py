#############################
### main executor file
###########################

from __future__ import absolute_import, division, print_function, unicode_literals
from Source.utilities import arg_creator
from Source.ThreeDEdgeDetect import ThreeDEdgeDetector
from Source.ThreeDEdgeDetectExp1 import ThreeDEdgeDetectorExp1
from Source.AutoEncoder import AutoEncoder


if __name__ == "__main__":
    arguments = arg_creator()
    parsed_args = arguments.parse_args()
    # detector = AutoEncoder()
    detector = ThreeDEdgeDetector()
    # detector = ThreeDEdgeDetectorExp1()
    detector.run()