#############################
### main executor file
###########################


from Source.utilities import arg_creator
from Source.ThreeDEdgeDetect import ThreeDEdgeDetector


if __name__ == "__main__":
    arguments = arg_creator()
    parsed_args = arguments.parse_args()
    detector = ThreeDEdgeDetector(parsed_args)
    detector.run()