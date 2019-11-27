#############################
### main executor file
###########################


from Source.utilities import arg_creater
from ThreeDEdgeDetect import ThreeDEdgeDetector


if __name__ == "__main__":
    arguments = arg_creater()
    parsed_args = arguments.parse_args()
    detector = ThreeDEdgeDetector(parsed_args)
