###############################
# Custom 3D canny edge filter layer
###############################

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.keras as tfk


class Canny3DEdge(tfk.layers.Layer):
    def __init__(self, ):