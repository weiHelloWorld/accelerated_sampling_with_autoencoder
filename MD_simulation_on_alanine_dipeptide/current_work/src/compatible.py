"""used for backward compatibility"""
from pybrain.structure import *
from pybrain.structure.modules.circularlayer import *

layer_type_to_name_mapping = {TanhLayer: "Tanh", CircularLayer: "Circular", LinearLayer: "Linear", ReluLayer: "Relu"}
