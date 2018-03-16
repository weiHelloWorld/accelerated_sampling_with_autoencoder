"""used for backward compatibility"""
try:
    from pybrain.structure import *
    from pybrain.structure.modules.circularlayer import *
    layer_type_to_name_mapping = {TanhLayer: "Tanh", CircularLayer: "Circular", LinearLayer: "Linear", ReluLayer: "Relu"}
except ImportError as my_import_e:
    print my_import_e
    pass
