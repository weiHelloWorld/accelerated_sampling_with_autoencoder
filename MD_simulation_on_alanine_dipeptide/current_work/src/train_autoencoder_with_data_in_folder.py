# train autoencoder with data in a folder

from ANN_simulation import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("molecule_type", type=str, help="molecule type")
parser.add_argument("data_folder", type=str, help="folder that contains coordinates data")
parser.add_argument("autoencoder_file_name", type=str, help="name of autoencoder file")
parser.add_argument("--step_interval", type=int, default=1, help="step interval of data for training (default = 1)")
parser.add_argument("--max_num_of_training", type=int, default=100, help="max num of training")
parser.add_argument("--num_of_input_nodes", type=int, default=None, help="number of input nodes")
parser.add_argument("--num_of_hidden_nodes", type=int, default=15, help="number of hidden nodes")
parser.add_argument("--num_of_PCs", type=int, default=2, help="number of PCs")
parser.add_argument("--PC_layer_type", type=str, default='TanhLayer', help='PC layer type')
parser.add_argument("--learning_rate", type=float, default=0.002, help='learning rate')
parser.add_argument("--momentum", type=float, default=0.4, help= "momentum (the ratio by which the gradient of the last timestep is used)")
parser.add_argument("--weightdecay", type=float, default=0.1, help="weight decay")
parser.add_argument("--verbose", help="whether to print training info", action="store_true")
args = parser.parse_args()

molecule_type = Sutils.create_subclass_instance_using_name(args.molecule_type)
if args.num_of_input_nodes is None:
    if isinstance(molecule_type, Alanine_dipeptide):
        num_of_input_nodes = 8
    elif isinstance(molecule_type, Trp_cage):
        num_of_input_nodes = 76
    else:
        raise Exception('error')
else:
    num_of_input_nodes = args.num_of_input_nodes

if args.PC_layer_type == "TanhLayer":
    PC_layer_type = TanhLayer
    num_of_PCs = args.num_of_PCs
elif args.PC_layer_type == "CircularLayer":
    PC_layer_type = CircularLayer
    num_of_PCs = 2 * args.num_of_PCs
else:
    raise Exception("PC_layer_type not defined")

my_file_list = coordinates_data_files_list([args.data_folder])._list_of_coor_data_files
if isinstance(molecule_type, Alanine_dipeptide):
    data = molecule_type.get_many_cossin_from_coordiantes_in_list_of_files(my_file_list)
elif isinstance(molecule_type, Trp_cage):
    data = molecule_type.get_many_cossin_from_coordiantes_in_list_of_files(my_file_list,step_interval=args.step_interval)
else:
    raise Exception("molecule type not defined")

a = neural_network_for_simulation(index=1447,
                                  training_data_interval=1,
                                  data_set_for_training=data,
                                  node_num=[num_of_input_nodes, args.num_of_hidden_nodes, num_of_PCs, args.num_of_hidden_nodes, num_of_input_nodes],
                                  max_num_of_training=args.max_num_of_training,
                                  network_parameters=[args.learning_rate, args.momentum, args.weightdecay, 1],
                                  hidden_layers_types=[TanhLayer, PC_layer_type, TanhLayer],
                                  network_verbose = args.verbose
                                  )

a.train().save_into_file(args.autoencoder_file_name)
print a.get_fraction_of_variance_explained()
