"""train autoencoder and save into file
this file is typically used for running training in an iteration
"""

from ANN_simulation import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int, help="index of autoencoder")
parser.add_argument("--training_interval", type=int, default=1, help="training interval")
parser.add_argument("--num_of_trainings", type=int, default=CONFIG_13, help="total number of trainings (and pick the best one to save)")
parser.add_argument("--num_of_copies", type=int, default=CONFIG_52, help="num of copies for data augmentation")
parser.add_argument("--lr_m", type=str, default=None, help="learning rate and momentum")
parser.add_argument("--num_PCs", type=int, default=None, help="number of PCs")
parser.add_argument("--output_file", type=str, default=None, help="file name to save autoencoder")
parser.add_argument('--data_folder', type=str, default=None, help="folder containing training data")
args = parser.parse_args()

# used to process additional arguments
additional_argument_list = {}
if not args.output_file is None:
    additional_argument_list['filename_to_save_network'] = args.output_file
if not args.lr_m is None:
    temp_lr = float(args.lr_m.strip().split(',')[0])
    temp_momentum = float(args.lr_m.strip().split(',')[1])
    additional_argument_list['network_parameters'] = [temp_lr, temp_momentum, 0, True, [0.00, 0.0000, 0.00, 0.00]]
if not args.num_PCs is None:
    temp_node_num = CONFIG_3[:]  # deep copy list
    temp_node_num[2] = args.num_PCs
    additional_argument_list['node_num'] = temp_node_num

num_of_copies = args.num_of_copies
if not args.data_folder is None:
    temp_list_of_directories_contanining_data = [args.data_folder]
else:
    temp_list_of_directories_contanining_data = ['../target/' + CONFIG_30]

my_coor_data_obj = coordinates_data_files_list(
    list_of_dir_of_coor_data_files=temp_list_of_directories_contanining_data)
my_file_list = my_coor_data_obj.get_list_of_coor_data_files()

fraction_of_data_to_be_saved = 1   # save all training data by default
input_data_type, output_data_type = CONFIG_48, CONFIG_76

# getting input data
if input_data_type == 'cossin' and output_data_type == 'cossin':  # input type
    data_set = np.array(molecule_type.get_many_cossin_from_coordinates_in_list_of_files(
        my_file_list, step_interval=args.training_interval))
elif input_data_type == 'Cartesian':
    coor_data_obj_input = my_coor_data_obj.create_sub_coor_data_files_list_using_filter_conditional(lambda x: not 'aligned' in x)
    scaling_factor = CONFIG_49
    data_set = coor_data_obj_input.get_coor_data(scaling_factor)
    data_set = data_set[::args.training_interval]
    data_set = Sutils.remove_translation(data_set)
    assert (Sutils.check_center_of_mass_is_at_origin(data_set))
else:
    raise Exception('error input type')

# getting output data
if output_data_type == 'cossin':   # output type
    output_data_set = data_set   # done above
elif output_data_type == 'Cartesian':
    scaling_factor = CONFIG_49
    alignment_coor_file_suffix_list = CONFIG_61
    output_data_set = Sutils.prepare_output_Cartesian_coor_with_multiple_ref_structures(
        temp_list_of_directories_contanining_data, alignment_coor_file_suffix_list, scaling_factor)
    output_data_set = output_data_set[::args.training_interval]
    mixed_error_function = CONFIG_71    # TODO: refactor this part later
    if mixed_error_function:
        if CONFIG_30 == "Trp_cage":
            output_data_set_1 = Sutils.remove_translation(output_data_set[:, list(range(9 * 1, 9 * 8))])  # mixed_err
            output_data_set_2 = Sutils.remove_translation(output_data_set[:, list(range(180, 360))])
            output_data_set = np.concatenate([3.0 * output_data_set_1, output_data_set_2],
                                             axis=1)  # TODO: may modify this relative weight later
        elif CONFIG_30 == "Src_kinase":
            output_data_set_1 = Sutils.remove_translation(
                output_data_set[:, list(range(9 * 143, 9 * 170))])  # mixed_err
            output_data_set_2 = Sutils.remove_translation(output_data_set[:, list(range(2358 + 9 * 43, 2358 + 9 * 58))])
            output_data_set = np.concatenate([output_data_set_1, output_data_set_2], axis=1)
    assert (Sutils.check_center_of_mass_is_at_origin(output_data_set))
elif output_data_type == 'pairwise_distance':
    coor_data_obj_input = my_coor_data_obj.create_sub_coor_data_files_list_using_filter_conditional(
        lambda x: not 'aligned' in x)
    output_data_set = np.array(Sutils.get_non_repeated_pairwise_distance(
        coor_data_obj_input.get_list_of_corresponding_pdb_files(), step_interval=args.training_interval,
        atom_selection=CONFIG_73)) / CONFIG_49 / 2.0  # TODO: may need better scaling factor?
else:
    raise Exception('error output data type')

print ("min/max of output = %f, %f" % (np.min(output_data_set), np.max(output_data_set)))
assert (len(data_set) == len(output_data_set))
use_representative_points_for_training = CONFIG_58
if use_representative_points_for_training:
    data_set, output_data_set = Sutils.select_representative_points(data_set, output_data_set)
    
if input_data_type == 'Cartesian':
    print 'applying data augmentation...'
    data_set, output_data_set = Sutils.data_augmentation(data_set, output_data_set, num_of_copies,
                             is_output_reconstructed_Cartesian=(output_data_type == 'Cartesian'))
else:
    print "data augmentation not applied"

scaling_factor_for_expected_output = CONFIG_75  # this is useful if we want to put more weights on some components in the output
if not scaling_factor_for_expected_output is None:
    print "expected output is weighted"
    output_data_set = np.dot(output_data_set, np.diag(scaling_factor_for_expected_output))

if CONFIG_45 == 'keras':
    temp_network_list = [autoencoder_Keras(index=args.index,
                                         data_set_for_training=data_set,
                                         output_data_set=output_data_set,
                                         training_data_interval=1,
                                         **additional_argument_list
                                         ) for _ in range(args.num_of_trainings)]
else:
    raise Exception ('this training backend not implemented')

for item in temp_network_list: item.train()

temp_FVE_list = [item.get_fraction_of_variance_explained() for item in temp_network_list]
max_FVE = np.max(temp_FVE_list)
print 'temp_FVE_list = %s, max_FVE = %f' % (str(temp_FVE_list), max_FVE)
best_network = temp_network_list[temp_FVE_list.index(max_FVE)]

assert (isinstance(best_network, autoencoder))
assert (best_network.get_fraction_of_variance_explained() == max_FVE)
best_network.save_into_file(fraction_of_data_to_be_saved=fraction_of_data_to_be_saved)
print best_network._filename_to_save_network
