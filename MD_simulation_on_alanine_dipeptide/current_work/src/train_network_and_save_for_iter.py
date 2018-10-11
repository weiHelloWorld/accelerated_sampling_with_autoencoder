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
parser.add_argument("--output_file", type=str, default=None, help="file name to save autoencoder")
parser.add_argument('--data_folder', type=str, default=None, help="folder containing training data")
parser.add_argument('--in_data', type=str, default=None, help="npy file containing pre-computed input data")
parser.add_argument('--out_data', type=str, default=None, help="npy file containing pre-computed output data, if in_data is not None while out_data is None, then out_data is set to be in_data")
parser.add_argument('--node_num', type=str, default=None, help="node number")
parser.add_argument('--batch_size', type=int, default=None, help='batch size')
parser.add_argument('--auto_dim', type=int, default=CONFIG_79, help="automatically determine input/output dim based on data")
parser.add_argument('--auto_scale', type=int, default=False, help="automatically scale inputs and outputs")
parser.add_argument('--save_to_data_files', type=str, default=None, help="save training data to external files if it is not None, example: 'temp_in.npy,temp_out.npy' ")
parser.add_argument('--lag_time', type=int, default=0, help='lag time for time lagged autoencoder')
parser.add_argument('--rec_loss_type', type=int, default=True, help='0: standard rec loss, 1: lagged rec loss, 2: no rec loss (pytorch only)')
parser.add_argument('--rec_weight', type=float, default=1.0, help='weight of reconstruction loss (pytorch only)')
parser.add_argument('--autocorr_weight', type=float, default=1.0, help='weight of autocorrelation loss in the loss function (pytorch only)')
parser.add_argument('--pearson_weight', type=float, default=None, help='weight of pearson loss (pytorch only)')
parser.add_argument('--previous_CVs', type=str, default=None, help='npy file containing previous CVs to compute pearson loss (pytorch only)')
parser.add_argument('--sf', type=str, default=None, help='model to start with (pytorch only)')
args = parser.parse_args()

def get_data_from_folder(temp_folder, input_type, output_type):
    my_coor_data_obj = coordinates_data_files_list(
        list_of_dir_of_coor_data_files=[temp_folder])
    coor_data_obj_input = my_coor_data_obj.create_sub_coor_data_files_list_using_filter_conditional(
        lambda x: not 'aligned' in x)
    if input_type == 'cossin':
        data_set = np.array(molecule_type.get_many_cossin_from_coordinates_in_list_of_files(
            coor_data_obj_input.get_list_of_coor_data_files(), step_interval=args.training_interval))
    elif input_type == 'Cartesian':
        scaling_factor = CONFIG_49
        data_set = coor_data_obj_input.get_coor_data(scaling_factor)
        data_set = data_set[::args.training_interval]
        data_set = Sutils.remove_translation(data_set)
        assert (Sutils.check_center_of_mass_is_at_origin(data_set))
    elif input_type == 'pairwise_distance':
        data_set = np.array(Sutils.get_non_repeated_pairwise_distance(
            coor_data_obj_input.get_list_of_corresponding_pdb_files(), step_interval=args.training_interval,
            atom_selection=CONFIG_73)) / CONFIG_49
    else:
        raise Exception('error input type')

    if output_type == 'cossin':
        output_data_set = np.array(molecule_type.get_many_cossin_from_coordinates_in_list_of_files(
            coor_data_obj_input.get_list_of_coor_data_files(), step_interval=args.training_interval))
    elif output_type == 'Cartesian':
        scaling_factor = CONFIG_49
        alignment_coor_file_suffix_list = CONFIG_61
        output_data_set = Sutils.prepare_output_Cartesian_coor_with_multiple_ref_structures(
            [temp_folder], alignment_coor_file_suffix_list, scaling_factor)
        output_data_set = output_data_set[::args.training_interval]
        mixed_error_function = CONFIG_71  # TODO: refactor this part later
        if mixed_error_function:
            if CONFIG_30 == "Trp_cage":
                output_data_set_1 = Sutils.remove_translation(
                    output_data_set[:, list(range(9 * 1, 9 * 8))])  # mixed_err
                output_data_set_2 = Sutils.remove_translation(output_data_set[:, list(range(180, 360))])
                output_data_set = np.concatenate([4.0 * output_data_set_1, output_data_set_2],
                                                 axis=1)  # TODO: may modify this relative weight later
            elif CONFIG_30 == "Src_kinase":
                output_data_set_1 = Sutils.remove_translation(
                    output_data_set[:, list(range(9 * 143, 9 * 170))])  # mixed_err
                output_data_set_2 = Sutils.remove_translation(
                    output_data_set[:, list(range(2358 + 9 * 43, 2358 + 9 * 58))])
                output_data_set = np.concatenate([output_data_set_1, output_data_set_2], axis=1)
        assert (Sutils.check_center_of_mass_is_at_origin(output_data_set))
    elif output_type == 'pairwise_distance':
        output_data_set = np.array(Sutils.get_non_repeated_pairwise_distance(
            coor_data_obj_input.get_list_of_corresponding_pdb_files(), step_interval=args.training_interval,
            atom_selection=CONFIG_73)) / CONFIG_49
    elif output_type == 'combined':
        scaling_factor = CONFIG_49
        alignment_coor_file_suffix_list = CONFIG_61
        output_data_set = Sutils.prepare_output_Cartesian_coor_with_multiple_ref_structures(
            [temp_folder], alignment_coor_file_suffix_list, scaling_factor)
        output_data_set = output_data_set[::args.training_interval]
        mixed_error_function = CONFIG_71  # TODO: refactor this part later
        assert mixed_error_function  # mixed error is required
        if CONFIG_30 == "Trp_cage":
            output_data_set_1 = Sutils.remove_translation(output_data_set[:, list(range(9 * 1, 9 * 8))])  # mixed_err
            output_data_set_2 = Sutils.remove_translation(output_data_set[:, list(range(180, 360))])
            output_data_set = np.concatenate([4.0 * output_data_set_1, output_data_set_2],
                                             axis=1)  # TODO: may modify this relative weight later
        else:
            raise Exception('not defined')
        temp_output_data_set = np.array(Sutils.get_non_repeated_pairwise_distance(
            coor_data_obj_input.get_list_of_corresponding_pdb_files(), step_interval=args.training_interval,
            atom_selection=CONFIG_73)) / CONFIG_49
        output_data_set = np.concatenate([output_data_set, temp_output_data_set], axis=1)
    else:
        raise Exception('error output data type')
    return data_set, output_data_set

# used to process additional arguments
additional_argument_list = {}
if not args.output_file is None:
    additional_argument_list['filename_to_save_network'] = args.output_file
if not args.lr_m is None:
    temp_lr = float(args.lr_m.strip().split(',')[0])
    temp_momentum = float(args.lr_m.strip().split(',')[1])
    additional_argument_list['network_parameters'] = [temp_lr, temp_momentum, 0, True, CONFIG_4[4]]
if not args.batch_size is None:
    additional_argument_list['batch_size'] = args.batch_size

if args.data_folder is None:
    args.data_folder = '../target/' + CONFIG_30

fraction_of_data_to_be_saved = 1   # save all training data by default
input_data_type, output_data_type = CONFIG_48, CONFIG_76

# getting training data
if not args.in_data is None:
    data_set = np.load(args.in_data)
    if args.out_data is None:
        output_data_set = data_set
    else:
        output_data_set = np.load(args.out_data)
else:
    data_set, output_data_set = get_data_from_folder(args.data_folder, input_data_type, output_data_type)

assert (len(data_set) == len(output_data_set))
use_representative_points_for_training = CONFIG_58
if use_representative_points_for_training:
    data_set, output_data_set = Sutils.select_representative_points(data_set, output_data_set)
    
if input_data_type == 'Cartesian' and args.in_data is None:
    print('applying data augmentation...')
    data_set, output_data_set = Sutils.data_augmentation(data_set, output_data_set, args.num_of_copies,
                             is_output_reconstructed_Cartesian=(output_data_type == 'Cartesian'))
    fraction_of_data_to_be_saved = 1.0 / args.num_of_copies
else:
    print("data augmentation not applied")

scaling_factor_for_expected_output = CONFIG_75  # TODO: is this useful?
if not scaling_factor_for_expected_output is None:
    print("expected output is weighted by %s" % str(scaling_factor_for_expected_output))
    output_data_set = np.dot(output_data_set, np.diag(scaling_factor_for_expected_output))

if args.node_num is None:
    temp_node_num = CONFIG_3[:]       # deep copy list
else:
    temp_node_num = [int(item) for item in args.node_num.split(',')]

if args.auto_dim: temp_node_num[0], temp_node_num[-1] = data_set.shape[1], output_data_set.shape[1]
additional_argument_list['node_num'] = temp_node_num

if args.auto_scale:
    auto_scaling_factor = np.max(np.abs(data_set)).astype(np.float)
    print(("auto_scaling_factor = %f" % auto_scaling_factor))
    data_set /= auto_scaling_factor
    output_data_set /= (np.max(np.abs(output_data_set)).astype(np.float))
    assert np.max(np.abs(data_set)) == 1.0 and np.max(np.abs(output_data_set)) == 1.0

print(("min/max of output = %f, %f, min/max of input = %f, %f" % (np.min(output_data_set), np.max(output_data_set),
                                                                  np.min(data_set), np.max(data_set))))

if not args.save_to_data_files is None:
    args.save_to_data_files = args.save_to_data_files.split(',')

if CONFIG_45 == 'keras':
    temp_network_list = [autoencoder_Keras(index=args.index,
                                         data_set_for_training=data_set,
                                         output_data_set=output_data_set,
                                         data_files=args.save_to_data_files,
                                         **additional_argument_list
                                         ) for _ in range(args.num_of_trainings)]
elif CONFIG_45 == 'pytorch':
    additional_argument_list['rec_loss_type'] = args.rec_loss_type
    additional_argument_list['start_from'] = args.sf
    additional_argument_list['rec_weight'] = args.rec_weight
    additional_argument_list['autocorr_weight'] = args.autocorr_weight
    additional_argument_list['pearson_weight'] = args.pearson_weight
    if not args.previous_CVs is None:
        additional_argument_list['previous_CVs'] = np.load(args.previous_CVs)
    temp_network_list = [autoencoder_torch(index=args.index,
                                           data_set_for_training=data_set,
                                           output_data_set=output_data_set,
                                           data_files=args.save_to_data_files,
                                           **additional_argument_list
                                           ) for _ in range(args.num_of_trainings)]
else:
    raise Exception ('this training backend not implemented')

for item in temp_network_list: item.train(lag_time=args.lag_time)

if len(temp_network_list) == 1:
    best_network = temp_network_list[0]
    # if np.all(np.isnan(best_network.get_PCs())):
    #     best_network = None
else:
    temp_FVE_list = [item.get_fraction_of_variance_explained() for item in temp_network_list]
    max_FVE = np.max(temp_FVE_list)
    print('temp_FVE_list = %s, max_FVE = %f' % (str(temp_FVE_list), max_FVE))
    best_network = temp_network_list[temp_FVE_list.index(max_FVE)]
    assert (isinstance(best_network, autoencoder))
    assert (best_network.get_fraction_of_variance_explained() == max_FVE)

best_network.save_into_file(fraction_of_data_to_be_saved=fraction_of_data_to_be_saved)
print("excited! this is the name of best network: %s" % best_network._filename_to_save_network)  # this line is used to locate file name of neural network
