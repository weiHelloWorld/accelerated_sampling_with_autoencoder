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
args = parser.parse_args()

my_coor_data_obj = coordinates_data_files_list(
    list_of_dir_of_coor_data_files=['../target/' + CONFIG_30])
my_file_list = my_coor_data_obj.get_list_of_coor_data_files()
if CONFIG_48 == 'cossin':
    data_set = molecule_type.get_many_cossin_from_coordinates_in_list_of_files(
        my_file_list, step_interval=args.training_interval)
    output_data_set = None
    fraction_of_data_to_be_saved = 1
elif CONFIG_48 == 'Cartesian':
    coor_data_obj_input = my_coor_data_obj.create_sub_coor_data_files_list_using_filter_conditional(lambda x: not 'aligned' in x)
    alignment_coor_file_suffix_list = CONFIG_61
    num_of_copies = args.num_of_copies
    fraction_of_data_to_be_saved = 1.0 / num_of_copies
    data_set, output_data_set = Sutils.prepare_training_data_using_Cartesian_coordinates_with_data_augmentation(
        ['../target/' + CONFIG_30], alignment_coor_file_suffix_list, CONFIG_49, num_of_copies,
        molecule_type,
        use_representative_points_for_training=CONFIG_58
    )
    mixed_error_function = CONFIG_71
    if mixed_error_function:
        output_data_set_1 = Sutils.remove_translation(output_data_set[:, list(range(9 * 1, 9 * 8))])  # mixed_err
        output_data_set_2 = Sutils.remove_translation(output_data_set[:, list(range(180, 360))])
        output_data_set = np.concatenate([3.0 * output_data_set_1, output_data_set_2], axis=1)

    data_set = data_set[::args.training_interval]
    output_data_set = output_data_set[::args.training_interval]
else:
    raise Exception('error input data type')

max_FVE = 0
current_network = None

for _ in range(args.num_of_trainings):
    if CONFIG_45 == 'pybrain':
        temp_network = neural_network_for_simulation(index=args.index,
                                                     data_set_for_training=data_set,
                                                     training_data_interval=1,
                                                     )
    elif CONFIG_45 == 'keras':
        temp_network = autoencoder_Keras(index=args.index,
                                         data_set_for_training=data_set,
                                         output_data_set=output_data_set,
                                         training_data_interval=1,
                                         )
    else:
        raise Exception ('this training backend not implemented')

    temp_network.train()
    print("temp FVE = %f" % (temp_network.get_fraction_of_variance_explained()))
    if temp_network.get_fraction_of_variance_explained() > max_FVE:
        max_FVE = temp_network.get_fraction_of_variance_explained()
        print("max_FVE = %f" % max_FVE)
        assert (max_FVE > 0)
        current_network = copy.deepcopy(temp_network)

assert (isinstance(current_network, autoencoder))
current_network.save_into_file(fraction_of_data_to_be_saved=fraction_of_data_to_be_saved)
print current_network._filename_to_save_network
