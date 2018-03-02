from config import *
from molecule_spec_sutils import *  # import molecule specific unitity code
from coordinates_data_files_list import *
from sklearn.cluster import KMeans
from keras.models import Sequential, Model, load_model
from keras.optimizers import *
from keras.layers import Dense, Activation, Lambda, Reshape, Input
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras import layers
from keras import backend as K
import random

##################    set types of molecules  ############################

molecule_type = Sutils.create_subclass_instance_using_name(CONFIG_30)

##########################################################################


class autoencoder(object):
    """the neural network for simulation
    this class includes abstract methods, that must be implemented by subclasses
    """
    def __init__(self,
                 index,  # the index of the current network
                 data_set_for_training,
                 output_data_set = None,  # output data may not be the same with the input data
                 autoencoder_info_file=None,  # this might be expressions, or coefficients
                 training_data_interval=CONFIG_2,
                 in_layer_type=LinearLayer,
                 hidden_layers_types=CONFIG_17,
                 out_layer_type=CONFIG_78,  # different layers
                 node_num=CONFIG_3,  # the structure of ANN
                 max_num_of_training=CONFIG_5,
                 filename_to_save_network=CONFIG_6,
                 hierarchical=CONFIG_44,
                 network_verbose=CONFIG_46,
                 output_as_circular=CONFIG_47,
                 *args, **kwargs           # for extra init functions for subclasses
                 ):

        self._index = index
        self._data_set = data_set_for_training
        self._output_data_set = output_data_set
        self._training_data_interval = training_data_interval
        if autoencoder_info_file is None:
            self._autoencoder_info_file = "../resources/%s/autoencoder_info_%d.txt" % (CONFIG_30, index)
        else:
            self._autoencoder_info_file = autoencoder_info_file

        if not in_layer_type is None: self._in_layer_type = in_layer_type
        if not hidden_layers_types is None: self._hidden_layers_type = hidden_layers_types
        if not out_layer_type is None: self._out_layer_type = out_layer_type

        self._node_num = node_num
        self._max_num_of_training = max_num_of_training
        if filename_to_save_network is None:
            self._filename_to_save_network = "../resources/%s/network_%s.pkl" % (
            CONFIG_30, str(self._index))  # by default naming with its index
        else:
            self._filename_to_save_network = filename_to_save_network

        self._hierarchical = hierarchical
        self._network_verbose = network_verbose
        num_of_PC_nodes_for_each_PC = 2 if self._hidden_layers_type[1] == CircularLayer else 1
        self._num_of_PCs = self._node_num[2] / num_of_PC_nodes_for_each_PC
        self._connection_between_layers_coeffs = None
        self._connection_with_bias_layers_coeffs = None
        self._molecule_net_layers = self._molecule_net = self._encoder_net = self._decoder_net = None
        self._output_as_circular = output_as_circular
        self._init_extra(*args, **kwargs)
        return

    @abc.abstractmethod
    def _init_extra(self):
        """must be implemented by subclasses"""
        pass

    @staticmethod
    def load_from_pkl_file(filename):
        a = Sutils.load_object_from_pkl_file(filename)
        if os.path.isfile(filename.replace('.pkl','.hdf5')):
            a._molecule_net = load_model(filename.replace('.pkl','.hdf5'),custom_objects={'mse_weighted': mse_weighted})
            a._molecule_net_layers = a._molecule_net.layers
        elif not hasattr(a, '_molecule_net') and hasattr(a, '_molecule_net_layers'):  # for backward compatibility
            a._molecule_net = Sequential()
            for item in a._molecule_net_layers:
                a._molecule_net.add(item)
        else:
            raise Exception('cannot load attribute _molecule_net')
        if os.path.isfile(filename.replace('.pkl', '_encoder.hdf5')):
            a._encoder_net = load_model(filename.replace('.pkl', '_encoder.hdf5'),custom_objects={'mse_weighted': mse_weighted})
        else:
            raise Exception('TODO: construct encoder from _molecule_net') # TODO
        return a

    def save_into_file(self, filename=CONFIG_6, fraction_of_data_to_be_saved = 1.0):
        if filename is None:
            filename = self._filename_to_save_network

        if fraction_of_data_to_be_saved != 1.0:
            number_of_data_points_to_be_saved = int(self._data_set.shape[0] * fraction_of_data_to_be_saved)
            print ("Warning: only %f of data (%d out of %d) are saved into pkl file" % (fraction_of_data_to_be_saved,
                                                                                        number_of_data_points_to_be_saved,
                                                                                        self._data_set.shape[0]))
            self._data_set = self._data_set[:number_of_data_points_to_be_saved]
            if not self._output_data_set is None:        # for backward compatibility
                self._output_data_set = self._output_data_set[:number_of_data_points_to_be_saved]

        hdf5_file_name = filename.replace('.pkl', '.hdf5')
        hdf5_file_name_encoder = hdf5_file_name.replace('.hdf5', '_encoder.hdf5')
        hdf5_file_name_decoder = hdf5_file_name.replace('.hdf5', '_decoder.hdf5')
        for item_filename in [filename, hdf5_file_name, hdf5_file_name_encoder, hdf5_file_name_decoder]:
            Helper_func.backup_rename_file_if_exists(item_filename)
        self._molecule_net.save(hdf5_file_name)
        self._encoder_net.save(hdf5_file_name_encoder)
        if not self._decoder_net is None: self._decoder_net.save(hdf5_file_name_decoder)
        self._molecule_net = self._molecule_net_layers = self._encoder_net = self._decoder_net = None  # we save model in hdf5, not in pkl
        with open(filename, 'wb') as my_file:
            pickle.dump(self, my_file, pickle.HIGHEST_PROTOCOL)

        self._molecule_net = load_model(hdf5_file_name, custom_objects={'mse_weighted': mse_weighted})
        self._encoder_net = load_model(hdf5_file_name_encoder, custom_objects={'mse_weighted': mse_weighted})
        # self._decoder_net = load_model(hdf5_file_name_decoder, custom_objects={'mse_weighted': mse_weighted})
        self._molecule_net_layers = self._molecule_net.layers
        return

    def get_expression_of_network(self):
        """
        this function generates expression of PCs in terms of inputs
        """
        # FIXME: 1. the expression no longer works, since I made input list for input layer of autoencoder consistent
        # for both alanine dipeptide and trp-cage, always [cos, sin, cos, sin ....],
        # which is consistent with ANN_Force, instead of [cos, cos, cos, cos, sin, sin, sin, sin]
        # FIXME: 2. this does not support multi-hidden layer cases
        type_of_middle_hidden_layer = self._hidden_layers_type[1]

        node_num = self._node_num
        expression = ""

        # 1st part: network
        for i in range(2):
            expression = '\n' + expression
            mul_coef = self._connection_between_layers_coeffs[i].reshape(node_num[i + 1], node_num[i])
            bias_coef = self._connection_with_bias_layers_coeffs[i]

            for j in range(np.size(mul_coef, 0)):
                temp_expression = 'in_layer_%d_unit_%d = ' % (i + 1, j)

                for k in range(np.size(mul_coef, 1)):
                    temp_expression += ' %f * out_layer_%d_unit_%d +' % (mul_coef[j, k], i, k)

                temp_expression += ' %f;\n' % (bias_coef[j])
                expression = temp_expression + expression  # order of expressions matter in OpenMM

            if i == 1 and type_of_middle_hidden_layer == CircularLayer:
                for j in range(np.size(mul_coef, 0) / 2):
                    temp_expression = 'out_layer_%d_unit_%d = ( in_layer_%d_unit_%d ) / radius_of_circular_pair_%d;\n' % \
                                      (i + 1, 2 * j, i + 1, 2 * j, j)
                    temp_expression += 'out_layer_%d_unit_%d = ( in_layer_%d_unit_%d ) / radius_of_circular_pair_%d;\n' % \
                                       (i + 1, 2 * j + 1, i + 1, 2 * j + 1, j)
                    temp_expression += 'radius_of_circular_pair_%d = sqrt( in_layer_%d_unit_%d * in_layer_%d_unit_%d + in_layer_%d_unit_%d * in_layer_%d_unit_%d );\n' \
                                       % (j, i + 1, 2 * j, i + 1, 2 * j, i + 1, 2 * j + 1, i + 1, 2 * j + 1)

                    expression = temp_expression + expression
            else:
                for j in range(np.size(mul_coef, 0)):
                    temp_expression = 'out_layer_%d_unit_%d = tanh( in_layer_%d_unit_%d );\n' % (i + 1, j, i + 1, j)
                    expression = temp_expression + expression

        # 2nd part: relate PCs to network
        if type_of_middle_hidden_layer == CircularLayer:
            temp_expression = 'PC0 = acos( out_layer_2_unit_0 ) * ( step( out_layer_2_unit_1 ) - 0.5) * 2;\n'
            temp_expression += 'PC1 = acos( out_layer_2_unit_2 ) * ( step( out_layer_2_unit_3 ) - 0.5) * 2;\n'
            expression = temp_expression + expression
        elif type_of_middle_hidden_layer == TanhLayer:
            temp_expression = 'PC0 = out_layer_2_unit_0;\nPC1 = out_layer_2_unit_1;\n'
            expression = temp_expression + expression

        # 3rd part: definition of inputs
        expression += molecule_type.get_expression_for_input_of_this_molecule()

        return expression

    def write_expression_into_file(self, out_file=None):
        if out_file is None: out_file = self._autoencoder_info_file

        expression = self.get_expression_of_network()
        with open(out_file, 'w') as f_out:
            f_out.write(expression)
        return

    def get_expression_script_for_plumed(self, mode="native"):
        index_CV_layer = (len(self._node_num) - 1) / 2
        plumed_script = ''
        if mode == "native":  # using native implementation by PLUMED (using COMBINE and MATHEVAL)
            plumed_script += "bias_const: CONSTANT VALUE=1.0\n"  # used for bias
            activation_function_list = ['tanh'] * index_CV_layer
            for layer_index in range(1, index_CV_layer + 1):
                for item in range(self._node_num[layer_index]):
                    plumed_script += "l_%d_in_%d: COMBINE PERIODIC=NO COEFFICIENTS=" % (layer_index, item)
                    plumed_script += "%s" % \
                                     str(self._connection_between_layers_coeffs[layer_index - 1][
                                         item * self._node_num[layer_index - 1]:(item + 1) * self._node_num[
                                             layer_index - 1]].tolist())[1:-1].replace(' ', '')
                    plumed_script += ',%f' % self._connection_with_bias_layers_coeffs[layer_index - 1][item]
                    plumed_script += " ARG="
                    for _1 in range(self._node_num[layer_index - 1]):
                        plumed_script += 'l_%d_out_%d,' % (layer_index - 1, _1)

                    plumed_script += 'bias_const\n'
                    plumed_script += 'l_%d_out_%d: MATHEVAL ARG=l_%d_in_%d FUNC=%s(x) PERIODIC=NO\n' % (
                        layer_index, item, layer_index,item, activation_function_list[layer_index - 1])
        elif mode == "ANN":  # using ANN class
            temp_num_of_layers_used = index_CV_layer + 1
            temp_input_string = ','.join(['l_0_out_%d' % item for item in range(self._node_num[0])])
            temp_num_nodes_string = ','.join([str(item) for item in self._node_num[:temp_num_of_layers_used]])
            temp_layer_type_string = map(lambda x: layer_type_to_name_mapping[x], CONFIG_17[:2])
            temp_layer_type_string = ','.join(temp_layer_type_string)
            temp_coeff_string = ''
            temp_bias_string = ''
            for _1, item_coeff in enumerate(self._connection_between_layers_coeffs[:temp_num_of_layers_used - 1]):
                temp_coeff_string += ' COEFFICIENTS_OF_CONNECTIONS%d=%s' % \
                                     (_1, ','.join([str(item) for item in item_coeff]))
            for _1, item_bias in enumerate(self._connection_with_bias_layers_coeffs[:temp_num_of_layers_used - 1]):
                temp_bias_string += ' VALUES_OF_BIASED_NODES%d=%s' % \
                                     (_1, ','.join([str(item) for item in item_bias]))

            plumed_script += "ann_force: ANN ARG=%s NUM_OF_NODES=%s LAYER_TYPES=%s %s %s" % \
                (temp_input_string, temp_num_nodes_string, temp_layer_type_string,
                 temp_coeff_string, temp_bias_string)
        else:
            raise Exception("mode error")
        return plumed_script

    def write_expression_script_for_plumed(self, out_file=None, mode="native"):
        if out_file is None: out_file = self._autoencoder_info_file
        expression = self.get_expression_script_for_plumed(mode=mode)
        with open(out_file, 'w') as f_out:
            f_out.write(expression)
        return

    def write_coefficients_of_connections_into_file(self, out_file=None):
        index_CV_layer = (len(self._node_num) - 1) / 2
        if out_file is None: out_file = self._autoencoder_info_file
        with open(out_file, 'w') as f_out:
            for item in range(index_CV_layer):
                f_out.write(str(list(self._connection_between_layers_coeffs[item])))
                f_out.write(',\n')
            for item in range(index_CV_layer):
                f_out.write(str(list(self._connection_with_bias_layers_coeffs[item])))
                f_out.write(',\n')
        return

    def check_PC_consistency(self, another_autoencoder, input_data = None, single_component_pair=None):
        from sklearn import linear_model
        assert (isinstance(another_autoencoder, autoencoder))
        if input_data is None:  input_data = self._data_set
        PCs_1 = self.get_PCs(input_data)
        PCs_2 = another_autoencoder.get_PCs(input_data)
        if not single_component_pair is None:  # in this case, we check consistency of single component of PCs
            PCs_1 = PCs_1[:, [single_component_pair[0]]]
            PCs_2 = PCs_2[:, [single_component_pair[1]]]
            # print PCs_1.shape, PCs_2.shape
        temp_regression = linear_model.LinearRegression().fit(PCs_1, PCs_2)
        predicted_PCs_2 = temp_regression.predict(PCs_1)
        r_value = temp_regression.score(PCs_1, PCs_2)
        return PCs_1, PCs_2, predicted_PCs_2, r_value

    @staticmethod
    def pairwise_PC_consistency_check(autoencoder_list, input_data=None, single_component_pair=None):
        result = [[item_1.check_PC_consistency(item_2, input_data=input_data, single_component_pair=single_component_pair)[3]
                  for item_1 in autoencoder_list] for item_2 in autoencoder_list]
        return np.array(result)

    def get_effective_numbers_of_occupied_bins_in_PC_space(self, input_data, range_of_PC_in_one_dim = [-1, 1],
                                                           num_of_bins=10, min_num_per_bin=2):
        PCs = self.get_PCs(input_data)
        dimensionality = len(PCs[0])
        range_of_PCs = [range_of_PC_in_one_dim for _ in range(dimensionality)]
        hist_matrix, edges = np.histogramdd(PCs, bins=num_of_bins * np.ones(dimensionality), range=range_of_PCs)
        return np.sum(hist_matrix >= min_num_per_bin), hist_matrix

    def cluster_configs_based_on_distances_in_PC_space(self, folder_for_pdb,
                                                num_clusters, output_folder, radius=0.02):
        """
        This function clusters configurations based on distance in PC space, and generates output pdb files
        containing configurations in each cluster which have distance smaller than 'radius' to the
        corresponding cluster center.
        Why don't I use click-and-save approach (as is done in plotting object in ANN_simulation.py)?
        Because 1. it is not convenient to click for higher-D space, 2. I am lazy to click even for 2D.
        :param temp_autoencoder: autoencoder used to get PCs
        :param folder_for_pdb: folder containing pdb files for input
        :param num_clusters: number of clusters (for K-means)
        :param radius: configs with distance less than 'radius' to the cluster center in PC space will be included in the output pdb
        :return: cluster_pdb_files, cluster_centers
        """
        if not os.path.exists(output_folder):
            subprocess.check_output(['mkdir', output_folder])

        _1 = coordinates_data_files_list([folder_for_pdb])
        _1 = _1.create_sub_coor_data_files_list_using_filter_conditional(lambda x: not 'aligned' in x)
        scaling_factor = CONFIG_49
        input_data = _1.get_coor_data(scaling_factor)
        input_data = Sutils.remove_translation(input_data)
        PCs = self.get_PCs(input_data)
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
        kmeans.fit(PCs)
        indices_list = np.array([np.where(kmeans.labels_ == ii)[0]
                                 for ii in range(kmeans.n_clusters)])
        out_pdb_list = []
        for index, item in enumerate(indices_list):
            # save configurations with distance less than 'radius' to corresponding cluster center
            item = list(filter(lambda x: np.linalg.norm(PCs[x] - kmeans.cluster_centers_[index]) < radius, item))
            if len(item) > 0:
                output_pdb_name = '%s/%04d_temp_frames_%s.pdb' % \
                                    (output_folder, index, str(list(kmeans.cluster_centers_[index])).replace(' ',''))
                out_pdb_list.append(output_pdb_name)
                _1.write_pdb_frames_into_file_with_list_of_coor_index(item, output_pdb_name, verbose=False)
                # assertion part
                molecule_type.generate_coordinates_from_pdb_files(path_for_pdb=output_pdb_name)
                temp_input_data = np.loadtxt(output_pdb_name.replace('.pdb', '_coordinates.txt')) / scaling_factor
                temp_input_data = Sutils.remove_translation(temp_input_data)
                PCs_of_points_selected = self.get_PCs(input_data=temp_input_data)
                assert_almost_equal(PCs_of_points_selected, PCs[item], decimal=4)
        return out_pdb_list, kmeans.cluster_centers_

    @abc.abstractmethod
    def get_PCs(self, input_data=None):
        """must be implemented by subclasses"""
        pass

    @abc.abstractmethod
    def train(self):
        """must be implemented by subclasses"""
        pass

    @abc.abstractmethod
    def get_output_data(self, input_data=None, num_of_PCs=None):
        """must be implemented by subclasses"""
        pass

    @abc.abstractmethod
    def get_mid_result(self, input_data=None):
        """must be implemented by subclasses"""
        pass

    def get_training_error(self, num_of_PCs=None):
        """
        :param num_of_PCs: this option only works for hierarchical case, indicate you would like to get error with
        a specific number of PCs (instead of all PCs)
        """
        input_data = np.array(self._data_set)
        actual_output_data = self.get_output_data(num_of_PCs)
        if hasattr(self, '_output_data_set') and not self._output_data_set is None:
            expected_output_data = self._output_data_set
        else:
            expected_output_data = input_data
        return np.linalg.norm(expected_output_data - actual_output_data) / sqrt(self._node_num[0] * len(input_data))

    def get_relative_error_for_each_point(self, input_data=None, output_data=None):
        if input_data is None: input_data = self._data_set
        if output_data is None:
            if self._output_data_set is None: output_data = self._data_set
            else: output_data = self._output_data_set
        temp_output = self.get_output_data(input_data)
        relative_err = np.linalg.norm(temp_output - output_data, axis=1) / np.linalg.norm(output_data, axis=1)
        assert (len(relative_err) == len(input_data)), (len(relative_err), len(input_data))
        return relative_err

    def get_fraction_of_variance_explained(self, num_of_PCs=None, hierarchical_FVE=False):
        """ here num_of_PCs is the same with that in get_training_error() """
        index_CV_layer = (len(self._node_num) - 1) / 2
        input_data = np.array(self._data_set)
        actual_output_data = self.get_output_data(num_of_PCs)
        if hasattr(self, '_output_data_set') and not self._output_data_set is None:
            expected_output_data = self._output_data_set
        else:
            expected_output_data = input_data

        var_of_output = sum(np.var(expected_output_data, axis=0))
        var_of_err = sum(np.var(actual_output_data - expected_output_data, axis=0))
        if self._hierarchical:
            num_PCs = self._node_num[index_CV_layer] / 2 if self._hidden_layers_type[index_CV_layer - 1] == CircularLayer \
                else self._node_num[index_CV_layer]
            length_for_hierarchical_component = expected_output_data.shape[1] / num_PCs
            hierarchical_actual_output_list = [actual_output_data[:,
                                    item * length_for_hierarchical_component:
                                    (item + 1) * length_for_hierarchical_component]
                                               for item in range(num_PCs)]
            expected_output_component = expected_output_data[:, -length_for_hierarchical_component:]
            assert (expected_output_component.shape == hierarchical_actual_output_list[0].shape)
            var_of_expected_output_component = sum(np.var(expected_output_component, axis=0))
            var_of_actual_output_component_list = [sum(np.var(item - expected_output_component, axis=0)) for item in hierarchical_actual_output_list]
            result = [1 - item / var_of_expected_output_component for item in var_of_actual_output_component_list]
            if not hierarchical_FVE:
                result = result[-1]   # it is more reasonable to return only last FVE (which includes information from all CV nodes)
        else:
            result = 1 - var_of_err / var_of_output
        return result

    def get_commands_for_further_biased_simulations(self, list_of_potential_center=None,
                                                    num_of_simulation_steps=None,
                                                    autoencoder_info_file=None,
                                                    force_constant_for_biased=None,
                                                    bias_method=CONFIG_65
                                                    ):
        """this function creates a list of commands for further biased simulations that should be done later,
        either in local machines or on the cluster
        """
        if num_of_simulation_steps is None:
            num_of_simulation_steps = CONFIG_8
        if autoencoder_info_file is None:
            autoencoder_info_file = self._autoencoder_info_file
        if bias_method == "US":
            PCs_of_network = self.get_PCs()
            if self._hidden_layers_type[1] == CircularLayer:
                assert (len(PCs_of_network[0]) == self._node_num[2] / 2)
            else:
                assert (len(PCs_of_network[0]) == self._node_num[2])
            if list_of_potential_center is None:
                list_of_potential_center = molecule_type.get_boundary_points(list_of_points=PCs_of_network)

            start_from_nearest_config = CONFIG_74
            if start_from_nearest_config:
                nearest_pdb_frame_index_list = []
                _1 = coordinates_data_files_list(['../target/%s' % CONFIG_30])
                _1 = _1.create_sub_coor_data_files_list_using_filter_conditional(lambda x: not 'aligned' in x)
                temp_input_data = _1.get_coor_data(scaling_factor=CONFIG_49)
                temp_input_data = Sutils.remove_translation(temp_input_data)
                temp_all_PCs = list(self.get_PCs(temp_input_data))
                assert len(temp_all_PCs) == np.sum(_1.get_list_of_line_num_of_coor_data_file())
                for item_2 in list_of_potential_center:
                    temp_distances = np.array([np.linalg.norm(item_3 - item_2) for item_3 in temp_all_PCs])
                    index_of_nearest_config = np.argmin(temp_distances)

                    nearest_pdb, nearest_frame_index = _1.get_pdb_name_and_corresponding_frame_index_with_global_coor_index(index_of_nearest_config)
                    nearest_pdb_frame_index_list.append([nearest_pdb, nearest_frame_index])
                    # assertion part
                    temp_input_data_2 = np.loadtxt(nearest_pdb.replace('.pdb', '_coordinates.txt')) / CONFIG_49
                    temp_input_data_2 = Sutils.remove_translation(temp_input_data_2)
                    temp_PC_2 = self.get_PCs(temp_input_data_2)[nearest_frame_index]
                    print temp_distances[index_of_nearest_config]
                    expected = temp_distances[index_of_nearest_config]
                    actual = np.linalg.norm(temp_PC_2 - item_2)
                    assert_almost_equal(expected, actual, decimal=3)

            if force_constant_for_biased is None:
                if isinstance(molecule_type, Trp_cage):
                    temp_state_coor_file = '../resources/1l2y_coordinates.txt'
                elif isinstance(molecule_type, Alanine_dipeptide):
                    temp_state_coor_file = '../resources/alanine_dipeptide_coordinates.txt'
                elif isinstance(molecule_type, Src_kinase) or isinstance(molecule_type, BetaHairpin):
                    temp_state_coor_file = None
                else:
                    raise Exception('molecule type error')

                if CONFIG_53 == "fixed":
                    force_constant_for_biased = [CONFIG_9 for _ in list_of_potential_center]
                elif CONFIG_53 == "flexible":
                    input_folded_state = np.loadtxt(temp_state_coor_file) / CONFIG_49
                    PC_folded_state = self.get_PCs(Sutils.remove_translation(input_folded_state))[0]
                    print("PC_folded_state = %s" % str(PC_folded_state))
                    force_constant_for_biased = [2 * CONFIG_54 / np.linalg.norm(np.array(item) - PC_folded_state) ** 2
                                                 for item in list_of_potential_center]
                elif CONFIG_53 == "truncated":
                    input_folded_state = np.loadtxt(temp_state_coor_file) / CONFIG_49
                    PC_folded_state = self.get_PCs(Sutils.remove_translation(input_folded_state))[0]
                    print("PC_folded_state = %s" % str(PC_folded_state))
                    force_constant_for_biased = [min(2 * CONFIG_54 / np.linalg.norm(np.array(item) - PC_folded_state) ** 2,
                                                     CONFIG_9) for item in list_of_potential_center]
                else:
                    raise Exception("error")

            todo_list_of_commands_for_simulations = []
            if CONFIG_48 == 'Cartesian' or CONFIG_48 == 'pairwise_distance':
                input_data_type = 1
            elif CONFIG_48 == 'cossin':
                input_data_type = 0
            else:
                raise Exception("error input data type")

            for index, potential_center in enumerate(list_of_potential_center):
                if isinstance(molecule_type, Alanine_dipeptide):
                    parameter_list = (str(CONFIG_16), str(num_of_simulation_steps), str(force_constant_for_biased[index]),
                                      '../target/Alanine_dipeptide/network_%d' % self._index,
                                      autoencoder_info_file,
                                      'pc_' + str(potential_center).replace(' ', '')[1:-1],
                                      input_data_type
                                      # need to remove white space, otherwise parsing error
                                      )
                    command = "python ../src/biased_simulation.py %s %s %s %s %s %s --data_type_in_input_layer %d" % parameter_list
                    if CONFIG_42:  # whether the force constant adjustable mode is enabled
                        command = command + ' --fc_adjustable --autoencoder_file %s --remove_previous ' % (
                            '../resources/Alanine_dipeptide/network_%d.pkl' % self._index)
                    if CONFIG_17[1] == CircularLayer:
                        command += ' --layer_types Tanh,Circular'
                else:
                    parameter_list = (
                            str(CONFIG_16), str(num_of_simulation_steps), str(force_constant_for_biased[index]),
                            '../target/placeholder_1/network_%d/' % self._index, autoencoder_info_file,
                            'pc_' + str(potential_center).replace(' ', '')[1:-1],
                            CONFIG_40, CONFIG_51, index % 2)
                    command = "python ../src/biased_simulation_general.py placeholder_2 %s %s %s %s %s %s %s %s --device %d" % parameter_list
                    if not input_data_type: command += ' --data_type_in_input_layer 0'
                    if CONFIG_72: command += ' --fast_equilibration 1'
                    if CONFIG_42:
                        command += ' --fc_adjustable --autoencoder_file %s --remove_previous' % (
                            '../resources/placeholder_1/network_%d.pkl' % self._index)
                    if start_from_nearest_config:
                        command += ' --starting_pdb_file %s --starting_frame %d ' % (nearest_pdb_frame_index_list[index][0],
                                                                                     nearest_pdb_frame_index_list[index][1])
                    if isinstance(molecule_type, Trp_cage): command = command.replace('placeholder_1', 'Trp_cage').replace('placeholder_2', 'Trp_cage')
                    elif isinstance(molecule_type, Src_kinase): command = command.replace('placeholder_1', 'Src_kinase').replace('placeholder_2', '2src')
                    elif isinstance(molecule_type, BetaHairpin): command = command.replace('placeholder_1', 'BetaHairpin').replace('placeholder_2', 'BetaHairpin')
                    else: raise Exception("molecule type not defined")

                todo_list_of_commands_for_simulations += [command]
        elif bias_method == "MTD":
            todo_list_of_commands_for_simulations = []
            self.write_expression_script_for_plumed()
            dimensionality = CONFIG_36
            pc_string = 'pc_' + ','.join(['0' for _ in range(dimensionality)])
            if isinstance(molecule_type, Alanine_dipeptide):
                for mtd_sim_index in range(5):
                    parameter_list = (str(CONFIG_16), str(num_of_simulation_steps), str(mtd_sim_index),
                                      '../target/Alanine_dipeptide/network_%d/' % self._index,
                                      self._autoencoder_info_file, pc_string)
                    command = "python ../src/biased_simulation.py %s %s %s %s %s %s --data_type_in_input_layer 1 --bias_method MTD" % parameter_list
                    todo_list_of_commands_for_simulations += [command]
            elif isinstance(molecule_type, Trp_cage):
                for mtd_sim_index in range(6):
                    parameter_list = (str(CONFIG_16), str(num_of_simulation_steps), str(mtd_sim_index),
                                      '../target/Trp_cage/network_%d/' % self._index, self._autoencoder_info_file,
                                      pc_string, CONFIG_40, CONFIG_51, mtd_sim_index % 2)
                    command = "python ../src/biased_simulation_general.py Trp_cage %s %s %s %s %s %s %s %s --data_type_in_input_layer 1 --bias_method MTD --device %d" % parameter_list
                    todo_list_of_commands_for_simulations += [command]
            else:
                raise Exception("molecule type not defined")
        else:
            raise Exception("bias method not found")

        return todo_list_of_commands_for_simulations

    def get_proper_potential_centers_for_WHAM(self, list_of_points, threshold_radius, min_num_of_neighbors):
        """
        This function selects some 'proper' potential centers within the domain from list_of_points, by "proper"
        we mean there are at least min_num_of_neighbors data points that are located within the radius of threshold_radius
        of the specific potential center.
        Typically list_of_points could be evenly distributed grid points in PC space
        """
        data_points = np.array(self.get_PCs())
        list_of_points = np.array(list_of_points)
        assert (data_points.shape[1] == list_of_points.shape[1])
        proper_potential_centers = []

        for item in list_of_points:
            neighbors_num = sum([np.dot(item - x, item - x) < threshold_radius * threshold_radius for x in data_points])

            if neighbors_num >= min_num_of_neighbors:
                proper_potential_centers += [item]

        return proper_potential_centers

    def get_proper_potential_centers_for_WHAM_2(self, total_number_of_potential_centers):
        data_points = np.array(self.get_PCs())
        kmeans = KMeans(init='k-means++', n_clusters=total_number_of_potential_centers, n_init=10)
        kmeans.fit(data_points)
        return kmeans.cluster_centers_

    def generate_mat_file_for_WHAM_reweighting(self,
                                               directory_containing_coor_files,
                                               mode="Bayes",  # mode = "standard" or "Bayes"
                                               folder_to_store_files='./standard_WHAM/', dimensionality=2,
                                               input_data_type='cossin',  # input_data_type could be 'cossin' or 'Cartesian'
                                               scaling_factor=CONFIG_49,  # only works for 'Cartesian'
                                               dihedral_angle_range=[1,2],  # only used for alanine dipeptide
                                               starting_index_of_last_few_frames=0,  # number of last few frames used in calculation, 0 means to use all frames
                                               ending_index_of_frames = 0,  # end index, for FES convergence check
                                               random_dataset = False,  # pick random dataset to estimate variance
                                               num_of_bins = 20
                                               ):
        """
        note: 
        dihedral_angle_range, starting_index_of_last_few_frames, ending_index_of_frames, random_dataset 
        may not work for Bayes mode
        num_of_bins only works for Bayes mode
        """
        if folder_to_store_files[-1] != '/':
            folder_to_store_files += '/'
        if not os.path.exists(folder_to_store_files):
            subprocess.check_output(['mkdir', folder_to_store_files])

        if mode == "Bayes":
            for item in ['bias', 'hist', 'traj', 'traj_proj']:
                directory = folder_to_store_files + item
                subprocess.check_output(['mkdir', '-p', directory])
                assert (os.path.exists(directory))
        else: pass

        temp_coor_file_obj = coordinates_data_files_list([directory_containing_coor_files])
        list_of_coor_data_files = temp_coor_file_obj.get_list_of_coor_data_files()
        force_constants = []
        harmonic_centers = []
        window_counts = []
        coords = []
        umbOP = []
        num_of_random_points_to_pick_in_each_file = None
        if random_dataset:
            temp_total_num_points = np.sum(temp_coor_file_obj.get_list_of_line_num_of_coor_data_file())
            temp_total_num_files = len(temp_coor_file_obj.get_list_of_line_num_of_coor_data_file())
            temp_rand_array = np.random.rand(temp_total_num_files)
            temp_rand_array *= (temp_total_num_points / np.sum(temp_rand_array))
            temp_rand_array = temp_rand_array.round()
            temp_rand_array[0] = temp_total_num_points - np.sum(temp_rand_array[1:])
            assert (temp_rand_array.sum() == temp_total_num_points)
            num_of_random_points_to_pick_in_each_file = temp_rand_array.astype(int)
        for temp_index, item in enumerate(list_of_coor_data_files):
            # print('processing %s' %item)
            temp_force_constant = float(item.split('output_fc_')[1].split('_pc_')[0])
            force_constants += [[temp_force_constant] * dimensionality  ]
            temp_harmonic_center_string = item.split('_pc_[')[1].split(']')[0]
            harmonic_centers += [[float(item_1) for item_1 in temp_harmonic_center_string.split(',')]]
            if input_data_type == 'cossin':
                temp_coor = self.get_PCs(molecule_type.get_many_cossin_from_coordinates_in_list_of_files([item]))
            elif input_data_type == 'Cartesian':
                temp_coor = self.get_PCs(Sutils.remove_translation(np.loadtxt(item) / scaling_factor))
            else:
                raise Exception('error input_data_type')

            if random_dataset:
                # data_index_list = random.sample(range(temp_coor.shape[0]), int(0.5 * temp_coor.shape[0]))  # nonrepeated
                # bootstrap for error estimation
                data_index_list = [random.choice(range(temp_coor.shape[0]))
                                   for _ in range(num_of_random_points_to_pick_in_each_file[temp_index])]
                # print "random data_index_list"
            else:
                data_index_list = np.arange(temp_coor.shape[0])
                data_index_list = data_index_list[starting_index_of_last_few_frames:]
                if ending_index_of_frames != 0: data_index_list = data_index_list[:ending_index_of_frames]

            temp_coor = temp_coor[data_index_list]
            assert len(temp_coor) == len(data_index_list)
            temp_window_count = temp_coor.shape[0]
            window_counts += [float(temp_window_count)]   # there exists problems if using int

            coords += list(temp_coor)
            if isinstance(molecule_type, Alanine_dipeptide):
                temp_angles = np.array(molecule_type.get_many_dihedrals_from_coordinates_in_file([item]))[data_index_list]
                temp_umbOP = [[a[temp_dihedral_index] for temp_dihedral_index in dihedral_angle_range] for a in temp_angles]
                assert (temp_window_count == len(temp_umbOP)), (temp_window_count, len(temp_umbOP))
                assert (len(dihedral_angle_range) == len(temp_umbOP[0]))
                umbOP += temp_umbOP
            elif isinstance(molecule_type, Trp_cage):
                temp_corresponding_pdb_list = coordinates_data_files_list([item]).get_list_of_corresponding_pdb_files()
                temp_CA_RMSD = np.array(Trp_cage.metric_RMSD_of_atoms(temp_corresponding_pdb_list))
                temp_helix_RMSD = np.array(Trp_cage.metric_RMSD_of_atoms(temp_corresponding_pdb_list,
                                                                atom_selection_statement='resid 2:8 and name CA'))
                umbOP += list(zip(temp_CA_RMSD[data_index_list], temp_helix_RMSD[data_index_list]))

        if mode == "standard":
            max_of_coor = map(lambda x: round(x, 1) + 0.1, map(max, list(zip(*coords))))
            min_of_coor = map(lambda x: round(x, 1) - 0.1, map(min, list(zip(*coords))))
            interval = 0.1

            window_counts = np.array(window_counts)
            sciio.savemat(folder_to_store_files + 'WHAM_nD__preprocessor.mat', {'window_counts': window_counts,
                                                                                'force_constants': force_constants,
                                                                                'harmonic_centers': harmonic_centers,
                                                                                'coords': coords, 'dim': dimensionality,
                                                                                'temperature': 300.0,
                                                                                'periodicity': [[0.0] * dimensionality],
                                                                                'dF_tol': 0.001,
                                                                                'min_gap_max_ORIG': [
                                                                                    [min_of_coor[item_2], interval,
                                                                                     max_of_coor[item_2]] for item_2 in range(dimensionality)]
                                                                                })
            sciio.savemat(folder_to_store_files + 'umbrella_OP.mat',
                          {'umbOP': umbOP
                           })

        elif mode == "Bayes":
            # write info into files
            # 1st: bias potential info
            with open(folder_to_store_files + 'bias/harmonic_biases.txt', 'w') as f_out:
                for item in range(len(force_constants)):
                    f_out.write('%d\t' % (item + 1))
                    for write_item in harmonic_centers[item]:
                        f_out.write('%f\t' % write_item)
                    for write_item in force_constants[item]:
                        f_out.write('%f\t' % write_item)
                    f_out.write("\n")

            # 2nd: trajectory, and projection trajectory in phi-psi space (for reweighting), and histogram
            epsilon = 1e-5
            coords = np.array(coords)
            binEdges_list = []
            with open(folder_to_store_files + 'hist/hist_binEdges.txt', 'w') as f_out:
                for item_100 in range(dimensionality):
                    binEdges = np.linspace(np.min(coords[:, item_100]) - epsilon,
                                           np.max(coords[:, item_100]) + epsilon, num_of_bins + 1)
                    binEdges_list.append(binEdges.tolist())
                    for item in binEdges:
                        f_out.write('%f\t' % item)
                    f_out.write('\n')

            num_of_bins_proj = 40
            umbOP = np.array(umbOP)
            with open(folder_to_store_files + 'hist/hist_binEdges_proj.txt', 'w') as f_out:
                for item_100 in range(len(umbOP[0])):
                    binEdges_proj = np.linspace(np.min(umbOP[:, item_100]) - epsilon,
                                                np.max(umbOP[:, item_100]) + epsilon, num_of_bins_proj + 1)
                    for item in binEdges_proj:
                        f_out.write('%f\t' % item)
                    f_out.write('\n')

            end_index = 0
            for item, count in enumerate(window_counts):
                start_index = int(end_index)
                end_index = int(start_index + count)
                with open(folder_to_store_files + 'traj/traj_%d.txt' % (item + 1), 'w') as f_out_1, \
                        open(folder_to_store_files + 'traj_proj/traj_%d.txt' % (item + 1), 'w') as f_out_2, \
                        open(folder_to_store_files + 'hist/hist_%d.txt' % (item + 1), 'w') as f_out_3:
                    for line in coords[start_index:end_index]:
                        for item_1 in line:
                            f_out_1.write('%f\t' % item_1)

                        f_out_1.write("\n")

                    for line in umbOP[start_index:end_index]:
                        for item_1 in line:
                            f_out_2.write('%f\t' % item_1)

                        f_out_2.write("\n")

                    temp_hist, _ = np.histogramdd(np.array(coords[start_index:end_index]),
                                                     bins=binEdges_list)
                    for _1 in temp_hist.flatten():
                        f_out_3.write('%d\t' % _1)
        else:
            raise Exception("error mode!")

        return


class autoencoder_Keras(autoencoder):
    def _init_extra(self,
                    network_parameters = CONFIG_4,
                    batch_size = 100,
                    enable_early_stopping=True
                    ):
        self._network_parameters = network_parameters
        self._batch_size = batch_size
        self._enable_early_stopping = enable_early_stopping
        self._molecule_net_layers = None              # why don't I save molecule_net (Keras model) instead? since it it not picklable:
                                                      # https://github.com/luispedro/jug/issues/30
                                                      # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        return

    def get_output_data(self, input_data=None, num_of_PCs = None):
        if input_data is None: input_data = self._data_set
        return self._molecule_net.predict(input_data)

    def get_PCs(self, input_data=None):
        index_CV_layer = (len(self._node_num) - 1) / 2
        if input_data is None: input_data = self._data_set
        if hasattr(self, '_output_as_circular') and self._output_as_circular:  # use hasattr for backward compatibility
            raise Exception('no longer supported')

        if self._hidden_layers_type[index_CV_layer - 1] == CircularLayer:
            PCs = np.array([[acos(item[2 * _1]) * np.sign(item[2 * _1 + 1]) for _1 in range(len(item) / 2)]
                   for item in self._encoder_net.predict(input_data)])
            assert (len(PCs[0]) == self._node_num[index_CV_layer] / 2), (len(PCs[0]), self._node_num[index_CV_layer] / 2)
        elif self._hidden_layers_type[index_CV_layer - 1] == TanhLayer:
            PCs = self._encoder_net.predict(input_data)
            assert (len(PCs[0]) == self._node_num[index_CV_layer])
        else:
            raise Exception("PC layer type error")
        return PCs

    def get_outputs_from_PC(self, input_PC):
        index_CV_layer = (len(self._node_num) - 1) / 2
        if self._hidden_layers_type[index_CV_layer - 1] == CircularLayer: raise Exception('not implemented')
        inputs = Input(shape=(self._node_num[index_CV_layer],))
        x = inputs
        for item in self._molecule_net_layers[-index_CV_layer:]:
            x = item(x)     # using functional API
        model = Model(input=inputs, output=x)
        return model.predict(input_PC)

    def train(self, hierarchical=None, hierarchical_variant = CONFIG_77):
        if hierarchical is None: hierarchical = self._hierarchical
        output_layer_activation = layer_type_to_name_mapping[self._out_layer_type].lower()
        node_num = self._node_num
        data = self._data_set
        if hasattr(self, '_output_data_set') and not self._output_data_set is None:
            print ("outputs different from inputs")
            output_data_set = self._output_data_set
        else:
            output_data_set = data

        num_of_hidden_layers = len(self._hidden_layers_type)
        index_CV_layer = (len(node_num) - 1) / 2
        num_CVs = node_num[index_CV_layer] / 2 if self._hidden_layers_type[index_CV_layer - 1] == CircularLayer else \
            node_num[index_CV_layer]
        if hierarchical:
            # functional API: https://keras.io/getting-started/functional-api-guide
            temp_output_shape = output_data_set.shape
            output_data_set = np.repeat(output_data_set, num_CVs, axis=0).reshape(temp_output_shape[0],
                                        temp_output_shape[1] * num_CVs)   # repeat output for hierarchical case
            # check if the output data are correct
            temp_data_for_checking = output_data_set[0]
            for item in range(num_CVs):
                assert_almost_equal (
                    temp_data_for_checking[item * temp_output_shape[1]: (item + 1) * temp_output_shape[1]],
                    temp_data_for_checking[:temp_output_shape[1]])
            self._output_data_set = output_data_set
            inputs_net = Input(shape=(node_num[0],))
            x = Dense(node_num[1], activation='tanh',
                      kernel_regularizer=l2(self._network_parameters[4][0]))(inputs_net)
            for item in range(2, index_CV_layer):
                x = Dense(node_num[item], activation='tanh', kernel_regularizer=l2(self._network_parameters[4][item - 1]))(x)
            if self._hidden_layers_type[index_CV_layer - 1] == CircularLayer:
                x = Dense(node_num[index_CV_layer], activation='linear',
                            kernel_regularizer=l2(self._network_parameters[4][index_CV_layer - 1]))(x)
                x = Reshape((num_CVs, 2), input_shape=(node_num[index_CV_layer],))(x)
                x = Lambda(temp_lambda_func_for_circular_for_Keras)(x)
                encoded = Reshape((node_num[index_CV_layer],))(x)
                encoded_split = [temp_lambda_slice_layers_circular[item](encoded) for item in range(num_CVs)]
            elif self._hidden_layers_type[index_CV_layer - 1] == TanhLayer:
                encoded = Dense(node_num[index_CV_layer], activation='tanh',
                                kernel_regularizer=l2(self._network_parameters[4][index_CV_layer - 1]))(x)
                encoded_split = [temp_lambda_slice_layers[item](encoded) for item in range(num_CVs)]
            else: raise Exception('layer error')

            if hierarchical_variant == 0:  # this is logically equivalent to original version by Scholz
                x_next = [Dense(node_num[index_CV_layer + 1], activation='linear',
                                kernel_regularizer=l2(self._network_parameters[4][index_CV_layer]))(item) for item in encoded_split]
                x_next_1 = [x_next[0]]
                for item in range(2, len(x_next) + 1):
                    x_next_1.append(layers.Add()(x_next[:item]))
                x_next_1 = [temp_lambda_tanh_layer(item) for item in x_next_1]
                assert (len(x_next) == len(x_next_1))
                for item_index in range(index_CV_layer + 2, len(node_num) - 1):
                    x_next_1 = [Dense(node_num[item_index], activation='tanh', kernel_regularizer=l2(self._network_parameters[4][item_index - 1]))(item_2)
                                for item_2 in x_next_1]
                shared_final_layer = Dense(node_num[-1], activation=output_layer_activation,
                                           kernel_regularizer=l2(self._network_parameters[4][-1]))
                outputs_net = layers.Concatenate()([shared_final_layer(item) for item in x_next_1])
                encoder_net = Model(inputs=inputs_net, outputs=encoded)
                molecule_net = Model(inputs=inputs_net, outputs=outputs_net)
            elif hierarchical_variant == 1:   # simplified version, no shared layer after CV (encoded) layer
                concat_layers = [encoded_split[0]]
                concat_layers += [layers.Concatenate()(encoded_split[:item]) for item in range(2, num_CVs + 1)]
                x = [Dense(node_num[index_CV_layer + 1], activation='tanh',
                                kernel_regularizer=l2(self._network_parameters[4][index_CV_layer]))(item) for item in concat_layers]
                for item_index in range(index_CV_layer + 2, len(node_num) - 1):
                    x = [Dense(node_num[item_index], activation='tanh',
                               kernel_regularizer=l2(self._network_parameters[4][item_index - 1]))(item) for item in x]
                x = [Dense(node_num[-1], activation=output_layer_activation,
                                kernel_regularizer=l2(self._network_parameters[4][-1]))(item) for item in x]
                outputs_net = layers.Concatenate()(x)
                encoder_net = Model(inputs=inputs_net, outputs=encoded)
                molecule_net = Model(inputs=inputs_net, outputs=outputs_net)
            elif hierarchical_variant == 2:
                # boosted hierarchical autoencoders, CV i in encoded layer learns remaining error that has
                # not been learned by previous CVs
                x = [Dense(node_num[index_CV_layer + 1], activation='tanh',
                           kernel_regularizer=l2(self._network_parameters[4][index_CV_layer]))(item) for item in encoded_split]
                for item_index in range(index_CV_layer + 2, len(node_num) - 1):
                    x = [Dense(node_num[item_index], activation='tanh',
                               kernel_regularizer=l2(self._network_parameters[4][item_index - 1]))(item) for item in x]
                x = [Dense(node_num[-1], activation=output_layer_activation,
                           kernel_regularizer=l2(self._network_parameters[4][-1]))(item) for item in x]
                x_out = [x[0]]
                for item in range(2, len(x) + 1):
                    x_out.append(layers.Add()(x[:item]))
                assert (len(x_out) == len(x))
                outputs_net = layers.Concatenate()(x_out)
                encoder_net = Model(inputs=inputs_net, outputs=encoded)
                molecule_net = Model(inputs=inputs_net, outputs=outputs_net)
            else: raise Exception('error variant')
            # print molecule_net.summary()
            loss_function = mse_weighted
        # elif num_of_hidden_layers != 3:
        #     raise Exception('not implemented for this case')
        else:
            inputs_net = Input(shape=(node_num[0],))
            x = Dense(node_num[1], activation='tanh',
                      kernel_regularizer=l2(self._network_parameters[4][0]))(inputs_net)
            for item in range(2, index_CV_layer):
                x = Dense(node_num[item], activation='tanh', kernel_regularizer=l2(self._network_parameters[4][item - 1]))(x)
            if self._hidden_layers_type[index_CV_layer - 1] == CircularLayer:
                x = Dense(node_num[index_CV_layer], activation='linear',
                            kernel_regularizer=l2(self._network_parameters[4][index_CV_layer - 1]))(x)
                x = Reshape((node_num[index_CV_layer] / 2, 2), input_shape=(node_num[index_CV_layer],))(x)
                x = Lambda(temp_lambda_func_for_circular_for_Keras)(x)
                encoded = Reshape((node_num[index_CV_layer],))(x)
            elif self._hidden_layers_type[index_CV_layer - 1] == TanhLayer:
                encoded = Dense(node_num[index_CV_layer], activation='tanh',
                            kernel_regularizer=l2(self._network_parameters[4][index_CV_layer - 1]))(x)
            else:
                raise Exception('CV layer type error')
            x = Dense(node_num[index_CV_layer + 1], activation='tanh',
                      kernel_regularizer=l2(self._network_parameters[4][index_CV_layer]))(encoded)
            for item_index in range(index_CV_layer + 2, len(node_num)):
                x = Dense(node_num[item_index], activation=output_layer_activation,
                          kernel_regularizer=l2(self._network_parameters[4][item_index - 1]))(x)
            molecule_net = Model(inputs=inputs_net, outputs=x)
            encoder_net = Model(inputs=inputs_net, outputs=encoded)
            loss_function = 'mean_squared_error'

        try:
            from keras.utils import plot_model
            Helper_func.backup_rename_file_if_exists('model.png')
            plot_model(molecule_net, show_shapes=True, to_file='model.png')
        except: pass

        molecule_net.compile(loss=loss_function, metrics=[loss_function],
                             optimizer=SGD(lr=self._network_parameters[0],
                                           momentum=self._network_parameters[1],
                                           decay=self._network_parameters[2],
                                           nesterov=self._network_parameters[3])
                             )
        encoder_net.compile(loss=loss_function, metrics=[loss_function],
                             optimizer=SGD(lr=self._network_parameters[0],
                                           momentum=self._network_parameters[1],
                                           decay=self._network_parameters[2],
                                           nesterov=self._network_parameters[3])
                             )  # not needed, but do not want to see endless warning...

        training_print_info = '''training network with index = %d, training maxEpochs = %d, structure = %s, layers = %s, num of data = %d,
parameter = [learning rate: %f, momentum: %f, lrdecay: %f, regularization coeff: %s], output as circular = %s\n''' % \
                              (self._index, self._max_num_of_training, str(self._node_num),
                               str(self._hidden_layers_type).replace("class 'pybrain.structure.modules.", ''),
                               len(data),
                               self._network_parameters[0], self._network_parameters[1],
                               self._network_parameters[2], str(self._network_parameters[4]),
                               str(self._output_as_circular))

        print("Start " + training_print_info + str(datetime.datetime.now()))
        call_back_list = []
        earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
        if self._enable_early_stopping:
            call_back_list += [earlyStopping]

        train_history = molecule_net.fit(data, output_data_set, epochs=self._max_num_of_training, batch_size=self._batch_size,
                         verbose=int(self._network_verbose), validation_split=0.2, callbacks=call_back_list)

        dense_layers = [item for item in molecule_net.layers if isinstance(item, Dense)]
        for _1 in range(2):  # check first two layers only
            assert (dense_layers[_1].get_weights()[0].shape[0] == node_num[_1]), (
            dense_layers[_1].get_weights()[0].shape[1], node_num[_1])  # check shapes of weights

        self._connection_between_layers_coeffs = [item.get_weights()[0].T.flatten() for item in
                                                  molecule_net.layers if isinstance(item,
                                                                                    Dense)]  # transpose the weights for consistency
        self._connection_with_bias_layers_coeffs = [item.get_weights()[1] for item in molecule_net.layers if
                                                    isinstance(item, Dense)]

        print('Done ' + training_print_info + str(datetime.datetime.now()))
        self._molecule_net = molecule_net
        self._molecule_net_layers = molecule_net.layers
        self._encoder_net = encoder_net
        try:
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(train_history.history['loss'])
            axes[1].plot(train_history.history['val_loss'])
            png_file = self._filename_to_save_network.replace('.pkl', '.png')
            Helper_func.backup_rename_file_if_exists(png_file)
            fig.savefig(png_file)
        except: print "training history not plotted!"; pass
        return self, train_history

    def train_bak(self):
        """this is kept for old version"""
        node_num = self._node_num
        data = self._data_set
        if hasattr(self, '_output_data_set') and not self._output_data_set is None:
            print ("outputs different from inputs")
            output_data_set = self._output_data_set
        else:
            output_data_set = data

        num_of_hidden_layers = len(self._hidden_layers_type)
        if self._hierarchical:
            raise Exception('hierarchical version not implemented')
        elif num_of_hidden_layers != 3:
            raise Exception('not implemented for this case')
        else:
            molecule_net = Sequential()
            molecule_net.add(Dense(input_dim=node_num[0], output_dim=node_num[1], activation='tanh',W_regularizer=l2(self._network_parameters[4][0])))   # input layer
            if self._hidden_layers_type[1] == CircularLayer:
                molecule_net.add(Dense(input_dim=node_num[1], output_dim=node_num[2], activation='linear',W_regularizer=l2(self._network_parameters[4][1])))
                molecule_net.add(Reshape((node_num[2] / 2, 2), input_shape=(node_num[2],)))
                molecule_net.add(Lambda(temp_lambda_func_for_circular_for_Keras))  # circular layer
                molecule_net.add(Reshape((node_num[2],)))
                molecule_net.add(Dense(input_dim=node_num[2], output_dim=node_num[3], activation='tanh',W_regularizer=l2(self._network_parameters[4][2])))
                molecule_net.add(Dense(input_dim=node_num[3], output_dim=node_num[4], activation='linear',W_regularizer=l2(self._network_parameters[4][3])))

            elif self._hidden_layers_type[1] == TanhLayer:
                molecule_net.add(Dense(input_dim=node_num[1], output_dim=node_num[2], activation='tanh',W_regularizer=l2(self._network_parameters[4][1])))
                molecule_net.add(Dense(input_dim=node_num[2], output_dim=node_num[3], activation='tanh',W_regularizer=l2(self._network_parameters[4][2])))
                molecule_net.add(Dense(input_dim=node_num[3], output_dim=node_num[4], activation='linear',W_regularizer=l2(self._network_parameters[4][3])))
            else:
                raise Exception ('this type of hidden layer not implemented')

            if hasattr(self, '_output_as_circular') and self._output_as_circular:
                molecule_net.add(Reshape((node_num[4] / 2, 2), input_shape=(node_num[4],)))
                molecule_net.add(Lambda(temp_lambda_func_for_circular_for_Keras))  # circular layer
                molecule_net.add(Reshape((node_num[4],)))

            molecule_net.compile(loss='mean_squared_error', metrics=['accuracy'],
                                 optimizer=SGD(lr=self._network_parameters[0],
                                               momentum=self._network_parameters[1],
                                               decay= self._network_parameters[2],
                                               nesterov=self._network_parameters[3])
                                 )

            training_print_info = '''training network with index = %d, training maxEpochs = %d, structure = %s, layers = %s, num of data = %d,
parameter = [learning rate: %f, momentum: %f, lrdecay: %f, regularization coeff: %s], output as circular = %s\n''' % \
                                  (self._index, self._max_num_of_training, str(self._node_num),
                                   str(self._hidden_layers_type).replace("class 'pybrain.structure.modules.", ''),
                                   len(data),
                                   self._network_parameters[0], self._network_parameters[1],
                                   self._network_parameters[2], str(self._network_parameters[4]), str(self._output_as_circular))

            print("Start " + training_print_info + str(datetime.datetime.now()))
            call_back_list = []
            earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
            if self._enable_early_stopping:
                call_back_list += [earlyStopping]


            molecule_net.fit(data, output_data_set, nb_epoch=self._max_num_of_training, batch_size=self._batch_size,
                             verbose=int(self._network_verbose), validation_split=0.2, callbacks=call_back_list)

            dense_layers = [item for item in molecule_net.layers if isinstance(item, Dense)]
            for _1 in range(len(dense_layers)):
                assert(dense_layers[_1].get_weights()[0].shape[0] == node_num[_1]), (dense_layers[_1].get_weights()[0].shape[1], node_num[_1])   # check shapes of weights

            self._connection_between_layers_coeffs = [item.get_weights()[0].T.flatten() for item in molecule_net.layers if isinstance(item, Dense)]  # transpose the weights for consistency
            self._connection_with_bias_layers_coeffs = [item.get_weights()[1] for item in molecule_net.layers if isinstance(item, Dense)]

            print('Done ' + training_print_info + str(datetime.datetime.now()))
            self._molecule_net = molecule_net
            self._molecule_net_layers = molecule_net.layers

        return self

def temp_lambda_func_for_circular_for_Keras(x):
    """This has to be defined at the module level here, otherwise the pickle will not work
    """
    return x / ((x ** 2).sum(axis=2, keepdims=True).sqrt())

temp_lambda_tanh_layer = Lambda(lambda x: K.tanh(x))
# not sure if there are better ways to do this, since Lambda layer has to be defined at top level of the file,
# following line does not work
# temp_lambda_slice_layers = [Lambda(lambda x: x[:, [index]], output_shape=(1,)) for index in range(20)]
temp_lambda_slice_layers = [
    Lambda(lambda x: x[:, [0]], output_shape=(1,)), Lambda(lambda x: x[:, [1]], output_shape=(1,)),
    Lambda(lambda x: x[:, [2]], output_shape=(1,)), Lambda(lambda x: x[:, [3]], output_shape=(1,)),
    Lambda(lambda x: x[:, [4]], output_shape=(1,)), Lambda(lambda x: x[:, [5]], output_shape=(1,)),
    Lambda(lambda x: x[:, [6]], output_shape=(1,)), Lambda(lambda x: x[:, [7]], output_shape=(1,)),
    Lambda(lambda x: x[:, [8]], output_shape=(1,)), Lambda(lambda x: x[:, [9]], output_shape=(1,))]
temp_lambda_slice_layers_circular = [
    Lambda(lambda x: x[:, [0,1]], output_shape=(2,)),   Lambda(lambda x: x[:, [2,3]], output_shape=(2,)),
    Lambda(lambda x: x[:, [4,5]], output_shape=(2,)),   Lambda(lambda x: x[:, [6,7]], output_shape=(2,)),
    Lambda(lambda x: x[:, [8,9]], output_shape=(2,)),   Lambda(lambda x: x[:, [10,11]], output_shape=(2,)),
    Lambda(lambda x: x[:, [12,13]], output_shape=(2,)), Lambda(lambda x: x[:, [14,15]], output_shape=(2,)),
    Lambda(lambda x: x[:, [16,17]], output_shape=(2,)), Lambda(lambda x: x[:, [18,19]], output_shape=(2,))
]

def get_hierarchical_weights(weight_factor_for_hierarchical_err = 1):
    # following is custom loss function for hierarchical error of hierarchical autoencoder
    # it may be useful to assign different weights for hierarchical error,
    # instead of having E = E_1 + E_{1,2} + E_{1,2,3}
    # we have E = a^2 E_1 + a E_{1,2} + E_{1,2,3}, a < 1
    # to avoid too large bias towards reconstruction error using first few components
    # see progress report 20171101
    weight_for_hierarchical_error = np.ones(CONFIG_3[-1] * CONFIG_36)
    for item in range(CONFIG_36):
        weight_for_hierarchical_error[: item * CONFIG_3[-1]] *= weight_factor_for_hierarchical_err
    return weight_for_hierarchical_error

# weighted MSE
weight_for_MSE = get_hierarchical_weights()
if CONFIG_44:    
    print "MSE is weighted by %s" % str(weight_for_MSE)

def mse_weighted(y_true, y_pred):
    # return K.mean(K.variable(weight_for_MSE) * K.square(y_pred - y_true), axis=-1)  # TODO: do this later
    return K.mean(K.square(y_pred - y_true), axis=-1)

