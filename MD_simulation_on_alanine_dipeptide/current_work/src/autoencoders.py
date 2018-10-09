from config import *
from molecule_spec_sutils import *  # import molecule specific unitity code
from coordinates_data_files_list import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.models import Sequential, Model, load_model
from keras.optimizers import *
from keras.layers import Dense, Activation, Lambda, Reshape, Input
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras import layers
from keras import backend as K
import random
from compatible import *

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
                 data_files = None,      # if None, store data in pkl, otherwise store data in separate npy files (use relative path), to save storage when we want to use the same data to train many different models
                 autoencoder_info_file=None,  # this might be expressions, or coefficients
                 hidden_layers_types=CONFIG_17,
                 out_layer_type=CONFIG_78,  # different layers
                 node_num=CONFIG_3,  # the structure of ANN
                 index_CV=None,   # index of CV layer
                 epochs=CONFIG_5,
                 batch_size=100,
                 filename_to_save_network=CONFIG_6,
                 hierarchical=CONFIG_44,
                 hi_variant=CONFIG_77,
                 *args, **kwargs  # for extra init functions for subclasses
                 ):
        self._batch_size = batch_size
        self._index = index
        self._data_set = data_set_for_training
        self._output_data_set = output_data_set
        if data_files is None:
            self._data_files = data_files
        else:
            self._data_files = [os.path.basename(item_f) for item_f in data_files]   # require that data are store in the same folder as pkl file
        if autoencoder_info_file is None:
            self._autoencoder_info_file = "../resources/%s/autoencoder_info_%d.txt" % (CONFIG_30, index)
        else:
            self._autoencoder_info_file = autoencoder_info_file
        self._hidden_layers_type = hidden_layers_types
        self._out_layer_type = out_layer_type
        self._node_num = node_num
        if index_CV is None:
            self._index_CV = (len(self._node_num) - 1) / 2
        else:
            self._index_CV = index_CV
        self._epochs = epochs
        if filename_to_save_network is None:
            self._filename_to_save_network = "../resources/%s/network_%s.pkl" % (
            CONFIG_30, str(self._index))  # by default naming with its index
        else:
            self._filename_to_save_network = filename_to_save_network

        self._hierarchical = hierarchical
        self._hi_variant = hi_variant
        self._connection_between_layers_coeffs = None
        self._connection_with_bias_layers_coeffs = None
        self._molecule_net_layers = self._molecule_net = self._encoder_net = self._decoder_net = None
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
            a._molecule_net = load_model(filename.replace('.pkl','.hdf5'),custom_objects={'mse_weighted': get_mse_weighted()})
        elif not hasattr(a, '_molecule_net') and hasattr(a, '_molecule_net_layers') and (not a._molecule_net_layers is None):  # for backward compatibility
            a._molecule_net = Sequential()
            for item in a._molecule_net_layers:
                a._molecule_net.add(item)
        else:
            raise Exception('cannot load attribute _molecule_net')
        if os.path.isfile(filename.replace('.pkl', '_encoder.hdf5')):
            a._encoder_net = load_model(filename.replace('.pkl', '_encoder.hdf5'),custom_objects={'mse_weighted': get_mse_weighted()})
        else:
            raise Exception('obsolete, not going to do this')
        if not hasattr(a, '_index_CV'):
            a._index_CV = (len(a._node_num) - 1) / 2
        a.helper_load_data(filename)
        return a

    @staticmethod
    def move_data_from_pkl_to_external_files(model_pkl, data_files):
        """this function is used for backward compatibility, move training data from pkl to external files to save storage"""
        ae = autoencoder.load_from_pkl_file(model_pkl)
        assert (isinstance(ae, autoencoder))
        ae._data_files = data_files
        print "data_files = %s" % str(ae._data_files)
        ae.save_into_file(model_pkl)
        return
    
    def remove_pybrain_dependency(self):    
        """previously pybrain layers are directly used in attributes of this object, should be replaced by string to remove dependency"""
        self._in_layer_type = None
        self._hidden_layers_type = [layer_type_to_name_mapping[item] for item in self._hidden_layers_type]
        self._out_layer_type = layer_type_to_name_mapping[self._out_layer_type]
        return

    @staticmethod
    def remove_pybrain_dependency_and_save_to_file(filename):
        ae = autoencoder.load_from_pkl_file(filename)
        ae.remove_pybrain_dependency()
        ae.save_into_file(filename)
        return

    def helper_save_data(self, filename):
        """helper function to save data externally if _data_files are defined"""
        if hasattr(self, '_data_files') and not self._data_files is None:
            folder_of_pkl = os.path.dirname(os.path.realpath(filename))
            data_file_paths = [os.path.join(folder_of_pkl, item_file) for item_file in self._data_files]    # use relative paths to store data files to avoid issue when loading model from a different directory (e.g. from Jupyter notebook)
            data_file_paths[0] = Helper_func.attempt_to_save_npy(data_file_paths[0], self._data_set)       # save to external files
            data_file_paths[1] = Helper_func.attempt_to_save_npy(data_file_paths[1], self._output_data_set)
            print data_file_paths
            self._data_files = [os.path.basename(item_1) for item_1 in data_file_paths]   # restore relative paths
            self._data_set = self._output_data_set = None
        else:
            data_file_paths = None
        return data_file_paths

    def helper_load_data(self, filename):
        if hasattr(self, '_data_files') and not self._data_files is None:
            data_file_paths = [os.path.join(os.path.dirname(os.path.realpath(filename)), item_file)
                              for item_file in self._data_files]
            self._data_set = np.load(data_file_paths[0])
            self._output_data_set = np.load(data_file_paths[1])
        return

    def save_into_file(self, filename=CONFIG_6, fraction_of_data_to_be_saved = 1.0):
        if filename is None:
            filename = self._filename_to_save_network

        if fraction_of_data_to_be_saved != 1.0:
            number_of_data_points_to_be_saved = int(self._data_set.shape[0] * fraction_of_data_to_be_saved)
            print(("Warning: only %f of data (%d out of %d) are saved into pkl file" % (fraction_of_data_to_be_saved,
                                                                                        number_of_data_points_to_be_saved,
                                                                                        self._data_set.shape[0])))
            self._data_set = self._data_set[:number_of_data_points_to_be_saved]
            if not (self._output_data_set is None or self._output_data_set == np.array(None)):        # for backward compatibility
                self._output_data_set = self._output_data_set[:number_of_data_points_to_be_saved]

        hdf5_file_name = filename.replace('.pkl', '.hdf5')
        hdf5_file_name_encoder = hdf5_file_name.replace('.hdf5', '_encoder.hdf5')
        hdf5_file_name_decoder = hdf5_file_name.replace('.hdf5', '_decoder.hdf5')
        for item_filename in [filename, hdf5_file_name, hdf5_file_name_encoder, hdf5_file_name_decoder]:
            Helper_func.backup_rename_file_if_exists(item_filename)
        folder_to_store_files = os.path.dirname(filename)
        if folder_to_store_files != '' and (not os.path.exists(folder_to_store_files)):
            os.makedirs(folder_to_store_files)
        self._molecule_net.save(hdf5_file_name)
        self._encoder_net.save(hdf5_file_name_encoder)
        if not self._decoder_net is None: self._decoder_net.save(hdf5_file_name_decoder)
        self._molecule_net = self._molecule_net_layers = self._encoder_net = self._decoder_net = None  # we save model in hdf5, not in pkl
        data_file_paths = self.helper_save_data(filename=filename)
        with open(filename, 'wb') as my_file:
            pickle.dump(self, my_file, pickle.HIGHEST_PROTOCOL)

        # restore
        self._molecule_net = load_model(hdf5_file_name, custom_objects={'mse_weighted': get_mse_weighted()})
        self._encoder_net = load_model(hdf5_file_name_encoder, custom_objects={'mse_weighted': get_mse_weighted()})
        self.helper_load_data(filename)
        # self._decoder_net = load_model(hdf5_file_name_decoder, custom_objects={'mse_weighted': mse_weighted})
        return

    def get_expression_script_for_plumed(self, mode="native", node_num=None, connection_between_layers_coeffs=None,
                                         connection_with_bias_layers_coeffs=None,
                                         activation_function_list=None):
        if node_num is None: node_num = self._node_num
        if connection_between_layers_coeffs is None: connection_between_layers_coeffs = self._connection_between_layers_coeffs
        if connection_with_bias_layers_coeffs is None: connection_with_bias_layers_coeffs = self._connection_with_bias_layers_coeffs
        plumed_script = ''
        if mode == "native":  # using native implementation by PLUMED (using COMBINE and MATHEVAL)
            plumed_script += "bias_const: CONSTANT VALUE=1.0\n"  # used for bias
            if activation_function_list is None: activation_function_list = ['tanh'] * self._index_CV
            for layer_index in range(1, self._index_CV + 1):
                for item in range(node_num[layer_index]):
                    plumed_script += "l_%d_in_%d: COMBINE PERIODIC=NO COEFFICIENTS=" % (layer_index, item)
                    plumed_script += "%s" % \
                                     str(connection_between_layers_coeffs[layer_index - 1][
                                         item * node_num[layer_index - 1]:(item + 1) * node_num[
                                             layer_index - 1]].tolist())[1:-1].replace(' ', '')
                    plumed_script += ',%f' % connection_with_bias_layers_coeffs[layer_index - 1][item]
                    plumed_script += " ARG="
                    for _1 in range(node_num[layer_index - 1]):
                        plumed_script += 'l_%d_out_%d,' % (layer_index - 1, _1)

                    plumed_script += 'bias_const\n'

                if activation_function_list[layer_index - 1] == 'tanh':
                    for item in range(node_num[layer_index]):
                        plumed_script += 'l_%d_out_%d: MATHEVAL ARG=l_%d_in_%d FUNC=tanh(x) PERIODIC=NO\n' % (
                            layer_index, item, layer_index,item)
                elif activation_function_list[layer_index - 1] == 'softmax':    # generalization for classifier
                    plumed_script += "sum_output_layer: MATHEVAL ARG="
                    for item in range(node_num[layer_index]): plumed_script += 'l_%d_in_%d,' % (layer_index, item)
                    plumed_script = plumed_script[:-1]  + ' VAR='                 # remove last ','
                    for item in range(node_num[layer_index]): plumed_script += 't_var_%d,' % item
                    plumed_script = plumed_script[:-1] + ' FUNC='
                    for item in range(node_num[layer_index]): plumed_script += 'exp(t_var_%d)+' % item
                    plumed_script = plumed_script[:-1] + ' PERIODIC=NO\n'
                    for item in range(node_num[layer_index]):
                        plumed_script += 'l_%d_out_%d: MATHEVAL ARG=l_%d_in_%d,sum_output_layer FUNC=exp(x)/y PERIODIC=NO\n' % (
                            layer_index, item, layer_index, item)
        elif mode == "ANN":  # using ANN class
            temp_num_of_layers_used = self._index_CV + 1
            temp_input_string = ','.join(['l_0_out_%d' % item for item in range(node_num[0])])
            temp_num_nodes_string = ','.join([str(item) for item in node_num[:temp_num_of_layers_used]])
            temp_layer_type_string = CONFIG_17[:2]
            temp_layer_type_string = ','.join(temp_layer_type_string)
            temp_coeff_string = ''
            temp_bias_string = ''
            for _1, item_coeff in enumerate(connection_between_layers_coeffs[:temp_num_of_layers_used - 1]):
                temp_coeff_string += ' COEFFICIENTS_OF_CONNECTIONS%d=%s' % \
                                     (_1, ','.join([str(item) for item in item_coeff]))
            for _1, item_bias in enumerate(connection_with_bias_layers_coeffs[:temp_num_of_layers_used - 1]):
                temp_bias_string += ' VALUES_OF_BIASED_NODES%d=%s' % \
                                     (_1, ','.join([str(item) for item in item_bias]))

            plumed_script += "ann_force: ANN ARG=%s NUM_OF_NODES=%s LAYER_TYPES=%s %s %s" % \
                (temp_input_string, temp_num_nodes_string, temp_layer_type_string,
                 temp_coeff_string, temp_bias_string)
        else:
            raise Exception("mode error")
        return plumed_script

    def get_plumed_script_for_biased_simulation_with_INDUS_cg_input_and_ANN(self,
            water_index_string, atom_indices, r_low, r_high, scaling_factor, sigma=0.1, cutoff=0.2,
            potential_center=None, force_constant=None, out_plumed_file=None):
        """ used to generate plumed script for biased simulation, with INDUS coarse grained water
        molecule numbers as input for ANN, and biasing force is applied on outputs of ANN
        :param water_index_string: example: '75-11421:3'
        :param atom_indices: example: range(1, 25)
        :param scaling_factor: scaling factor for input of ANN
        :param sigma, cutoff: these are parameters for Gaussian, in unit of A (by default in plumed it is nanometer)
        :param potential_center: if it is None, does not generate biasing part in script
        """
        return self.get_plumed_script_for_biased_simulation_with_solute_pairwise_dis_and_solvent_cg_input_and_ANN(
            [], 0, water_index_string=water_index_string, solute_atoms_cg=atom_indices,
            r_low=r_low, r_high=r_high, scaling_solvent=scaling_factor, sigma=sigma, cutoff=cutoff,
            potential_center=potential_center, force_constant=force_constant, out_plumed_file=out_plumed_file
        )

    def get_plumed_script_for_biased_simulation_with_solute_pairwise_dis_and_solvent_cg_input_and_ANN(
            self, solute_atom_indices, scaling_solute, water_index_string, solute_atoms_cg, r_low, r_high,
            scaling_solvent, sigma=0.1, cutoff=0.2, potential_center=None, force_constant=None, out_plumed_file=None
    ):
        """ used to generate plumed script for biased simulation, with 1. pairwise distances of solute atoms,
        2. INDUS coarse grained water molecule numbers as input for ANN, and biasing force is applied on outputs of ANN
        :param solute_atom_indices: solute atoms for computing pairwise distances
        :param scaling_solute: scaling factor for solute when computing pairwise distances
        :param water_index_string: example: '75-11421:3', used in plumed script
        :param solute_atoms_cg: solute atoms for computing cg water counts, example: range(1, 25)
        :param scaling_solvent: scaling factor for solvent cg counts
        :param sigma, cutoff: these are parameters for Gaussian, in unit of A (by default in plumed it is nanometer)
        :param potential_center: if it is None, does not generate biasing part in script
        """
        result = ''
        result += Sutils._get_plumed_script_with_pairwise_dis_as_input(solute_atom_indices, scaling_factor=scaling_solute)
        num_pairwise_dis = len(solute_atom_indices) * (len(solute_atom_indices) - 1) / 2
        for _1, item in enumerate(solute_atoms_cg):
            result += "sph_%d: SPHSHMOD ATOMS=%s ATOMREF=%d RLOW=%f RHIGH=%f SIGMA=%.4f CUTOFF=%.4f\n" % (
                _1, water_index_string, item, r_low / 10.0, r_high / 10.0,
                sigma / 10.0, cutoff / 10.0)  # factor of 10.0 is used to convert A to nm
            result += "l_0_out_%d: COMBINE PERIODIC=NO COEFFICIENTS=%f ARG=sph_%d.Ntw\n" % (
                _1 + num_pairwise_dis, 1.0 / scaling_solvent, _1)
        result += self.get_expression_script_for_plumed(mode='ANN')  # add string generated by ANN plumed plugin
        if not potential_center is None:
            arg_string = ','.join(['ann_force.%d' % _2 for _2 in range(len(potential_center))])
            pc_string = ','.join([str(_2) for _2 in potential_center])
            if out_plumed_file is None:
                out_plumed_file = "temp_plumed_out_%s.txt" % pc_string
            kappa_string = ','.join([str(force_constant) for _ in potential_center])
            arg_string_2 = ','.join(['l_0_out_%d' % _2 for _2 in range(len(solute_atoms_cg))])
            result += """\nmypotential: RESTRAINT ARG=%s AT=%s KAPPA=%s
ave: COMBINE PERIODIC=NO ARG=%s

PRINT STRIDE=50 ARG=%s,ave FILE=%s""" % (
                arg_string, pc_string, kappa_string, arg_string_2, arg_string, out_plumed_file
            )
        return result

    def write_expression_script_for_plumed(self, out_file=None, mode="native"):
        if out_file is None: out_file = self._autoencoder_info_file
        expression = self.get_expression_script_for_plumed(mode=mode)
        with open(out_file, 'w') as f_out:
            f_out.write(expression)
        return

    def write_coefficients_of_connections_into_file(self, out_file=None):
        if out_file is None: out_file = self._autoencoder_info_file
        Helper_func.backup_rename_file_if_exists(out_file)
        with open(out_file, 'w') as f_out:
            for item in range(self._index_CV):
                f_out.write(str(list(self._connection_between_layers_coeffs[item])))
                f_out.write(',\n')
            for item in range(self._index_CV):
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
            item = list([x for x in item if np.linalg.norm(PCs[x] - kmeans.cluster_centers_[index]) < radius])
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
    def get_output_data(self, input_data=None):
        """must be implemented by subclasses"""
        pass

    @abc.abstractmethod
    def get_mid_result(self, input_data=None):
        """must be implemented by subclasses"""
        pass

    def get_fraction_of_variance_explained(self, hierarchical_FVE=False,
                                           output_index_range=None, featurewise=False, lag_time=0):
        # TODO: lag_time is obsolete
        """ here num_of_PCs is the same with that in get_training_error() """
        input_data = np.array(self._data_set)
        actual_output_data = self.get_output_data()
        if hasattr(self, '_output_data_set') and not self._output_data_set is None:
            expected_output_data = self._output_data_set
        else:
            expected_output_data = input_data
        if lag_time != 0:
            expected_output_data = expected_output_data[lag_time:]
            actual_output_data = actual_output_data[:-lag_time]

        if self._hierarchical:
            num_PCs = self._node_num[self._index_CV] / 2 if self._hidden_layers_type[self._index_CV - 1] == "Circular" \
                else self._node_num[self._index_CV]
            length_for_hierarchical_component = actual_output_data.shape[1] / num_PCs
            actual_output_list = [actual_output_data[:,
                                    item * length_for_hierarchical_component:
                                    (item + 1) * length_for_hierarchical_component]
                                               for item in range(num_PCs)]
            expected_output_part = expected_output_data[:, -length_for_hierarchical_component:]
        else:
            actual_output_list = [actual_output_data]    # use list, consistent with hierarchical case
            expected_output_part = expected_output_data
        if not output_index_range is None:
            actual_output_list = [item[:, output_index_range] for item in actual_output_list]
            expected_output_part = expected_output_part[:, output_index_range]
        assert (expected_output_part.shape == actual_output_list[0].shape)
        var_of_expected_output_part = np.var(expected_output_part, axis=0)
        var_of_err_list = [np.var(item - expected_output_part, axis=0)
                                     for item in actual_output_list]
        if featurewise:
            result = [1 - item / var_of_expected_output_part for item in var_of_err_list]
        else:
            result = [1 - np.sum(item) / np.sum(var_of_expected_output_part) for item in var_of_err_list]
        if not hierarchical_FVE:
            result = result[-1]  # it is reasonable to return only last FVE (constructed from all CVs)
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
        PCs_of_network = self.get_PCs()
        if self._hidden_layers_type[1] == "Circular":
            assert (len(PCs_of_network[0]) == self._node_num[2] / 2)
        else:
            assert (len(PCs_of_network[0]) == self._node_num[2])
        if list_of_potential_center is None:
            list_of_potential_center = molecule_type.get_boundary_points(list_of_points=PCs_of_network)
        if bias_method == "US":
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
                    print(temp_distances[index_of_nearest_config])
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
            if CONFIG_48 == 'Cartesian':
                input_data_type = 1
            elif CONFIG_48 == 'cossin':
                input_data_type = 0
            elif CONFIG_48 == 'pairwise_distance':
                input_data_type = 2
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
                    if CONFIG_17[1] == "Circular":
                        command += ' --layer_types Tanh,Circular'
                else:
                    parameter_list = (
                            str(CONFIG_16), str(num_of_simulation_steps), str(force_constant_for_biased[index]),
                            '../target/placeholder_1/network_%d/' % self._index, autoencoder_info_file,
                            'pc_' + str(potential_center).replace(' ', '')[1:-1],
                            CONFIG_40, CONFIG_51, index % 2)
                    command = "python ../src/biased_simulation_general.py placeholder_2 %s %s %s %s %s %s %s %s --device %d" % parameter_list
                    command += ' --data_type_in_input_layer %d ' % input_data_type
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
        elif bias_method == "US on pairwise distances":
            todo_list_of_commands_for_simulations = []
            if isinstance(molecule_type, Trp_cage):
                dim_of_CVs = len(list_of_potential_center[0])
                pc_arg_string = ['ann_force.%d' % index_ann for index_ann in range(dim_of_CVs)]
                pc_arg_string = ','.join(pc_arg_string)
                plumed_string = Sutils._get_plumed_script_with_pairwise_dis_as_input(
                    get_index_list_with_selection_statement('../resources/1l2y.pdb', CONFIG_73), CONFIG_49)
                plumed_string += self.get_expression_script_for_plumed(mode='ANN')
                for item_index, item_pc in enumerate(list_of_potential_center):
                    pc_string = ','.join([str(_1) for _1 in item_pc])
                    kappa_string = ','.join([str(CONFIG_9) for _ in range(dim_of_CVs)])
                    temp_plumed_file = '../resources/Trp_cage/temp_plumed_%02d_%02d.txt' % (self._index, item_index)
                    with open(temp_plumed_file, 'w') as my_f:
                        my_f.write(
                            plumed_string + '\nRESTRAINT ARG=%s AT=%s KAPPA=%s LABEL=mypotential\n' % (
                                pc_arg_string, pc_string, kappa_string)
                        )
                    parameter_list = (
                        str(CONFIG_16), str(num_of_simulation_steps), str(CONFIG_9),
                        '../target/Trp_cage/network_%d/' % self._index, 'none',
                        'pc_' + str(item_pc).replace(' ', '')[1:-1],
                        CONFIG_40, CONFIG_51, temp_plumed_file, item_index % 2)
                    command = "python ../src/biased_simulation_general.py Trp_cage %s %s %s %s %s %s %s %s --bias_method plumed_other --plumed_file %s --device %d" % parameter_list
                    todo_list_of_commands_for_simulations += [command]

            else: raise Exception("molecule type not defined")
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
                data_index_list = [random.choice(list(range(temp_coor.shape[0])))
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
            max_of_coor = [round(x, 1) + 0.1 for x in list(map(max, list(zip(*coords))))]
            min_of_coor = [round(x, 1) - 0.1 for x in list(map(min, list(zip(*coords))))]
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

    @staticmethod
    def tune_hyperparams_using_Bayes_optimization(in_data, out_data, folder, lr_range, momentum_range,
                                                  lr_log_scale=True, train_num_per_iter=5,
                                                  total_iter_num=20,
                                                  num_training_per_param=3,
                                                  print_command_only=False   # only print commands, does no training, basically doing random search of parameters
                                                  ):
        """use Bayes optimization for tuning hyperparameters,
        see http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html#bayesian-optimization"""
        def next_parameter_by_ei(best_y, y_mean, y_std, x_choices, num_choices):
            expected_improvement = (y_mean + 1.0 * y_std) - best_y
            max_index = np.argsort(expected_improvement)[-num_choices:]
            return x_choices[max_index], expected_improvement[max_index]

        from sklearn.gaussian_process import GaussianProcessRegressor
        import glob
        for iter_index in range(total_iter_num):
            autoencoder_files = sorted(glob.glob('%s/*.pkl' % folder))
            if len(autoencoder_files) == 0:  # use random search as the start
                params = np.random.uniform(size=(train_num_per_iter, 2))
                params[:, 1] = params[:, 1] * (momentum_range[1] - momentum_range[0]) + momentum_range[0]
                if lr_log_scale:
                    params[:, 0] = np.exp(
                        params[:, 0] * (np.log(lr_range[1]) - np.log(lr_range[0])) + np.log(lr_range[0]))
                else:
                    params[:, 0] = params[:, 0] * (lr_range[1] - lr_range[0]) + lr_range[0]
                next_params = params[:]
            else:  # generate params based on Bayes optimization
                gp = GaussianProcessRegressor()
                X_train, y_train = [], []
                for item_AE_file in autoencoder_files:
                    temp_AE = autoencoder.load_from_pkl_file(item_AE_file)
                    assert (isinstance(temp_AE, autoencoder_Keras))
                    X_train.append(temp_AE._network_parameters[:2])
                    if not np.isnan(temp_AE.get_fraction_of_variance_explained()):
                        y_train.append(temp_AE.get_fraction_of_variance_explained())
                    else:
                        y_train.append(-1.0)  # TODO: is it good?
                X_train, y_train = np.array(X_train), np.array(y_train)
                print(np.concatenate([X_train,y_train.reshape(y_train.shape[0], 1)], axis=-1))
                current_best_y_train = np.max(y_train)
                gp.fit(X_train, y_train)
                params = np.random.uniform(size=(100, 2))
                params[:, 1] = params[:, 1] * (momentum_range[1] - momentum_range[0]) + momentum_range[0]
                if lr_log_scale:
                    params[:, 0] = np.exp(
                        params[:, 0] * (np.log(lr_range[1]) - np.log(lr_range[0])) + np.log(lr_range[0]))
                else:
                    params[:, 0] = params[:, 0] * (lr_range[1] - lr_range[0]) + lr_range[0]
                y_mean, y_std = gp.predict(params, return_std=True)
                next_params, next_ei = next_parameter_by_ei(current_best_y_train, y_mean, y_std, params, train_num_per_iter)
                print(next_params, next_ei)

            assert (len(next_params) == train_num_per_iter)
            command_list = []
            cuda_index = 0
            for item_param in next_params:
                for index in range(num_training_per_param):
                    command = "python train_network_and_save_for_iter.py 1447 --num_of_trainings 1 --lr_m %f,%f --output_file %s/temp_%02d_%s_%02d.pkl --in_data %s --out_data %s" % (
                        item_param[0], item_param[1], folder, iter_index,
                        str(item_param).strip().replace(' ','').replace('[','').replace(']',''), index, in_data, out_data
                    )
                    if temp_home_directory == "/home/kengyangyao":
                        command = "THEANO_FLAGS=device=cuda%d " % cuda_index + command
                        cuda_index = 1 - cuda_index      # use two GPUs
                    command_list.append(command)
            if not print_command_only:
                num_failed_jobs = Helper_func.run_multiple_jobs_on_local_machine(
                    command_list, num_of_jobs_in_parallel=2)
            else:
                for item_commad in command_list: print item_commad
                return
            print("num_failed_jobs = %d" % num_failed_jobs)
        return


class autoencoder_Keras(autoencoder):
    def _init_extra(self,
                    network_parameters = CONFIG_4,
                    enable_early_stopping=True,
                    mse_weights=None
                    ):
        self._network_parameters = network_parameters
        if not isinstance(self._network_parameters[4], list):
            self._network_parameters[4] = [self._network_parameters[4]] * (len(self._node_num) - 1)    # simplify regularization for deeper networks
        assert isinstance(self._network_parameters[4], list)
        self._enable_early_stopping = enable_early_stopping
        if not (mse_weights is None) and self._hierarchical and self._node_num[self._index_CV] > 1:
            self._mse_weights = np.array(mse_weights.tolist() * self._node_num[self._index_CV])
        else:
            self._mse_weights = mse_weights
        self._molecule_net_layers = None              # why don't I save molecule_net (Keras model) instead? since it it not picklable:
                                                      # https://github.com/luispedro/jug/issues/30
                                                      # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
                                                      # obsolete: should not save _molecule_net_layers in the future, kept for backward compatibility
        return

    def get_output_data(self, input_data=None):
        if input_data is None: input_data = self._data_set
        return self._molecule_net.predict(input_data)

    def get_PCs(self, input_data=None):
        if input_data is None: input_data = self._data_set
        if self._hidden_layers_type[self._index_CV - 1] == "Circular":
            PCs = np.array([[acos(item[2 * _1]) * np.sign(item[2 * _1 + 1]) for _1 in range(len(item) / 2)]
                   for item in self._encoder_net.predict(input_data)])
            assert (len(PCs[0]) == self._node_num[self._index_CV] / 2), (len(PCs[0]), self._node_num[self._index_CV] / 2)
        else:
            PCs = self._encoder_net.predict(input_data)
            assert (len(PCs[0]) == self._node_num[self._index_CV])
        return PCs

    @staticmethod
    def layerwise_pretrain(data, dim_in, dim_out):
        """ref: https://www.kaggle.com/baogorek/autoencoder-with-greedy-layer-wise-pretraining/notebook"""
        # TODO: 1. use better training parameters. 2. use consistant activation functions, 3. consider how to do this for hierarchical case
        # TODO: 4. make activation function consistent with neural network
        data_in = Input(shape=(dim_in,))
        encoded = Dense(dim_out, activation='tanh')(data_in)
        data_out = Dense(dim_in, activation='tanh')(encoded)
        temp_ae = Model(inputs=data_in, outputs=data_out)
        encoder = Model(inputs=data_in, outputs=encoded)
        sgd = SGD(lr=0.3, decay=0, momentum=0.9, nesterov=True)
        temp_ae.compile(loss='mean_squared_error', optimizer=sgd)
        temp_ae.fit(data, data, epochs=20, batch_size=50,
                    validation_split=0.20, shuffle=True, verbose=False)
        encoded_data = encoder.predict(data)
        reconstructed = temp_ae.predict(data)
        var_of_output = np.sum(np.var(data, axis=0))
        var_of_err = np.sum(np.var(reconstructed - data, axis=0))
        fve = 1 - var_of_err / var_of_output
        return temp_ae.layers[1].get_weights(), encoded_data, fve

    def get_pca_fve(self, data=None):
        """compare the autoencoder against PCA"""
        if data is None: data = self._data_set
        pca = PCA(n_components=self._node_num[(len(self._node_num) - 1) / 2])
        actual_output = pca.inverse_transform(pca.fit_transform(data))
        return 1 - np.sum((actual_output - data).var(axis=0)) / np.sum(data.var(axis=0)), pca

    @staticmethod
    def get_outbound_layers(layer):
        """get outbound layers to which this layer is connected,
        see https://stackoverflow.com/questions/50814880/keras-retrieve-layers-that-the-layer-connected-to"""
        temp_nodes = layer._outbound_nodes
        return [item.outbound_layer for item in temp_nodes]

    def train(self, hierarchical=None, hierarchical_variant = None):
        """lag_time is included for training time-lagged autoencoder"""
        act_funcs = [item.lower() for item in self._hidden_layers_type] + [self._out_layer_type.lower()]
        if hierarchical is None: hierarchical = self._hierarchical
        if hierarchical_variant is None: hierarchical_variant = self._hi_variant
        node_num = self._node_num
        data = self._data_set
        if hasattr(self, '_output_data_set') and not self._output_data_set is None:
            print ("outputs may be different from inputs")
            output_data_set = self._output_data_set
        else:
            output_data_set = data

        num_CVs = node_num[self._index_CV] / 2 if act_funcs[self._index_CV - 1] == "circular" else \
            node_num[self._index_CV]
        if self._node_num[self._index_CV] == 1:
            hierarchical = False
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
            # self._output_data_set = output_data_set
            inputs_net = Input(shape=(node_num[0],))
            x = Dense(node_num[1], activation=act_funcs[0],
                      kernel_regularizer=l2(self._network_parameters[4][0]))(inputs_net)
            for item in range(2, self._index_CV):
                x = Dense(node_num[item], activation=act_funcs[item - 1], kernel_regularizer=l2(self._network_parameters[4][item - 1]))(x)
            if act_funcs[self._index_CV - 1] == "circular":
                # TODO: make this part consistent with else branch
                x = Dense(node_num[self._index_CV], activation='linear',
                            kernel_regularizer=l2(self._network_parameters[4][self._index_CV - 1]))(x)
                x = Reshape((num_CVs, 2), input_shape=(node_num[self._index_CV],))(x)
                x = Lambda(temp_lambda_func_for_circular_for_Keras)(x)
                encoded = Reshape((node_num[self._index_CV],))(x)
                encoded_split = [temp_lambda_slice_layers_circular[item](encoded) for item in range(num_CVs)]
            else:
                encoded_split = [Dense(1, activation=act_funcs[self._index_CV - 1],
                                kernel_regularizer=l2(self._network_parameters[4][self._index_CV - 1]))(x) for _ in range(num_CVs)]
                encoded = layers.Concatenate()(encoded_split)

            if hierarchical_variant == 0:  # this is logically equivalent to original version by Scholz
                x_next = [Dense(node_num[self._index_CV + 1], activation='linear',
                                kernel_regularizer=l2(self._network_parameters[4][self._index_CV]))(item) for item in encoded_split]
                x_next_1 = [x_next[0]]
                for item in range(2, len(x_next) + 1):
                    x_next_1.append(layers.Add()(x_next[:item]))
                if act_funcs[self._index_CV] == 'tanh':
                    x_next_1 = [temp_lambda_tanh_layer(item) for item in x_next_1]
                elif act_funcs[self._index_CV] == 'sigmoid':
                    x_next_1 = [temp_lambda_sigmoid_layer(item) for item in x_next_1]
                elif act_funcs[self._index_CV] == 'linear':
                    x_next_1 = x_next_1
                else:
                    raise Exception('activation function not implemented')
                assert (len(x_next) == len(x_next_1))
                for item_index in range(self._index_CV + 2, len(node_num) - 1):
                    x_next_1 = [Dense(node_num[item_index], activation=act_funcs[item_index - 1], kernel_regularizer=l2(self._network_parameters[4][item_index - 1]))(item_2)
                                for item_2 in x_next_1]
                shared_final_layer = Dense(node_num[-1], activation=act_funcs[-1],
                                           kernel_regularizer=l2(self._network_parameters[4][-1]))
                outputs_net = layers.Concatenate()([shared_final_layer(item) for item in x_next_1])
            elif hierarchical_variant == 1:   # simplified version, no shared layer after CV (encoded) layer
                concat_layers = [encoded_split[0]]
                concat_layers += [layers.Concatenate()(encoded_split[:item]) for item in range(2, num_CVs + 1)]
                x = [Dense(node_num[self._index_CV + 1], activation=act_funcs[self._index_CV],
                                kernel_regularizer=l2(self._network_parameters[4][self._index_CV]))(item) for item in concat_layers]
                for item_index in range(self._index_CV + 2, len(node_num) - 1):
                    x = [Dense(node_num[item_index], activation=act_funcs[item_index - 1],
                               kernel_regularizer=l2(self._network_parameters[4][item_index - 1]))(item) for item in x]
                x = [Dense(node_num[-1], activation=act_funcs[-1],
                                kernel_regularizer=l2(self._network_parameters[4][-1]))(item) for item in x]
                outputs_net = layers.Concatenate()(x)
            elif hierarchical_variant == 2:
                # boosted hierarchical autoencoders, CV i in encoded layer learns remaining error that has
                # not been learned by previous CVs
                x = [Dense(node_num[self._index_CV + 1], activation=act_funcs[self._index_CV],
                           kernel_regularizer=l2(self._network_parameters[4][self._index_CV]))(item) for item in encoded_split]
                for item_index in range(self._index_CV + 2, len(node_num) - 1):
                    x = [Dense(node_num[item_index], activation=act_funcs[item_index - 1],
                               kernel_regularizer=l2(self._network_parameters[4][item_index - 1]))(item) for item in x]
                x = [Dense(node_num[-1], activation=act_funcs[-1],
                           kernel_regularizer=l2(self._network_parameters[4][-1]))(item) for item in x]
                x_out = [x[0]]
                for item in range(2, len(x) + 1):
                    x_out.append(layers.Add()(x[:item]))
                assert (len(x_out) == len(x))
                outputs_net = layers.Concatenate()(x_out)
            else: raise Exception('error variant')
            encoder_net = Model(inputs=inputs_net, outputs=encoded)
            molecule_net = Model(inputs=inputs_net, outputs=outputs_net)
            # print molecule_net.summary()
            loss_function = get_mse_weighted(self._mse_weights)
        # elif num_of_hidden_layers != 3:
        #     raise Exception('not implemented for this case')
        else:
            inputs_net = Input(shape=(node_num[0],))
            x = Dense(node_num[1], activation=act_funcs[0],
                      kernel_regularizer=l2(self._network_parameters[4][0]))(inputs_net)
            for item in range(2, self._index_CV):
                x = Dense(node_num[item], activation=act_funcs[item - 1], kernel_regularizer=l2(self._network_parameters[4][item - 1]))(x)
            if act_funcs[self._index_CV - 1] == "circular":
                x = Dense(node_num[self._index_CV], activation='linear',
                            kernel_regularizer=l2(self._network_parameters[4][self._index_CV - 1]))(x)
                x = Reshape((node_num[self._index_CV] / 2, 2), input_shape=(node_num[self._index_CV],))(x)
                x = Lambda(temp_lambda_func_for_circular_for_Keras)(x)
                encoded = Reshape((node_num[self._index_CV],))(x)
            else:
                encoded = Dense(node_num[self._index_CV], activation=act_funcs[self._index_CV - 1],
                            kernel_regularizer=l2(self._network_parameters[4][self._index_CV - 1]))(x)
            x = Dense(node_num[self._index_CV + 1], activation=act_funcs[self._index_CV],
                      kernel_regularizer=l2(self._network_parameters[4][self._index_CV]))(encoded)
            for item_index in range(self._index_CV + 2, len(node_num)):
                x = Dense(node_num[item_index], activation=act_funcs[item_index - 1],
                          kernel_regularizer=l2(self._network_parameters[4][item_index - 1]))(x)
            molecule_net = Model(inputs=inputs_net, outputs=x)
            encoder_net = Model(inputs=inputs_net, outputs=encoded)
            loss_function = get_mse_weighted(self._mse_weights)

        try:
            from keras.utils import plot_model
            Helper_func.backup_rename_file_if_exists('model.png')
            plot_model(molecule_net, show_shapes=True, to_file='model.png')
        except: pass

        temp_optimizer_name = "Adam"
        if temp_optimizer_name == 'SGD':
            temp_optimizer = SGD(lr=self._network_parameters[0],
                                   momentum=self._network_parameters[1],
                                   decay=self._network_parameters[2],
                                   nesterov=self._network_parameters[3])
        elif temp_optimizer_name == 'Adam':
            temp_optimizer = Adam(lr=self._network_parameters[0])

        molecule_net.compile(loss=loss_function, metrics=[loss_function],
                             optimizer= temp_optimizer)
        # encoder_net.compile(loss=loss_function, metrics=[loss_function],
        #                      optimizer=temp_optimizer)  # not needed, but do not want to see endless warning...
        pretraining = False
        data_for_pretraining = self._data_set
        if pretraining:
            for index_layer in range(1, 3):   # TODO: currently only for first 2 Dense layers
                temp_weights, data_for_pretraining, fve = self.layerwise_pretrain(
                    data_for_pretraining, self._node_num[index_layer - 1], self._node_num[index_layer])
                molecule_net.layers[index_layer].set_weights(temp_weights)
                print "fve of pretraining for layer %d = %f" % (index_layer, fve)

        training_print_info = '''training, index = %d, maxEpochs = %d, node_num = %s, layers = %s, num_data = %d,
parameter = %s, optimizer = %s, hierarchical = %d with variant %d, FVE should not be less than %f (PCA)\n''' % (
            self._index, self._epochs, str(self._node_num),
            str(act_funcs), len(data), str(self._network_parameters), temp_optimizer_name,
            self._hierarchical, self._hi_variant, self.get_pca_fve()[0])

        print("Start " + training_print_info + str(datetime.datetime.now()))
        call_back_list = []
        earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')
        if self._enable_early_stopping:
            call_back_list += [earlyStopping]
        [train_in, train_out] = Helper_func.shuffle_multiple_arrays([data, output_data_set])
        train_history = molecule_net.fit(train_in, train_out, epochs=self._epochs, batch_size=self._batch_size,
                                         verbose=True, validation_split=0.2, callbacks=call_back_list)
        self._connection_between_layers_coeffs, self._connection_with_bias_layers_coeffs = [], []
        for item_l in encoder_net.layers:
            outbound_layers = autoencoder_Keras.get_outbound_layers(item_l)
            if len(outbound_layers) > 0 and isinstance(outbound_layers[0], Dense):
                self._connection_between_layers_coeffs.append(
                    np.concatenate([item_o.get_weights()[0] for item_o in outbound_layers], axis=-1).T.flatten())
                self._connection_with_bias_layers_coeffs.append(
                    np.concatenate([item_o.get_weights()[1] for item_o in outbound_layers], axis=-1))

        # self._connection_between_layers_coeffs = [item.get_weights()[0].T.flatten() for item in
        #                                           encoder_net.layers if isinstance(item, Dense)]  # transpose the weights for consistency
        # self._connection_with_bias_layers_coeffs = [item.get_weights()[1] for item in encoder_net.layers if
        #                                             isinstance(item, Dense)]

        print('Done ' + training_print_info + str(datetime.datetime.now()))
        self._molecule_net = molecule_net
        self._encoder_net = encoder_net
        try:
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(train_history.history['loss'])
            axes[1].plot(train_history.history['val_loss'])
            fig.suptitle(str(self._node_num) + str(self._network_parameters) + temp_optimizer_name)
            png_file = 'history_%02d.png' % self._index
            Helper_func.backup_rename_file_if_exists(png_file)
            fig.savefig(png_file)
        except: print("training history not plotted!"); pass
        return self, train_history


def temp_lambda_func_for_circular_for_Keras(x):
    """This has to be defined at the module level here, otherwise the pickle will not work
    """
    return x / ((x ** 2).sum(axis=2, keepdims=True).sqrt())

temp_lambda_tanh_layer = Lambda(lambda x: K.tanh(x))
temp_lambda_sigmoid_layer = Lambda(lambda x: K.sigmoid(x))
# not sure if there are better ways to do this, since Lambda layer has to be defined at top level of the file,
# following line does not work
# temp_lambda_slice_layers = [Lambda(lambda x: x[:, [index]], output_shape=(1,)) for index in range(20)]
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
# if CONFIG_44:
#     print "MSE is weighted by %s" % str(weight_for_MSE)

def get_mse_weighted(weight_for_MSE=None):   # take weight as input, return loss function
    if weight_for_MSE is None:
        weight_for_MSE = 1
    else:
        print("error weighted by %s" % str(weight_for_MSE))
    def mse_weighted(y_true, y_pred):
        return K.mean(K.variable(weight_for_MSE) * K.square(y_pred - y_true), axis=-1)  # TODO: do this later
        #  return K.mean(K.square(y_pred - y_true), axis=-1)
    return mse_weighted

mse_weighted = get_mse_weighted()      # requires a global mse_weighted(), for backward compatibility


###################################################
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

class AE_net(nn.Module):
    def __init__(self, node_num_1, node_num_2, activations=None, hierarchical=None, hi_variant=None):
        super(AE_net, self).__init__()
        self._activations = activations
        if self._activations is None:
            self._activations = [['tanh'] * (len(node_num_1) + len(node_num_2) - 3)]    # why -3, since last layer is linear
        encoder_act = self._activations[:(len(node_num_1) - 1)]
        decoder_act = self._activations[(len(node_num_1) - 1):]   # note final layer is linear
        # print encoder_act, decoder_act
        assert (len(decoder_act) == len(node_num_2) - 1), (len(decoder_act), len(node_num_2) - 1)
        self._hierarchical = hierarchical
        self._hi_variant = hi_variant
        encoder_list = [self.get_layer(node_num_1[item], node_num_1[item + 1], activation=encoder_act[item])
                        for item in range(len(node_num_1) - 2)]
        self._encoder_1 = nn.Sequential(*encoder_list)
        if not hierarchical:
            # use ModuleList instead of plain list for saving parameters
            self._encoder_2 = nn.ModuleList([self.get_layer(node_num_1[-2], node_num_1[-1],
                                                            activation=encoder_act[-1])])
            decoder_list = [self.get_layer(node_num_2[item], node_num_2[item + 1], activation=decoder_act[item])
                            for item in range(len(node_num_2) - 1)]
            self._decoder = nn.Sequential(*decoder_list)
        else:
            self._encoder_2 = nn.ModuleList([self.get_layer(node_num_1[-2], 1, activation=encoder_act[-1])
                               for _ in range(node_num_1[-1])])
            if hi_variant == 2:
                temp_node_num_2 = node_num_2[:]
                temp_node_num_2[0] = 1
                self._decoder = nn.ModuleList([nn.Sequential(*[
                    self.get_layer(temp_node_num_2[item], temp_node_num_2[item + 1], activation=decoder_act[item])
                    for item in range(len(temp_node_num_2) - 1)])
                                 for _ in range(node_num_2[0])])
            elif hi_variant == 1:
                decoder_list = []
                for num_item in range(node_num_2[0]):
                    temp_node_num_2 = node_num_2[:]
                    temp_node_num_2[0] = num_item + 1
                    decoder_list.append(
                        nn.Sequential(*[self.get_layer(
                            temp_node_num_2[item], temp_node_num_2[item + 1], activation=decoder_act[item])
                        for item in range(len(temp_node_num_2) - 1)]))
                self._decoder = nn.ModuleList(decoder_list)
        return

    @staticmethod
    def get_layer(in_node, out_node, activation):
        if activation == 'linear':
            return nn.Sequential(nn.Linear(in_node, out_node))
        elif activation == 'tanh':
            return nn.Sequential(nn.Linear(in_node, out_node), nn.Tanh())
        elif activation == 'sigmoid':
            return nn.Sequential(nn.Linear(in_node, out_node), nn.Sigmoid())

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            # use default initializer of Keras for now
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
        return

    def apply_weight_init(self):
        self.apply(AE_net.weights_init)   # Applies a function recursively to every submodule
        return

    def forward(self, x):
        temp = self._encoder_1(x)
        latent_z_split = [item_l(temp) for item_l in self._encoder_2]
        latent_z = torch.cat(latent_z_split, dim=-1)
        if not self._hierarchical:
            rec_x = self._decoder(latent_z)
        elif self._hi_variant == 2:
            temp_decoded = [self._decoder[item](latent_z_split[item]) for item in range(len(self._decoder))]
            decoded_list = [temp_decoded[0]]
            for item in temp_decoded[1:]:
                decoded_list.append(torch.add(decoded_list[-1], item))
            rec_x = torch.cat(decoded_list, dim=-1)
        elif self._hi_variant == 1:
            decoded_list = []
            for num_item in range(len(latent_z_split)):
                decoded_list.append(
                    self._decoder[num_item](torch.cat(latent_z_split[:num_item + 1], dim=-1)))
            rec_x = torch.cat(decoded_list, dim=-1)
        return rec_x, latent_z

class autoencoder_torch(autoencoder):
    class EarlyStoppingTorch(object):
        """modified from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d"""
        def __init__(self, patience=50):
            self._patience = patience
            self._num_bad_epochs = 0
            self._best = None
            self._is_better = lambda x, y: x < y

        def step(self, metrics):
            if self._best is None:
                self._best = metrics
                return False
            if np.isnan(metrics):
                return True
            if self._is_better(metrics, self._best):
                self._num_bad_epochs = 0
                self._best = metrics
            else:
                self._num_bad_epochs += 1

            if self._num_bad_epochs >= self._patience:
                return True
            return False

    class My_dataset(Dataset):
        def __init__(self, *data):
            # accept variable number of arrays to construct dataset
            self._data = data

        def __len__(self):
            return len(self._data[0])

        def __getitem__(self, index):
            return [item[index] for item in self._data if not item is None]

    def get_var_from_np(self, np_array, cuda=None, requires_grad=False):
        if cuda is None:
            cuda = self._cuda
        temp = Variable(torch.from_numpy(np_array.astype(np.float32)), requires_grad=requires_grad)
        if cuda: temp = temp.cuda()
        return temp

    def _init_extra(self,
                    network_parameters = CONFIG_4, cuda=True,
                    rec_loss_type = 0,      # 0: standard rec loss, 1: lagged rec loss, 2: no rec loss
                    rec_weight = 1,         # weight of reconstruction loss
                    autocorr_weight = 1,       # weight of autocorrelation loss in the loss function
                    pearson_weight = None,      # weight for pearson correlation loss for imposing orthogonality, None means no pearson loss
                    previous_CVs = None,       # previous CVs for sequential learning, requiring new CVs be orthogonal to them
                    start_from=None         # initialize with this model
                    ):
        self._network_parameters = network_parameters
        self._cuda = cuda
        self._rec_loss_type = rec_loss_type
        self._rec_weight = rec_weight
        self._autocorr_weight = autocorr_weight
        self._pearson_weight = pearson_weight
        self._previous_CVs = previous_CVs - previous_CVs.mean(axis=0) if not previous_CVs is None else None
        act_funcs = [item.lower() for item in self._hidden_layers_type] + [self._out_layer_type.lower()]
        if start_from is None:
            self._ae = AE_net(self._node_num[:self._index_CV + 1], self._node_num[self._index_CV:],
                                   activations=act_funcs, hierarchical=self._hierarchical,
                                   hi_variant=self._hi_variant)
            self._ae.apply_weight_init()
        else:
            self._ae = torch.load(start_from)
        if self._cuda: self._ae = self._ae.cuda()
        return

    def get_train_valid_split(self, dataset, valid_size=0.2):
        from torch.utils.data import SubsetRandomSampler
        assert (isinstance(dataset, self.My_dataset))
        # modified from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * len(indices)))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_loader = DataLoader(dataset, batch_size=self._batch_size,
                                  sampler=SubsetRandomSampler(train_idx), drop_last=False)
        valid_loader = DataLoader(dataset, batch_size=self._batch_size,
                                  sampler=SubsetRandomSampler(valid_idx), drop_last=False)
        return train_loader, valid_loader

    def train(self, lag_time=0):
        if self._output_data_set is None:
            self._output_data_set = self._data_set
        data_in, data_out = self._data_set, self._output_data_set
        temp_in_shape = data_in.shape
        if lag_time > 0:
            data_in = np.concatenate([data_in[:-lag_time], data_in[lag_time:]], axis=-1)
            if self._rec_loss_type == 1:
                data_out = data_out[lag_time:]
                self._data_set = self._data_set[:-lag_time]    # directly modify stored data, for convenience of computing FVE
                self._output_data_set = self._output_data_set[lag_time:]
            else:  # otherwise use standard reconstruction loss
                data_out = data_out[:-lag_time]
        else:
            data_in = np.concatenate([data_in, data_in], axis=-1)

        assert (data_in.shape[0] == temp_in_shape[0] - lag_time), (data_in.shape[0], temp_in_shape[0] - lag_time)
        assert (data_in.shape[1] == 2 * temp_in_shape[1]), (data_in.shape[1], 2 * temp_in_shape[1])
        if self._hierarchical:
            num_CVs = self._node_num[self._index_CV]
            data_out = np.concatenate([data_out] * num_CVs, axis=-1)
        if self._previous_CVs is None:
            all_data = self.My_dataset(self.get_var_from_np(data_in).data,
                                       self.get_var_from_np(data_out).data)
        else:
            all_data = self.My_dataset(self.get_var_from_np(data_in).data,
                                       self.get_var_from_np(data_out).data,
                                       self.get_var_from_np(self._previous_CVs).data)
        train_set, valid_set = self.get_train_valid_split(all_data)
        print """
data size = %d, train set size = %d, valid set size = %d, batch size = %d, rec_weight = %f, autocorr_weight = %f
""" % (
            len(all_data), len(train_set), len(valid_set), self._batch_size, self._rec_weight, self._autocorr_weight)
        optimizer = torch.optim.Adam(self._ae.parameters(), lr=self._network_parameters[0], weight_decay=0)
        self._ae.train()    # set to training mode
        train_history, valid_history = [], []
        my_early_stopping = self.EarlyStoppingTorch(patience=100)

        for index_epoch in range(self._epochs):
            temp_train_history, temp_valid_history = [], []
            print index_epoch
            # training
            for item_batch in train_set:
                if len(item_batch) == 2:   # without previous CVs
                    loss, rec_loss = self.get_loss(item_batch[0], item_batch[1], temp_in_shape[1])
                elif len(item_batch) == 3:
                    loss, rec_loss = self.get_loss(item_batch[0], item_batch[1], temp_in_shape[1],
                                                   previous_CVs=item_batch[2])
                plot_model_loss = False
                if plot_model_loss:
                    from torchviz import make_dot, make_dot_from_trace
                    model_plot = make_dot(loss)
                    model_plot.save('temp_model.dot')  # save model plot for visualization
                optimizer.zero_grad()
                loss.backward()
                loss_list = np.array([loss.cpu().data.numpy()])
                # print loss_list
                temp_train_history.append(loss_list)
                optimizer.step()
            train_history.append(np.array(temp_train_history).mean(axis=0))

            # validation
            for item_batch in valid_set:
                with torch.no_grad():
                    if len(item_batch) == 2:
                        loss, rec_loss = self.get_loss(item_batch[0], item_batch[1], temp_in_shape[1])
                    elif len(item_batch) == 3:
                        loss, rec_loss = self.get_loss(item_batch[0], item_batch[1], temp_in_shape[1],
                                                       previous_CVs=item_batch[2])
                loss_list = np.array([loss.cpu().data.numpy()])
                temp_valid_history.append(loss_list)
            temp_valid_history = np.array(temp_valid_history).mean(axis=0)
            valid_history.append(temp_valid_history)
            if my_early_stopping.step(temp_valid_history[-1]):    # monitor loss
                print "best in history is %f, current is %f" % (my_early_stopping._best, temp_valid_history[-1])
                break             # early stopping
        try:
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(train_history)
            axes[1].plot(valid_history)
            fig.suptitle(str(self._node_num) + str(self._network_parameters))
            png_file = 'history_%s.png' % os.path.basename(self._filename_to_save_network)
            Helper_func.backup_rename_file_if_exists(png_file)
            fig.savefig(png_file)
        except:
            try:
                print("training history not plotted! save history into npy file instead")
                history_npy = 'history_%02d.npy' % self._index
                Helper_func.backup_rename_file_if_exists(history_npy)
                np.save(history_npy, np.array([train_history, valid_history]))
            except: pass
        return

    def get_loss(self, batch_in, batch_out, dim_input, previous_CVs=None):
        """previous_CVs are for Pearson loss only"""
        rec_x, latent_z_1 = self._ae(Variable(batch_in[:, :dim_input]))
        if self._rec_loss_type == 2:
            rec_loss = 0
        else:
            rec_loss = nn.MSELoss()(rec_x, Variable(batch_out))
        if self._autocorr_weight > 0:
            _, latent_z_2 = self._ae(Variable(batch_in[:, dim_input:]))
            latent_z_1 = latent_z_1 - torch.mean(latent_z_1, dim=0)
            # print latent_z_1.shape
            latent_z_2 = latent_z_2 - torch.mean(latent_z_2, dim=0)
            constraint_type = 'natural'
            if constraint_type == 'regularization':
                autocorr_loss_num = torch.mean(latent_z_1 * latent_z_2, dim=0)
                autocorr_loss_den = torch.norm(latent_z_1, dim=0) * torch.norm(latent_z_2, dim=0)
                # print autocorr_loss_num.shape, autocorr_loss_den.shape
                autocorr_loss = - torch.sum(autocorr_loss_num / autocorr_loss_den)
                # add pearson correlation loss
                if not (self._pearson_weight is None or self._pearson_weight == 0):   # include pearson correlation for first two CVs as loss function
                    new_CVs = [latent_z_1[:, index] for index in range(2)]
                    pearson_corr = 0
                    for xx in range(len(new_CVs) - 1):    # pairwise Pearson loss
                        for yy in range(xx + 1, len(new_CVs)):
                            pearson_corr += torch.sum(new_CVs[xx] * new_CVs[yy]) ** 2 / (
                                    torch.sum(new_CVs[xx] ** 2) * torch.sum(new_CVs[yy] ** 2))
                    if not previous_CVs is None:        # Pearson loss with respect to previous CVs
                        for item_new_CV in new_CVs:
                            for item_old_CV in torch.transpose(previous_CVs, 0, 1):
                                pearson_corr += torch.sum(item_new_CV * item_old_CV) ** 2 / (
                                    torch.sum(item_new_CV ** 2) * torch.sum(item_old_CV ** 2))
                    # print pearson_corr.cpu().data.numpy()
                    autocorr_loss = autocorr_loss + self._pearson_weight * pearson_corr
            elif constraint_type == 'natural':
                if previous_CVs is None: raise Exception('not implemented')
                for item_old_CV in torch.transpose(previous_CVs, 0, 1):
                    scaling_factor = torch.mean(item_old_CV * item_old_CV)
                    item_old_CV = item_old_CV.reshape(item_old_CV.shape[0], 1)
                    latent_z_1 = latent_z_1 - item_old_CV * torch.mean(latent_z_1 * item_old_CV) / scaling_factor
                    latent_z_2 = latent_z_2 - item_old_CV * torch.mean(latent_z_2 * item_old_CV) / scaling_factor
                    # print torch.mean(latent_z_1, dim=0).cpu().data.numpy(), torch.max(latent_z_1, dim=0)[0].cpu().data.numpy()
                    assert (latent_z_1.shape[1] == 2)
                    autocorr_loss_num = torch.mean(latent_z_1 * latent_z_2, dim=0)
                    autocorr_loss_den = torch.norm(latent_z_1, dim=0) * torch.norm(latent_z_2, dim=0)
                    temp_ratio = autocorr_loss_num / autocorr_loss_den
                    print temp_ratio.cpu().data.numpy()
                    autocorr_loss = - torch.sum(autocorr_loss_num / autocorr_loss_den)
            loss = self._rec_weight * rec_loss + self._autocorr_weight * autocorr_loss
        else:
            if self._autocorr_weight != 1.0:
                print ('warning: autocorrelation loss weight has no effect for model with reconstruction loss only')
            if self._rec_weight != 1.0:
                print ('warning: reconstruction loss weight has no effect for model with reconstruction loss only')
            loss = rec_loss
        return loss, rec_loss

    def save_into_file(self, filename=CONFIG_6, fraction_of_data_to_be_saved = 1.0):
        if filename is None:
            filename = self._filename_to_save_network
        folder_to_store_files = os.path.dirname(filename)
        if folder_to_store_files != '' and (not os.path.exists(folder_to_store_files)):
            os.makedirs(folder_to_store_files)
        # save both model and model parameters
        torch.save(self._ae, filename.replace('.pkl', '.pth'))
        torch.save(self._ae.state_dict(), filename.replace('.pkl', '_params.pth'))
        self._ae = None    # do not save model in pkl file
        data_file_paths = self.helper_save_data(filename)
        with open(filename, 'wb') as my_file:
            pickle.dump(self, my_file, pickle.HIGHEST_PROTOCOL)
        self._ae = torch.load(filename.replace('.pkl', '.pth'))
        self.helper_load_data(filename)
        return

    @staticmethod
    def load_from_pkl_file(filename):
        a = Sutils.load_object_from_pkl_file(filename)
        a._ae = torch.load(filename.replace('.pkl', '.pth'))
        assert (isinstance(a, autoencoder_torch))
        a.helper_load_data(filename)
        return a

    def get_output_data(self, input_data=None):
        if input_data is None: input_data = self._data_set
        self._ae.eval()
        with torch.no_grad():
            result = self._ae(self.get_var_from_np(input_data))[0]
        if self._cuda: result = result.cpu()
        return result.data.numpy()

    def get_PCs(self, input_data=None):
        if input_data is None: input_data = self._data_set
        self._ae.eval()
        with torch.no_grad():
            result = self._ae(self.get_var_from_np(input_data))[1]
        if self._cuda: result = result.cpu()
        return result.data.numpy()