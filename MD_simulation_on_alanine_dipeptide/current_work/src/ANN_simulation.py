import copy, pickle, re, os, time, subprocess, datetime, itertools, sys
from scipy import io as sciio
import numpy as np, pandas as pd, seaborn as sns
from numpy.testing import assert_almost_equal
from math import *
from pybrain.structure import *
from pybrain.structure.modules.circularlayer import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
import matplotlib.pyplot as plt
from config import * # configuration file
from cluster_management import *
from molecule_spec_sutils import *  # import molecule specific unitity code
from sklearn.neighbors import RadiusNeighborsRegressor

"""note that all configurations for a class should be in function __init__(), and take configuration parameters
from config.py
"""

##################    set types of molecules  ############################

if CONFIG_30 == "Alanine_dipeptide":
    molecule_type = Alanine_dipeptide()
elif CONFIG_30 == "Trp_cage":
    molecule_type = Trp_cage()
else:
    raise Exception("molecule type not found")

##########################################################################

class coordinates_data_files_list(object):
    def __init__(self,
                list_of_dir_of_coor_data_files = CONFIG_1, # this is the directory that holds corrdinates data files
                ):
        self._list_of_dir_of_coor_data_files = list_of_dir_of_coor_data_files
        self._list_of_coor_data_files = []

        for item in self._list_of_dir_of_coor_data_files:
            self._list_of_coor_data_files += subprocess.check_output(['find', item,'-name' ,'*coordinates.txt']).strip().split('\n')

        self._list_of_coor_data_files = list(set(self._list_of_coor_data_files))  # remove duplicates
        self._list_of_coor_data_files.sort()                # to be consistent
        self._list_of_line_num_of_coor_data_file = map(lambda x: int(subprocess.check_output(['wc', '-l', x]).strip().split()[0]),
                                                       self._list_of_coor_data_files)

        return

    def get_list_of_coor_data_files(self):
        return self._list_of_coor_data_files

    def get_list_of_corresponding_pdb_files(self):
        list_of_corresponding_pdb_files = map(lambda x: x.strip().split('_coordinates.txt')[0] + '.pdb',
                                              self.get_list_of_coor_data_files()
                                              )
        for item in list_of_corresponding_pdb_files:
            try:
                assert os.path.exists(item)
            except:
                raise Exception('%s does not exist!' % item)

        return list_of_corresponding_pdb_files

    def get_list_of_line_num_of_coor_data_file(self):
        return self._list_of_line_num_of_coor_data_file

    def write_pdb_frames_into_file_with_list_of_coor_index(self, list_of_coor_index, out_file_name):
        """
        This function picks several frames from pdb files, and write a new pdb file as output,
        we could use this together with the mouse-clicking callback implemented in the scatter plot:
        first we select a few points interactively in the scatter plot, and get corresponding index in the data point
        list, the we find the corresponding pdb frames with the index
        """
        if os.path.isfile(out_file_name):  # backup files
            os.rename(out_file_name,
                      out_file_name.split('.pdb')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pdb")

        list_of_coor_index.sort()
        pdb_files = self.get_list_of_corresponding_pdb_files()
        accum_sum = np.cumsum(np.array(self._list_of_line_num_of_coor_data_file))  # use accumulative sum to find corresponding pdb files
        for item in range(len(accum_sum)):
            if item == 0:
                temp_index_related_to_this_pdb_file = filter(lambda x: x < accum_sum[item], list_of_coor_index)
            else:
                temp_index_related_to_this_pdb_file = filter(lambda x: accum_sum[item - 1] <= x < accum_sum[item], list_of_coor_index)
                temp_index_related_to_this_pdb_file = map(lambda x: x - accum_sum[item - 1], temp_index_related_to_this_pdb_file)
            temp_index_related_to_this_pdb_file.sort()

            if len(temp_index_related_to_this_pdb_file) != 0:
                print(pdb_files[item])
                with open(pdb_files[item], 'r') as in_file:
                    content = in_file.read().split('MODEL')[1:]  # remove header
                    frames_to_use = [content[ii] for ii in temp_index_related_to_this_pdb_file]
                    with open(out_file_name, 'a') as out_file:
                        for frame in frames_to_use:
                            out_file.write("MODEL" + frame)

        return


class neural_network_for_simulation(object):
    """the neural network for simulation"""

    def __init__(self,
                 index,  # the index of the current network
                 data_set_for_training,
                 autoencoder_info_file = None,  # this might be expressions, or coefficients
                 training_data_interval = CONFIG_2,
                 in_layer_type = LinearLayer,
                 hidden_layers_types = CONFIG_17,
                 out_layer_type = LinearLayer,  # different layers
                 node_num = CONFIG_3,  # the structure of ANN
                 network_parameters = CONFIG_4,  # includes [learningrate,momentum, weightdecay, lrdecay]
                 max_num_of_training = CONFIG_5,
                 filename_to_save_network = CONFIG_6,
                 network_verbose = False,
                 trainer = None
                 ):

        self._index = index
        self._data_set = data_set_for_training
        self._training_data_interval = training_data_interval
        if autoencoder_info_file is None:
            self._autoencoder_info_file = "../resources/%s/autoencoder_info_%d.txt" %(CONFIG_30 ,index)
        else:
            self._autoencoder_info_file = autoencoder_info_file

        if not in_layer_type is None: self._in_layer_type = in_layer_type
        if not hidden_layers_types is None: self._hidden_layers_type = hidden_layers_types
        if not out_layer_type is None: self._out_layer_type = out_layer_type

        self._in_layer = None
        self._out_layer = None
        self._hidden_layers = None

        self._node_num = node_num
        self._network_parameters = network_parameters
        self._max_num_of_training = max_num_of_training
        if filename_to_save_network is None:
            self._filename_to_save_network = "../resources/%s/network_%s.pkl" % (CONFIG_30, str(self._index)) # by default naming with its index
        else:
            self._filename_to_save_network = filename_to_save_network

        self._network_verbose = network_verbose

        self._trainer = trainer  # save the trainer so that we could train this network step by step later
        return

    def save_into_file(self, filename = CONFIG_6):
        if filename is None:
            filename = self._filename_to_save_network

        if os.path.isfile(filename):  # backup file if previous one exists
            os.rename(filename, filename.split('.pkl')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.pkl')

        with open(filename, 'wb') as my_file:
            pickle.dump(self, my_file, pickle.HIGHEST_PROTOCOL)
        return

    def get_expression_of_network(self):
        """
        this function generates expression of PCs in terms of inputs
        """
        # FIXME: the expression no longer works, since I made input list for input layer of autoencoder consistent
        # for both alanine dipeptide and trp-cage, always [cos, sin, cos, sin ....], 
        # which is consistent with ANN_Force, instead of [cos, cos, cos, cos, sin, sin, sin, sin]
        type_of_middle_hidden_layer = self._hidden_layers_type[1]

        connection_between_layers = self._connection_between_layers
        connection_with_bias_layers = self._connection_with_bias_layers

        node_num = self._node_num
        expression = ""

        # 1st part: network
        for i in range(2):
            expression = '\n' + expression
            mul_coef = connection_between_layers[i].params.reshape(node_num[i + 1], node_num[i])
            bias_coef = connection_with_bias_layers[i].params

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
                    temp_expression += 'radius_of_circular_pair_%d = sqrt( in_layer_%d_unit_%d * in_layer_%d_unit_%d + in_layer_%d_unit_%d * in_layer_%d_unit_%d );\n'  \
                                    % (j, i + 1, 2 * j, i + 1, 2 * j , i + 1, 2 * j + 1, i + 1, 2 * j + 1)

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

    def write_expression_into_file(self, out_file = None):
        if out_file is None: out_file = self._autoencoder_info_file

        expression = self.get_expression_of_network()
        with open(out_file, 'w') as f_out:
            f_out.write(expression)
        return

    def write_coefficients_of_connections_into_file(self, out_file = None):
        if out_file is None: out_file = self._autoencoder_info_file

        with open(out_file, 'w') as f_out:
            for item in [0, 1]:
                f_out.write(str(list(self._connection_between_layers[item].params)))
                f_out.write(',\n')

            for item in [0, 1]:
                f_out.write(str(list(self._connection_with_bias_layers[item].params)))
                f_out.write(',\n')
        return

    def get_mid_result(self, input_data=None):
        if input_data is None: input_data = self._data_set
        connection_between_layers = self._connection_between_layers
        connection_with_bias_layers = self._connection_with_bias_layers

        node_num = self._node_num
        num_of_hidden_layers = len(self._hidden_layers_type)

        temp_mid_result = range(num_of_hidden_layers + 1)
        temp_mid_result_in = range(num_of_hidden_layers + 1)
        mid_result = []

        data_as_input_to_network = input_data

        hidden_and_out_layers = self._hidden_layers + [self._out_layer]

        for item in data_as_input_to_network:
            for i in range(num_of_hidden_layers + 1):
                mul_coef = connection_between_layers[i].params.reshape(node_num[i + 1], node_num[i]) # fix node_num
                bias_coef = connection_with_bias_layers[i].params
                previous_result = item if i == 0 else temp_mid_result[i - 1]
                temp_mid_result_in[i] = np.dot(mul_coef, previous_result) + bias_coef
                output_of_this_hidden_layer = range(len(temp_mid_result_in[i]))  # initialization
                hidden_and_out_layers[i]._forwardImplementation(temp_mid_result_in[i], output_of_this_hidden_layer)
                temp_mid_result[i] = output_of_this_hidden_layer

            mid_result.append(copy.deepcopy(temp_mid_result)) # note that should use deepcopy
        return mid_result

    def get_PCs(self, input_data = None):
        """
        write an independent function for getting PCs, since it is different for TanhLayer, and CircularLayer
        """
        if input_data is None: input_data = self._data_set
        num_of_hidden_layers = len(self._hidden_layers_type)
        index_of_bottleneck_hidden_layer = (num_of_hidden_layers - 1) / 2   # it works for both 3-layer and 5-layer structure
        type_of_middle_hidden_layer = self._hidden_layers_type[index_of_bottleneck_hidden_layer]
        temp_mid_result = self.get_mid_result(input_data=input_data)
        mid_result_1 = [item[index_of_bottleneck_hidden_layer] for item in temp_mid_result]
        
        if type_of_middle_hidden_layer == CircularLayer:
            PCs = [[acos(item[0]) * np.sign(item[1]), acos(item[2]) * np.sign(item[3])] for item in mid_result_1]
        else:
            PCs = mid_result_1

        if self._hidden_layers_type[1] == CircularLayer:
            assert (len(PCs[0]) == self._node_num[2] / 2)
        else:
            assert (len(PCs[0]) == self._node_num[2])

        return PCs

    def train(self):

        ####################### set up autoencoder begin #######################
        node_num = self._node_num

        in_layer = (self._in_layer_type)(node_num[0], "IL")
        num_of_hidden_layers = len(self._hidden_layers_type)

        if num_of_hidden_layers == 3:  # 5-layer autoencoder
            hidden_layers = [(self._hidden_layers_type[0])(node_num[1], "HL1"),
                             (self._hidden_layers_type[1])(node_num[2], "HL2"),
                             (self._hidden_layers_type[2])(node_num[3], "HL3")]
            bias_layers = [BiasUnit("B1"),BiasUnit("B2"),BiasUnit("B3"),BiasUnit("B4")]
        elif num_of_hidden_layers == 1:
            hidden_layers = [(self._hidden_layers_type[0])(node_num[1], "HL1")]
            bias_layers = [BiasUnit("B1"),BiasUnit("B2")]

        out_layer = (self._out_layer_type)(node_num[num_of_hidden_layers + 1], "OL")

        self._in_layer = in_layer
        self._out_layer = out_layer
        self._hidden_layers = hidden_layers

        layers_list = [in_layer] + hidden_layers + [out_layer]

        molecule_net = FeedForwardNetwork()

        molecule_net.addInputModule(in_layer)
        for item in (hidden_layers + bias_layers):
            molecule_net.addModule(item)

        molecule_net.addOutputModule(out_layer)

        connection_between_layers = range(num_of_hidden_layers + 1)
        connection_with_bias_layers = range(num_of_hidden_layers + 1)

        for i in range(num_of_hidden_layers + 1):
            connection_between_layers[i] = FullConnection(layers_list[i], layers_list[i+1])
            connection_with_bias_layers[i] = FullConnection(bias_layers[i], layers_list[i+1])
            molecule_net.addConnection(connection_between_layers[i])  # connect two neighbor layers
            molecule_net.addConnection(connection_with_bias_layers[i])

        molecule_net.sortModules()  # this is some internal initialization process to make this module usable

        ####################### set up autoencoder end #######################

        trainer = BackpropTrainer(molecule_net, learningrate=self._network_parameters[0],
                                                momentum=self._network_parameters[1],
                                                weightdecay=self._network_parameters[2],
                                                lrdecay=self._network_parameters[3],
                                                verbose=self._network_verbose)
        data_set = SupervisedDataSet(node_num[0], node_num[num_of_hidden_layers + 1])

        sincos = self._data_set[::self._training_data_interval]  # pick some of the data to train
        data_as_input_to_network = sincos

        for item in data_as_input_to_network:
            data_set.addSample(item, item)

        print('start training network with index = %d, training maxEpochs = %d, structure = %s, layers = %s\n' %
              (self._index, self._max_num_of_training, str(self._node_num), str(self._hidden_layers_type).replace("class 'pybrain.structure.modules.", '')))
        trainer.trainUntilConvergence(data_set, maxEpochs=self._max_num_of_training)

        self._connection_between_layers = connection_between_layers
        self._connection_with_bias_layers = connection_with_bias_layers

        print('Done training network with index = %d, training maxEpochs = %d, structure = %s, layers = %s\n' %
              (self._index, self._max_num_of_training, str(self._node_num), str(self._hidden_layers_type).replace("class 'pybrain.structure.modules.", '')))
        self._trainer = trainer
        return

    def get_training_error(self):
        # it turns out that this error info cannot be a good measure of the quality of the autoencoder
        num_of_hidden_layers = len(self._hidden_layers_type)
        input_data = np.array(self._data_set)
        output_data = np.array([item[num_of_hidden_layers] for item in self.get_mid_result()])
        return np.linalg.norm(input_data - output_data) / sqrt(self._node_num[0] * len(input_data))

    def get_fraction_of_variance_explained(self):
        input_data = np.array(self._data_set)
        num_of_hidden_layers = len(self._hidden_layers_type)

        output_data = np.array([item[num_of_hidden_layers] for item in self.get_mid_result()])
        var_of_input = sum(np.var(input_data, axis=0))
        var_of_err = sum(np.var(output_data - input_data, axis=0))
        return 1 - var_of_err / var_of_input

    def get_commands_for_further_biased_simulations(self,list_of_potential_center = None,
                                                  num_of_simulation_steps = None,
                                                  autoencoder_info_file=None,
                                                  force_constant_for_biased = None,
                                                  ):
        '''this function creates a list of commands for further biased simulations that should be done later,
        either in local machines or on the cluster
        '''
        PCs_of_network = self.get_PCs()
        if self._hidden_layers_type[1] == CircularLayer:
            assert (len(PCs_of_network[0]) == self._node_num[2] / 2)
        else:
            assert (len(PCs_of_network[0]) == self._node_num[2])

        if list_of_potential_center is None:
            list_of_potential_center = molecule_type.get_boundary_points(list_of_points= PCs_of_network)
        if num_of_simulation_steps is None:
            num_of_simulation_steps = CONFIG_8
        if autoencoder_info_file is None:
            autoencoder_info_file = self._autoencoder_info_file
        if force_constant_for_biased is None:
            force_constant_for_biased = CONFIG_9

        todo_list_of_commands_for_simulations = []

        for potential_center in list_of_potential_center:
            if isinstance(molecule_type, Alanine_dipeptide):
                parameter_list = (str(CONFIG_16), str(num_of_simulation_steps), str(force_constant_for_biased),
                            '../target/Alanine_dipeptide/network_%d' % (self._index),
                                  autoencoder_info_file,
                            'pc_' + str(potential_center).replace(' ','')[1:-1],  # need to remove white space, otherwise parsing error
                            '../resources/Alanine_dipeptide/network_%d.pkl' % (self._index)
                            )
                command = "python ../src/biased_simulation.py %s %s %s %s %s %s --fc_adjustable --autoencoder_file %s --remove_previous" \
                          % parameter_list
            elif isinstance(molecule_type, Trp_cage):
                # FIXME: this is outdated, should be fixed
                parameter_list =  (str(CONFIG_16), str(num_of_simulation_steps), str(force_constant_for_biased),
                                   '../target/Trp_cage/network_%d/' % (self._index),
                                   autoencoder_info_file,
                                   'pc_' + str(potential_center).replace(' ', '')[1:-1],
                                   CONFIG_40, 'NVT')
                command = "python ../src/biased_simulation_Trp_cage.py %s %s %s %s %s %s %s %s" % parameter_list
                pass
                # command = "python ../src/biased_simulation_Trp_cage.py %s %s %s %s %s %s with_water 500" % parameter_list
            else:
                raise Exception("molecule type not defined")

            todo_list_of_commands_for_simulations += [command]

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
        distance_cal = lambda x,y: sqrt(np.dot(x-y,x-y))

        proper_potential_centers = []

        for item in list_of_points:
            distances = map(lambda x: distance_cal(item, x),
                            data_points
                            )
            neighbors_num = len(filter(lambda x: x < threshold_radius,
                                       distances))
            if neighbors_num >= min_num_of_neighbors:
                proper_potential_centers += [item]

        return proper_potential_centers

    def generate_mat_file_for_WHAM_reweighting(self, directory_containing_coor_files, folder_to_store_files = './standard_WHAM/'):
        if folder_to_store_files[-1] != '/':
            folder_to_store_files += '/'
        if not os.path.exists(folder_to_store_files):
            subprocess.check_output(['mkdir', folder_to_store_files])

        list_of_coor_data_files = coordinates_data_files_list([directory_containing_coor_files])._list_of_coor_data_files
        force_constants = []
        harmonic_centers = []
        window_counts = []
        coords = []
        umbOP = []
        for item in list_of_coor_data_files:
            # print('processing %s' %item)
            temp_force_constant = float(item.split('output_fc_')[1].split('_pc_')[0])
            force_constants += [[temp_force_constant, temp_force_constant]]
            harmonic_centers += [[float(item.split('_pc_[')[1].split(',')[0]), float(item.split(',')[1].split(']')[0])]]
            temp_window_count = float(subprocess.check_output(['wc', '-l', item]).split()[0])  # there would be some problems if using int
            window_counts += [temp_window_count]
            temp_coor = self.get_PCs(molecule_type.get_many_cossin_from_coordiantes_in_list_of_files([item]))
            assert(temp_window_count == len(temp_coor))  # ensure the number of coordinates is window_count
            coords += temp_coor
            if isinstance(molecule_type, Alanine_dipeptide):
                temp_angles = molecule_type.get_many_dihedrals_from_coordinates_in_file([item])
                temp_umbOP = [a[1:3] for a in temp_angles]
                assert(temp_window_count == len(temp_umbOP))
                assert(2 == len(temp_umbOP[0]))
                umbOP += temp_umbOP

        max_of_coor = map(lambda x: round(x, 1) + 0.1, map(max, zip(*coords)))
        min_of_coor = map(lambda x: round(x, 1) - 0.1, map(min, zip(*coords)))
        interval = 0.1

        window_counts = np.array(window_counts)
        sciio.savemat(folder_to_store_files + 'WHAM_nD__preprocessor.mat', {'window_counts': window_counts,
            'force_constants': force_constants, 'harmonic_centers': harmonic_centers,
            'coords': coords, 'dim': 2.0, 'temperature': 300.0, 'periodicity': [[1.0],[1.0]],
            'dF_tol': 0.0001,
            'min_gap_max_ORIG': [[min_of_coor[0], interval, max_of_coor[0]], [min_of_coor[1], interval, max_of_coor[1]]]
            })
        sciio.savemat(folder_to_store_files + 'umbrella_OP.mat',
            {'umbOP': umbOP
            })
        return

    def generate_files_for_Bayes_WHAM(self, directory_containing_coor_files, folder_to_store_files = './wham_files/'):
        list_of_coor_data_files = coordinates_data_files_list([directory_containing_coor_files])._list_of_coor_data_files
        for item in ['bias', 'hist', 'traj', 'traj_proj']:
            directory = folder_to_store_files + item
            subprocess.check_output(['mkdir', '-p', directory])
            assert (os.path.exists(directory))

        force_constants = []
        harmonic_centers = []
        window_counts = []
        coords = []
        umbOP = []
        for item in list_of_coor_data_files:
            # print('processing %s' %item)
            temp_force_constant = float(item.split('output_fc_')[1].split('_pc_')[0])
            force_constants += [[temp_force_constant, temp_force_constant]]
            harmonic_centers += [[float(item.split('_pc_[')[1].split(',')[0]), float(item.split(',')[1].split(']')[0])]]
            temp_window_count = float(subprocess.check_output(['wc', '-l', item]).split()[0])  # there would be some problems if using int
            window_counts += [temp_window_count]
            temp_coor = self.get_PCs(molecule_type.get_many_cossin_from_coordiantes_in_list_of_files([item]))
            assert(temp_window_count == len(temp_coor))  # ensure the number of coordinates is window_count
            coords += temp_coor
            temp_angles = molecule_type.get_many_dihedrals_from_coordinates_in_file([item])
            temp_umbOP = [a[1:3] for a in temp_angles]
            assert(temp_window_count == len(temp_umbOP))
            assert(2 == len(temp_umbOP[0]))
            umbOP += temp_umbOP


        # write info into files
        # 1st: bias potential info
        with open(folder_to_store_files + 'bias/harmonic_biases.txt', 'w') as f_out:
            for item in range(len(force_constants)):
                temp = '%d\t%f\t%f\t%f\t%f\n' % (item + 1, harmonic_centers[item][0], harmonic_centers[item][1],
                                                force_constants[item][0], force_constants[item][1])
                f_out.write(temp)

        # 2nd: trajectory, and projection trajectory in phi-psi space (for reweighting), and histogram
        num_of_bins = 40
        binEdges = np.array([np.linspace(-np.pi, np.pi, num_of_bins), np.linspace(-np.pi, np.pi, num_of_bins)])
        with open(folder_to_store_files + 'hist/hist_binEdges.txt', 'w') as f_out:
            for row in binEdges:
                for item in row:
                    f_out.write('%f\t' % item)
                f_out.write('\n')

        binEdges_proj = np.array([np.linspace(-np.pi, np.pi, num_of_bins), np.linspace(-np.pi, np.pi, num_of_bins)])
        with open(folder_to_store_files + 'hist/hist_binEdges_proj.txt', 'w') as f_out:
            for row in binEdges_proj:
                for item in row:
                    f_out.write('%f\t' % item)
                f_out.write('\n')

        start_index = end_index = 0
        for item, count in enumerate(window_counts):
            start_index = int(end_index)
            end_index = int(start_index + count)
            with open(folder_to_store_files + 'traj/traj_%d.txt' % (item + 1), 'w') as f_out_1, \
                 open(folder_to_store_files + 'traj_proj/traj_%d.txt' % (item + 1), 'w') as f_out_2, \
                 open(folder_to_store_files + 'hist/hist_%d.txt' % (item + 1), 'w') as f_out_3:
                for line in coords[start_index:end_index]:
                    temp = '%f\t%f\n' % (line[0], line[1])
                    f_out_1.write(temp)

                for line in umbOP[start_index:end_index]:
                    temp = '%f\t%f\n' % (line[0], line[1])
                    f_out_2.write(temp)

                x = [item[0] for item in coords[start_index:end_index]]
                y = [item[1] for item in coords[start_index:end_index]]
                temp_hist, _, _ = np.histogram2d(y, x, bins=(binEdges[0], binEdges[1]))
                for row in temp_hist:
                    for item in row:
                        f_out_3.write('%d\t' % item)

        return


class plotting(object):
    """this class implements different plottings
    """

    def __init__(self, network, related_coor_list_obj = None):
        assert isinstance(network, neural_network_for_simulation)
        self._network = network
        self._related_coor_list_obj = related_coor_list_obj
        pass

    def plotting_with_coloring_option(self, plotting_space,  # means "PC" space or "phi-psi" space
                                            fig_object,
                                            axis_object,
                                            network=None,
                                            cossin_data_for_plotting=None,
                                            color_option='pure',
                                            other_coloring=None,
                                            contain_title=True,
                                            title=None,
                                            axis_ranges=None,
                                            contain_colorbar=True,
                                            smoothing_using_RNR = False,   # smooth the coloring values for data points using RadiusNeighborsRegressor()
                                            smoothing_radius = 0.1
                                      ):
        """
        by default, we are using training data, and we also allow external data input
        """
        if network is None: network = self._network
        if title is None: title = "plotting in %s, coloring with %s" % (plotting_space, color_option)  # default title
        if cossin_data_for_plotting is None:
            cossin_data = self._network._data_set
        else:
            cossin_data = cossin_data_for_plotting

        if plotting_space == "PC":
            PCs_to_plot = network.get_PCs(input_data= cossin_data)

            (x, y) = ([item[0] for item in PCs_to_plot], [item[1] for item in PCs_to_plot])
            labels = ["PC1", "PC2"]

        elif plotting_space == "phipsi":
            temp_dihedrals = molecule_type.get_many_dihedrals_from_cossin(cossin_data)

            (x,y) = ([item[1] for item in temp_dihedrals], [item[2] for item in temp_dihedrals])
            labels = ["phi", "psi"]

        # coloring
        if color_option == 'pure':
            coloring = 'red'
        elif color_option == 'step':
            coloring = range(len(x))
        elif color_option == 'phi':
            coloring = [item[1] for item in molecule_type.get_many_dihedrals_from_cossin(cossin_data)]
        elif color_option == 'psi':
            coloring = [item[2] for item in molecule_type.get_many_dihedrals_from_cossin(cossin_data)]
        elif color_option == 'other':
            assert (len(other_coloring) == len(x))
            coloring = other_coloring
            if smoothing_using_RNR:    # smooth coloring using RNR
                r_neigh = RadiusNeighborsRegressor(radius=smoothing_radius, weights='uniform')
                temp_coors = [list(item) for item in zip(x, y)]
                r_neigh.fit(temp_coors, coloring)
                coloring = r_neigh.predict(temp_coors)

        im = axis_object.scatter(x,y, c=coloring, cmap='gist_rainbow', picker=True)
        axis_object.set_xlabel(labels[0])
        axis_object.set_ylabel(labels[1])
        if contain_title:
            axis_object.set_title(title)

        if not axis_ranges is None:
            axis_object.set_xlim(axis_ranges[0])
            axis_object.set_ylim(axis_ranges[1])

        if contain_colorbar:
            fig_object.colorbar(im, ax=axis_object)

        # mouse clicking event
        import matplotlib
        # axis_object.text(-1.2, -1.2, 'save_frames', picker = True, fontsize=12)  # TODO: find better coordinates

        global temp_list_of_coor_index   # TODO: use better way instead of global variable
        temp_list_of_coor_index = []
        def onclick(event):
            global temp_list_of_coor_index
            if isinstance(event.artist, matplotlib.text.Text):
                if event.artist.get_text() == 'save_frames':
                    print temp_list_of_coor_index
                    if not self._related_coor_list_obj is None:
                        self._related_coor_list_obj.write_pdb_frames_into_file_with_list_of_coor_index(temp_list_of_coor_index,
                                                                                                       'temp_pdb/temp_frames.pdb')  # TODO: better naming
                    else:
                        raise Exception('related_coor_list_obj not defined!')
                    temp_list_of_coor_index = []  # output pdb file and clean up
                    print ('done saving frames!')
            elif isinstance(event.artist, matplotlib.collections.PathCollection):
                ind_list = list(event.ind)  # what is the index of this?
                print ('onclick:')
                temp_list_of_coor_index += ind_list

                for item in ind_list:
                    print(item, x[item], y[item])

        fig_object.canvas.mpl_connect('pick_event', onclick)

        return fig_object, axis_object, im

    def density_plotting(self,fig_object, axis_object,
                         network=None,
                         cossin_data_for_plotting=None,
                         n_levels=40
                         ):

        if network is None: network = self._network

        if cossin_data_for_plotting is None:
            cossin_data = self._network._data_set
        else:
            cossin_data = cossin_data_for_plotting

        x = [item[0] for item in network.get_PCs(cossin_data)]
        y = [item[1] for item in network.get_PCs(cossin_data)]

        df = pd.DataFrame({'x': x, 'y': y})
        sns.kdeplot(df.x, df.y, ax=axis_object, n_levels=n_levels)

        return fig_object, axis_object

    def plotting_potential_centers(self, fig_object, axis_object,
                                   list_of_coor_data_files, marker='x'):
        potential_centers = [single_biased_simulation_data(None, item)._potential_center for item in list_of_coor_data_files]
        (x, y) = zip(*potential_centers)

        axis_object.scatter(x, y, marker=marker)
        return


class iteration(object):
    def __init__(self, index,
                 network=None # if you want to start with existing network, assign value to "network"
                 ):
        self._index = index
        self._network = network

    def train_network_and_save(self, training_interval=None, num_of_trainings = CONFIG_13):
        '''num_of_trainings is the number of trainings that we are going to run, and 
        then pick one that has the largest Fraction of Variance Explained (FVE),
        by doing this, we might avoid network with very poor quality
        '''
        if training_interval is None: training_interval = self._index  # to avoid too much time on training
        my_file_list = coordinates_data_files_list(list_of_dir_of_coor_data_files=['../target/' + CONFIG_30]).get_list_of_coor_data_files()
        data_set = molecule_type.get_many_cossin_from_coordiantes_in_list_of_files(my_file_list)

        max_FVE = 0
        current_network = None
        
        # start of multiprocessing
        # from multiprocessing import Process
        # temp_training = lambda x: neural_network_for_simulation(index=self._index,
        #                                                  data_set_for_training= data_set,
        #                                                  training_data_interval=training_interval,
        #                                                 )
 
        # temp_list_of_trained_autoencoders = map(temp_training, range(num_of_trainings))
        # print (temp_list_of_trained_autoencoders)
        
        # task_list = range(num_of_trainings)
        # for item in range(num_of_trainings):
        #     task_list[item] = Process(target = temp_list_of_trained_autoencoders[item].train)
        #     task_list[item].start()
         
        # map(lambda x: x.join(), task_list)
            

        # print ('temp_FVE_list =')
        # print (temp_list_of_trained_autoencoders[0].get_fraction_of_variance_explained())
        # print (map(lambda x: x.get_fraction_of_variance_explained(), temp_list_of_trained_autoencoders))
        # current_network = max(temp_list_of_trained_autoencoders, get_fraction_of_variance_explained)  # find the network with largest FVE
        # print("max_FVE = %f" % current_network.get_fraction_of_variance_explained())
        # end of multiprocessing


        for item in range(num_of_trainings):
            temp_network = neural_network_for_simulation(index=self._index,
                                                         data_set_for_training= data_set,
                                                         training_data_interval=training_interval,
                                                        )

            temp_network.train()
            print("temp FVE = %f" % (temp_network.get_fraction_of_variance_explained()))
            if temp_network.get_fraction_of_variance_explained() > max_FVE:
                max_FVE = temp_network.get_fraction_of_variance_explained()
                print("max_FVE = %f" % max_FVE)
                assert(max_FVE > 0)
                current_network = copy.deepcopy(temp_network)

        current_network.save_into_file()
        self._network = current_network
        return

    def prepare_simulation(self, machine_to_run_simulations = CONFIG_24):
        if CONFIG_28 == "CustomManyParticleForce":
            self._network.write_expression_into_file()
        elif CONFIG_28 == "ANN_Force":
            self._network.write_coefficients_of_connections_into_file()
        else:
            raise Exception("force type not defined!")
            
        commands = self._network.get_commands_for_further_biased_simulations()
        # print ('in iteration.prepare_simulation: commands = ')
        # print (commands)
        if machine_to_run_simulations == "cluster":
            cluster_management.create_sge_files_for_commands(list_of_commands_to_run=commands)
        elif machine_to_run_simulations == 'local':
            pass
            # TODO
        return

    def run_simulation(self, machine_to_run_simulations = CONFIG_24):
        if machine_to_run_simulations == 'cluster':
            cluster_management.monitor_status_and_submit_periodically(num = CONFIG_14,
                                        num_of_running_jobs_when_allowed_to_stop = CONFIG_15)
        elif machine_to_run_simulations == 'local':
            commands = self._network.get_commands_for_further_biased_simulations()
            procs_to_run_commands = range(len(commands))
            for index, item in enumerate(commands):
                print ("running: \t" + item)
                procs_to_run_commands[index] = subprocess.Popen(item.split())

            exit_codes = [p.wait() for p in procs_to_run_commands]
            assert (sum(exit_codes) < CONFIG_31)  # we could not have more than CONFIG_31 simulations failed in each iteration

            # TODO: currently they are not run in parallel, fix this later
        
        # TODO: run next line only when the jobs are done, check this
        if CONFIG_29:
            molecule_type.remove_water_mol_and_Cl_from_pdb_file(preserve_original_file = False)
        molecule_type.generate_coordinates_from_pdb_files()
        return


class simulation_with_ANN_main(object):
    def __init__(self, num_of_iterations = 1,
                 initial_iteration=None,  # this is where we start with
                 training_interval = None,
                 ):
        self._num_of_iterations = num_of_iterations
        self._initial_iteration = initial_iteration
        self._training_interval = training_interval
        return

    def run_one_iteration(self, one_iteration):
        if one_iteration is None:
            one_iteration = iteration(1, network=None)
        if one_iteration._network is None:
            one_iteration.train_network_and_save(training_interval = self._training_interval)   # train it if it is empty

        one_iteration.prepare_simulation()
        print('running this iteration #index = %d' % one_iteration._index)
        one_iteration.run_simulation()
        return

    def run_mult_iterations(self, num=None):
        if num is None: num = self._num_of_iterations

        current_iter = self._initial_iteration
        for item in range(num):
            self.run_one_iteration(current_iter)
            next_index = current_iter._index + 1
            current_iter = iteration(next_index, None)

        return

class single_biased_simulation_data(object):
    '''TODO: This class is not completed'''
    def __init__(self, my_network, file_for_single_biased_simulation_coor):
        '''my_network is the corresponding network for this biased simulation'''
        self._file_for_single_biased_simulation_coor = file_for_single_biased_simulation_coor
        self._my_network = my_network
        temp_potential_center_string = file_for_single_biased_simulation_coor.split('_pc_[')[1].split(']')[0]
        self._potential_center = [float(item) for item in temp_potential_center_string.split(',')]
        self._force_constant = float(file_for_single_biased_simulation_coor.split('biased_output_fc_')[1].split('_pc_')[0])
        self._number_of_data = float(subprocess.check_output(['wc', '-l', file_for_single_biased_simulation_coor]).split()[0])

        if not self._my_network is None:
            if self._my_network._hidden_layers_type[1] == CircularLayer:
                self._dimension_of_PCs = self._my_network._node_num[2] / 2
            else:
                self._dimension_of_PCs = self._my_network._node_num[2]

        return

    def get_center_of_data_cloud_in_this_biased_simulation(self):
        cossin = molecule_type.get_many_cossin_from_coordiantes_in_list_of_files([self._file_for_single_biased_simulation_coor])
        PCs = self._my_network.get_PCs(cossin)
        assert(len(PCs[0]) == self._dimension_of_PCs)
        assert(len(PCs) == self._number_of_data)
        PCs_transpose = zip(*PCs)
        center_of_data_cloud = map(lambda x: sum(x) / len(x), PCs_transpose)
        return center_of_data_cloud

    def get_offset_between_potential_center_and_data_cloud_center(self):
        '''see if the push in this biased simulation actually works, large offset means it
        does not work well
        '''
        PCs_average = self.get_center_of_data_cloud_in_this_biased_simulation()
        offset = [PCs_average[item] - self._potential_center[item] for item in range(self._dimension_of_PCs)]
        return offset

