import copy, pickle, re, os, time, subprocess, datetime, itertools
from scipy import io as sciio
import numpy as np
from math import *
from pybrain.structure import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
import matplotlib.pyplot as plt


class coordinates_data_files(object):
    '''this class holds the information of data files'''
    def __init__(self,
                list_of_dir_of_coor_data_files = ['../target/'], # this is the directory that holds corrdinates data files
                ):
        self._list_of_dir_of_coor_data_files = list_of_dir_of_coor_data_files
        self._list_of_coor_data_files = []
        for item in self._list_of_dir_of_coor_data_files:
            self._list_of_coor_data_files += subprocess.check_output(['find', item,'-name' ,'*coordinates.txt']).split('\n')[:-1]    
        
        self._list_of_coor_data_files = list(set(self._list_of_coor_data_files))  # remove duplicates
        return

class neural_network_for_simulation(object):
    """the neural network for simulation"""
    
    def __init__(self, 
                 index, # the index of the current network
                 list_of_coor_data_files, # accept multiple files of training coordinate data
                 energy_expression_file = None, 
                 training_data_interval = 1, 
                 preprocessing_settings = None, 
                 whether_preprecess_data = False,  # by default, turn off preprocessing
                 connection_between_layers = None, connection_with_bias_layers = None,
                 node_num = None, # the structure of ANN
                 network_parameters = None, # includes [learningrate,momentum, weightdecay, lrdecay]
                 max_num_of_training = 100, 
                 PCs = None,  # principal components
                 ):
        
        self._index = index
        self._list_of_coor_data_files = list_of_coor_data_files
        self._training_data_interval = training_data_interval
        if energy_expression_file is None:
            self._energy_expression_file = "../resources/energy_expression_%d.txt" %(index)  
        else:
            self._energy_expression_file = energy_expression_file
            
        self._data_set = self.get_many_cossin_from_coordiantes_in_file(list_of_coor_data_files)    
        self._preprocessing_settings = preprocessing_settings
        self._whether_preprecess_data = whether_preprecess_data
        if not self._whether_preprecess_data:
            self._preprocessing_settings = [(1, 0) for item in range(8)]

        self._connection_between_layers = connection_between_layers
        self._connection_with_bias_layers = connection_with_bias_layers
        self._node_num = [8, 12, 2, 12, 8] if node_num is None else node_num
        self._network_parameters = [0.002, 0.4, 0.1, 1] if network_parameters is None else network_parameters
        self._max_num_of_training = max_num_of_training
        self._PCs = PCs
        return
        

    def save_into_file(self, filename = None):
        if filename is None:
            filename = "../resources/network_%s.pkl" % str(self._index) # by default naming with its index
            
        if os.path.isfile(filename):  # backup file if previous one exists
            os.rename(filename, filename.split('.pkl')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.pkl')

        with open(filename, 'wb') as my_file:
            pickle.dump(self, my_file, pickle.HIGHEST_PROTOCOL)
            
        return 
    
    
    def get_cossin_from_a_coordinate(self, a_coordinate):
        num_of_coordinates = len(a_coordinate) / 3
        a_coordinate = np.array(a_coordinate).reshape(num_of_coordinates, 3)
        diff_coordinates = a_coordinate[1:num_of_coordinates, :] - a_coordinate[0:num_of_coordinates - 1,:]  # bond vectors
        diff_coordinates_1=diff_coordinates[0:num_of_coordinates-2,:];diff_coordinates_2=diff_coordinates[1:num_of_coordinates-1,:]
        normal_vectors = np.cross(diff_coordinates_1, diff_coordinates_2);
        normal_vectors_normalized = np.array(map(lambda x: x / sqrt(np.dot(x,x)), normal_vectors))
        normal_vectors_normalized_1 = normal_vectors_normalized[0:num_of_coordinates-3, :];normal_vectors_normalized_2 = normal_vectors_normalized[1:num_of_coordinates-2,:];
        diff_coordinates_mid = diff_coordinates[1:num_of_coordinates-2]; # these are bond vectors in the middle (remove the first and last one), they should be perpendicular to adjacent normal vectors
    
        cos_of_angles = range(len(normal_vectors_normalized_1))
        sin_of_angles_vec = range(len(normal_vectors_normalized_1))
        sin_of_angles = range(len(normal_vectors_normalized_1)) # initialization
    
        for index in range(len(normal_vectors_normalized_1)):
            cos_of_angles[index] = np.dot(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
            sin_of_angles_vec[index] = np.cross(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
            sin_of_angles[index] = sqrt(np.dot(sin_of_angles_vec[index], sin_of_angles_vec[index])) * np.sign(sum(sin_of_angles_vec[index]) * sum(diff_coordinates_mid[index]));  
        return cos_of_angles + sin_of_angles
    
    def get_many_cossin_from_coordinates(self, coordinates):
        return map(self.get_cossin_from_a_coordinate, coordinates)

    def get_many_cossin_from_coordiantes_in_file (self, list_of_files):
        result = []
        for item in list_of_files:
            coordinates = np.loadtxt(item)
            temp = self.get_many_cossin_from_coordinates(coordinates)
            result += temp
        
        return result
    
    def get_many_dihedrals_from_coordinates_in_file (self, list_of_files):
        # why we need to get dihedreals from a list of coordinate files?
        # because we will probably need to plot other files outside self._list_of_coor_data_files
        temp = self.get_many_cossin_from_coordiantes_in_file(list_of_files)
        result = []
        for item in temp:
            temp_angle = np.multiply(np.arccos(item[0:4]), np.sign(item[4:8]))
            result += [list(temp_angle)]
        
        return result
    
    def mapminmax(self, my_list): # for preprocessing in network
        my_min = min(my_list)
        my_max = max(my_list)
        mul_factor = 2.0 / (my_max - my_min)
        offset = (my_min + my_max) / 2.0
        result_list = map(lambda x : (x - offset) * mul_factor, my_list)
        return (result_list, (mul_factor, offset)) # also return the parameters for processing

    def get_mapminmax_preprocess_result_and_coeff(self, data=None):
        if data is None:
            data = self._data_set
        
        data = np.transpose(np.array(data))
        result = []; params = []
        for item in data:
            temp_result, preprocess_params = self.mapminmax(item)
            result.append(temp_result)
            params.append(preprocess_params)
        return (np.transpose(np.array(result)), params)
    
    def mapminmax_preprocess_using_coeff(self, input_data=None, preprocessing_settings=None):
        if preprocessing_settings is None:
            preprocessing_settings = self._preprocessing_settings
            
        temp_setttings = np.transpose(np.array(preprocessing_settings))
        result = []
        
        for item in input_data:
            item = np.multiply(item - temp_setttings[1], temp_setttings[0])
            result.append(item)
        
        return result
    
    def get_expression_of_network(self, connection_between_layers=None, connection_with_bias_layers=None):
        if connection_between_layers is None:
            connection_between_layers = self._connection_between_layers
        if connection_with_bias_layers is None:
            connection_with_bias_layers = self._connection_with_bias_layers
            
        node_num = self._node_num
        expression = ""
        # first part: network
        for i in range(2):
            expression = '\n' + expression
            mul_coef = connection_between_layers[i].params.reshape(node_num[i + 1], node_num[i])
            bias_coef = connection_with_bias_layers[i].params
            for j in range(np.size(mul_coef, 0)):
                temp_expression = 'layer_%d_unit_%d = tanh( ' % (i + 1, j) 
                
                for k in range(np.size(mul_coef, 1)):
                    temp_expression += ' %f * layer_%d_unit_%d +' % (mul_coef[j, k], i, k)
                    
                temp_expression += ' %f);\n' % (bias_coef[j])
                expression = temp_expression + expression  # order of expressions matter in OpenMM
                
        # second part: definition of inputs
        index_of_backbone_atoms = [2, 5, 7, 9, 15, 17, 19];
        for i in range(len(index_of_backbone_atoms) - 3):
            index_of_coss = i
            index_of_sins = i + 4
            expression += 'layer_0_unit_%d = (raw_layer_0_unit_%d - %f) * %f;\n' % (index_of_coss, index_of_coss, self._preprocessing_settings[index_of_coss][1], self._preprocessing_settings[index_of_coss][0])
            expression += 'layer_0_unit_%d = (raw_layer_0_unit_%d - %f) * %f;\n' % (index_of_sins, index_of_sins, self._preprocessing_settings[index_of_sins][1], self._preprocessing_settings[index_of_sins][0])
            expression += 'raw_layer_0_unit_%d = cos(dihedral_angle_%d);\n' % (index_of_coss, i)
            expression += 'raw_layer_0_unit_%d = sin(dihedral_angle_%d);\n' % (index_of_sins, i)
            expression += 'dihedral_angle_%d = dihedral(p%d, p%d, p%d, p%d);\n' % (i, index_of_backbone_atoms[i], index_of_backbone_atoms[i+1],index_of_backbone_atoms[i+2],index_of_backbone_atoms[i+3])
        
        return expression
    
    def write_expression_into_file(self, out_file = None):
        if out_file is None: out_file = self._energy_expression_file
        
        expression = self.get_expression_of_network()
        with open(out_file, 'w') as f_out:
            f_out.write(expression)
        return
    
    def get_mid_result(self, input_data=None, connection_between_layers=None, connection_with_bias_layers=None):
        if input_data is None: input_data = self._data_set
        if connection_between_layers is None: connection_between_layers = self._connection_between_layers
        if connection_with_bias_layers is None: connection_with_bias_layers = self._connection_with_bias_layers
        
        node_num = self._node_num
        temp_mid_result = range(4)
        mid_result = []
        
        # first may need to do preprocessing
        if self._whether_preprecess_data:
            data_as_input_to_network = self.mapminmax_preprocess_using_coeff(input_data, self._preprocessing_settings)
        else:
            data_as_input_to_network = input_data

        for item in data_as_input_to_network: 
            for i in range(4):
                mul_coef = connection_between_layers[i].params.reshape(node_num[i + 1], node_num[i]) # fix node_num
                bias_coef = connection_with_bias_layers[i].params
                previous_result = item if i == 0 else temp_mid_result[i - 1]
                temp_mid_result[i] = np.dot(mul_coef, previous_result) + bias_coef
                if i != 3: # the last output layer is a linear layer, while others are tanh layers
                    temp_mid_result[i] = map(tanh, temp_mid_result[i])
                
            mid_result.append(copy.deepcopy(temp_mid_result)) # note that should use deepcopy
        return mid_result
    
    def get_PC_and_save_it_to_network(self): 
        '''get PCs and save the result into _PCs
        '''
        mid_result = self.get_mid_result()
        self._PCs = [item[1] for item in mid_result]
        return
    
    def train(self):
        
        ####################### set up autoencoder begin #######################
        node_num = self._node_num
        
        in_layer = LinearLayer(node_num[0], "IL")
        hidden_layers = [TanhLayer(node_num[1], "HL1"), TanhLayer(node_num[2], "HL2"), TanhLayer(node_num[3], "HL3")]
        bias_layers = [BiasUnit("B1"),BiasUnit("B2"),BiasUnit("B3"),BiasUnit("B4")]
        out_layer = LinearLayer(node_num[4], "OL")
        
        layer_list = [in_layer] + hidden_layers + [out_layer]
        
        molecule_net = FeedForwardNetwork()
        
        molecule_net.addInputModule(in_layer)
        for item in (hidden_layers + bias_layers):
            molecule_net.addModule(item)
        
        molecule_net.addOutputModule(out_layer)
        
        connection_between_layers = range(4); connection_with_bias_layers = range(4)
        
        for i in range(4):
            connection_between_layers[i] = FullConnection(layer_list[i], layer_list[i+1])
            connection_with_bias_layers[i] = FullConnection(bias_layers[i], layer_list[i+1])
            molecule_net.addConnection(connection_between_layers[i])  # connect two neighbor layers
            molecule_net.addConnection(connection_with_bias_layers[i])  
            
        molecule_net.sortModules()  # this is some internal initialization process to make this module usable
        
        ####################### set up autoencoder end #######################
        
        trainer = BackpropTrainer(molecule_net, learningrate=self._network_parameters[0],
                                                momentum=self._network_parameters[1], 
                                                weightdecay=self._network_parameters[2], 
                                                lrdecay=self._network_parameters[3], 
                                                verbose=False)
        data_set = SupervisedDataSet(node_num[0], node_num[4])
        
        sincos = self._data_set[::self._training_data_interval]  # pick some of the data to train
        if self._whether_preprecess_data:
            (sincos_after_process, self._preprocessing_settings) = self.get_mapminmax_preprocess_result_and_coeff(data = sincos)
            data_as_input_to_network = sincos_after_process
        else: # otherwise no preprocessing
            data_as_input_to_network = sincos

        for item in data_as_input_to_network:  
            data_set.addSample(item, item)
        
        print('start training network with index = %d\n' % self._index)
        trainer.trainUntilConvergence(data_set, maxEpochs=self._max_num_of_training)

        self._connection_between_layers = connection_between_layers
        self._connection_with_bias_layers = connection_with_bias_layers 
        
        print('Done training network with index = %d\n' % self._index)
        return 

    def get_training_error(self):
        # it turns out that this error info cannot be a good measure of the quality of the autoencoder
        if self._whether_preprecess_data == False:
            input_data = np.array(self._data_set)
        else:
            print("having done this yet!\n")
            pass

        output_data = np.array([item[3] for item in self.get_mid_result()])
        return np.linalg.norm(input_data - output_data) / sqrt(8 * len(input_data))

    def get_fraction_of_variance_explained(self):
        if self._whether_preprecess_data == False:
            input_data = np.array(self._data_set)
        else:
            print("having done this yet!\n")
            pass

        output_data = np.array([item[3] for item in self.get_mid_result()])
        var_of_input = sum(np.var(input_data, axis=0))
        var_of_output = sum(np.var(output_data, axis=0))
        return var_of_output / var_of_input

    def generate_mat_file_for_WHAM_reweighting(self, list_of_coor_data_files):
        force_constants = []
        harmonic_centers = []
        window_counts = []
        coords = []
        umbOP = []
        for item in list_of_coor_data_files:
            print('processing %s' %item)
            temp_force_constant = float(item.split('biased_output_fc_')[1].split('_x1_')[0])
            force_constants += [[temp_force_constant, temp_force_constant]]
            harmonic_centers += [[float(item.split('_x1_')[1].split('_x2_')[0]), float(item.split('_x2_')[1].split('_coordinates.txt')[0])]]
            temp_window_count = float(subprocess.check_output(['wc', '-l', item]).split()[0])  # there would be some problems if using int
            window_counts += [temp_window_count]
            temp_mid_result = self.get_mid_result(self.get_many_cossin_from_coordiantes_in_file([item]))
            temp_coor = [a[1] for a in temp_mid_result]
            assert(temp_window_count == len(temp_coor))  # ensure the number of coordinates is window_count
            coords += temp_coor
            temp_angles = self.get_many_dihedrals_from_coordinates_in_file([item])
            temp_umbOP = [a[1:3] for a in temp_angles]
            assert(temp_window_count == len(temp_umbOP))
            assert(2 == len(temp_umbOP[0]))
            umbOP += temp_umbOP

        window_counts = np.array(window_counts)
        sciio.savemat('WHAM_nD__preprocessor.mat', {'window_counts': window_counts, 
            'force_constants': force_constants, 'harmonic_centers': harmonic_centers,
            'coords': coords, 'dim': 2,
            })
        sciio.savemat('umbrella_OP.mat',
            {'umbOP': umbOP
            })
        return

class plotting(object):
    '''
    this class implements different plottings
    '''
    
    def __init__(self, network):
        self._network = network
        pass
        
    def plotting_with_coloring_option(self, plotting_space, # means "PC" space or "phi-psi" space
                                            network=None, 
                                            list_of_coordinate_files_for_plotting=None, # accept multiple files
                                            color_option='pure',
                                            other_coloring=None,
                                            title=None,
                                            axis_ranges=None
                                            ):
        '''
        by default, we are using training data, and we also allow external data input
        '''
        if network is None: network = self._network
        if list_of_coordinate_files_for_plotting is None: 
            list_of_coordinate_files_for_plotting = network._list_of_coor_data_files
            
        if plotting_space == "PC":
            temp_sincos = network.get_many_cossin_from_coordiantes_in_file(list_of_coordinate_files_for_plotting)

            temp_mid_result = network.get_mid_result(input_data = temp_sincos)
            PCs_to_plot = [item[1] for item in temp_mid_result]

            (x, y) = ([item[0] for item in PCs_to_plot], [item[1] for item in PCs_to_plot])
            labels = ["PC1", "PC2"]
            
        elif plotting_space == "phipsi":
            temp_dihedrals = network.get_many_dihedrals_from_coordinates_in_file(list_of_coordinate_files_for_plotting)
            
            (x,y) = ([item[1] for item in temp_dihedrals], [item[2] for item in temp_dihedrals])
            labels = ["phi", "psi"]
        
        # coloring
        if color_option == 'pure':
            coloring = 'red'
        elif color_option == 'step':
            coloring = range(len(x))
        elif color_option == 'other':
            coloring = other_coloring
            
        fig, ax = plt.subplots()
        ax.scatter(x,y, c=coloring)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        
        if not axis_ranges is None:
            ax.set_xlim(axis_ranges[0])
            ax.set_ylim(axis_ranges[1])
            
        return fig

class simulation_management(object):
    def __init__(self, mynetwork):
        self._mynetwork = mynetwork
        return
    
    def create_sge_files_for_simulation(self,list_of_potential_center = None, 
                                        num_of_simulation_steps = 2000,
                                        energy_expression_file=None):
        if list_of_potential_center is None: 
            list_of_potential_center = self.get_boundary_points()
        if energy_expression_file is None:
            energy_expression_file = self._mynetwork._energy_expression_file
            filename_of_exergy_expression = energy_expression_file.split('resources/')[1]
            
        for potential_center in list_of_potential_center:
    
            parameter_list = ("50", str(num_of_simulation_steps), "1000", str(potential_center[0]), str(potential_center[1]), 
                            'network_' + str(self._mynetwork._index),
                            filename_of_exergy_expression)
        
            file_name = "../sge_files/job_biased_%s_%s_%s_%s_%s_%s_%s.sge" % parameter_list
            command = "python ../src/biased_simulation.py %s %s %s %s %s %s %s" % parameter_list
    
            print("creating %s" % file_name)
    
            content_for_sge_files = '''#!/bin/bash

#$ -S /bin/bash           # use bash shell
#$ -V                     # inherit the submission environment 
#$ -cwd                   # start job in submission directory

#$ -m ae                 # email on abort, begin, and end
#$ -M wei.herbert.chen@gmail.com         # email address

#$ -q all.q               # queue name
#$ -l h_rt=1:00:00       # run time (hh:mm:ss)
####$ -l hostname=compute-0-3

%s

echo "This job is DONE!"

exit 0
''' % command

            with open(file_name, 'w') as f_out:
                f_out.write(content_for_sge_files);
                f_out.write("\n")

        return
    
    
    def get_boundary_points(self, list_of_points = None, num_of_bins = 5):
        if list_of_points is None: list_of_points = self._mynetwork._PCs
            
        x = [item[0] for item in list_of_points]
        y = [item[1] for item in list_of_points]
        
        temp = np.histogram2d(x,y, bins=[num_of_bins, num_of_bins])
        hist_matrix = temp[0]
        # add a set of zeros around this region
        hist_matrix = np.insert(hist_matrix, num_of_bins, np.zeros(num_of_bins), 0)
        hist_matrix = np.insert(hist_matrix, 0, np.zeros(num_of_bins), 0)
        hist_matrix = np.insert(hist_matrix, num_of_bins, np.zeros(num_of_bins + 2), 1)
        hist_matrix = np.insert(hist_matrix, 0, np.zeros(num_of_bins +2), 1)
        
        sum_of_neighbors = np.zeros(np.shape(hist_matrix)) # number of neighbors occupied with some points
        for i in range(np.shape(hist_matrix)[0]):
            for j in range(np.shape(hist_matrix)[1]):
                if i != 0: sum_of_neighbors[i,j] += hist_matrix[i - 1][j]
                if j != 0: sum_of_neighbors[i,j] += hist_matrix[i][j - 1]
                if i != np.shape(hist_matrix)[0] - 1: sum_of_neighbors[i,j] += hist_matrix[i + 1][j]
                if j != np.shape(hist_matrix)[1] - 1: sum_of_neighbors[i,j] += hist_matrix[i][j + 1]
                    
        bin_width_0 = temp[1][1]-temp[1][0]
        bin_width_1 = temp[2][1]-temp[2][0]
        min_coor_in_PC_space_0 = temp[1][0] - 0.5 * bin_width_0  # multiply by 0.5 since we want the center of the grid
        min_coor_in_PC_space_1 = temp[2][0] - 0.5 * bin_width_1
        
        potential_centers = []
        
        for i in range(np.shape(hist_matrix)[0]):
            for j in range(np.shape(hist_matrix)[1]):
                if hist_matrix[i,j] == 0 and sum_of_neighbors[i,j] != 0:  # no points in this block but there are points in neighboring blocks
                    temp_potential_center = [round(min_coor_in_PC_space_0 + i * bin_width_0, 2), round(min_coor_in_PC_space_1 + j * bin_width_1, 2)]
                    potential_centers.append(temp_potential_center)
        
        return potential_centers

    def get_boundary_points_2(self, list_of_points = None, num_of_bins = 5, num_of_boundary_points = 10):
        '''This is another version of get_boundary_points() function'''
        if list_of_points is None: list_of_points = self._mynetwork._PCs
            
        x = [item[0] for item in list_of_points]
        y = [item[1] for item in list_of_points]
        
        temp = np.histogram2d(x,y, bins=[num_of_bins, num_of_bins])
        hist_matrix = temp[0]
        # add a set of zeros around this region
        hist_matrix = np.insert(hist_matrix, num_of_bins, np.zeros(num_of_bins), 0)
        hist_matrix = np.insert(hist_matrix, 0, np.zeros(num_of_bins), 0)
        hist_matrix = np.insert(hist_matrix, num_of_bins, np.zeros(num_of_bins + 2), 1)
        hist_matrix = np.insert(hist_matrix, 0, np.zeros(num_of_bins +2), 1)
        
        sum_of_neighbors = np.zeros(np.shape(hist_matrix)) # number of neighbors occupied with some points
        for i in range(np.shape(hist_matrix)[0]):
            for j in range(np.shape(hist_matrix)[1]):
                if i != 0: sum_of_neighbors[i,j] += hist_matrix[i - 1][j]
                if j != 0: sum_of_neighbors[i,j] += hist_matrix[i][j - 1]
                if i != np.shape(hist_matrix)[0] - 1: sum_of_neighbors[i,j] += hist_matrix[i + 1][j]
                if j != np.shape(hist_matrix)[1] - 1: sum_of_neighbors[i,j] += hist_matrix[i][j + 1]
                    
        bin_width_0 = temp[1][1]-temp[1][0]
        bin_width_1 = temp[2][1]-temp[2][0]
        min_coor_in_PC_space_0 = temp[1][0] - 0.5 * bin_width_0  # multiply by 0.5 since we want the center of the grid
        min_coor_in_PC_space_1 = temp[2][0] - 0.5 * bin_width_1
        
        potential_centers = []
        
        # now sort these grids (that has no points in it) 
        # based on total number of points in its neighbors
        index_of_grids = list(itertools.product(range(np.shape(hist_matrix)[0]), range(np.shape(hist_matrix)[1])))
        print(index_of_grids)
        sorted_index_of_grids = sorted(index_of_grids, key = lambda x: sum_of_neighbors[x[0]][x[1]]) # sort based on histogram, return index values
        temp_count = 0
        for index in sorted_index_of_grids:
            if hist_matrix[index] == 0 and sum_of_neighbors[index] != 0:
                if temp_count >= num_of_boundary_points:
                    break
                else:
                    temp_count += 1
                    temp_potential_center = [round(min_coor_in_PC_space_0 + index[0] * bin_width_0, 2), \
                                             round(min_coor_in_PC_space_1 + index[1] * bin_width_1, 2)]
                    potential_centers.append(temp_potential_center)
            
        return potential_centers
    
    def get_num_of_running_jobs(self):
        output = subprocess.check_output(['qstat'])
        num_of_running_jobs = len(re.findall('weichen9', output))
        print('checking number of running jobs = %d\n' % num_of_running_jobs)
        return num_of_running_jobs
    
    def submit_sge_jobs_and_archive_files(self, job_file_lists, 
                                          num, # num is the max number of jobs submitted each time
                                          flag_of_whether_to_record_qsub_commands = False
                                        ):  
        dir_to_archive_files = '../sge_files/archive/'

        if not os.path.exists(dir_to_archive_files):
            os.makedirs(dir_to_archive_files)

        assert(os.path.exists(dir_to_archive_files))
        
        for item in job_file_lists[0:num]:
            subprocess.check_output(['qsub', item])
            print('submitting ' + str(item))
            subprocess.check_output(['mv', item, dir_to_archive_files]) # archive files
        return 
    
    def get_sge_files_list(self):
        result = filter(lambda x: x[-3:] == "sge",subprocess.check_output(['ls', '../sge_files']).split('\n'))
        result = map(lambda x: '../sge_files/' + x, result)
        return result
    
    def del_all_jobs(self):
        # TODO
        output = subprocess.check_output(['qstat'])
        return
    
    def submit_new_jobs_if_there_are_too_few_jobs(self, num):
        if self.get_num_of_running_jobs() < num:
            job_list = self.get_sge_files_list()
            self.submit_sge_jobs_and_archive_files(job_list, num)
        return
    
    def monitor_status_and_submit_periodically(self, num, 
                                            num_of_running_jobs_when_allowed_to_stop = 0,
                                            monitor_mode = 'normal', # monitor_mode determines whether it can go out of first while loop
                                            ):
        if monitor_mode == 'normal':
            min_num_of_unsubmitted_jobs = 0
        elif monitor_mode == 'always_wait_for_submit':
            min_num_of_unsubmitted_jobs = -1

        num_of_unsubmitted_jobs = len(self.get_sge_files_list())
        # first check if there are unsubmitted jobs
        while num_of_unsubmitted_jobs != min_num_of_unsubmitted_jobs:
            time.sleep(10)
            try:
                self.submit_new_jobs_if_there_are_too_few_jobs(num)
                num_of_unsubmitted_jobs = len(self.get_sge_files_list())
            except:
                print("not able to submit jobs!\n")
        
        # then check if all jobs are done
        while self.get_num_of_running_jobs() > num_of_running_jobs_when_allowed_to_stop:  
            time.sleep(10)
        return
    
    def generate_coordinates_from_pdb_files(self, folder_for_pdb = '../target'):
        filenames = subprocess.check_output(['find', folder_for_pdb, '-name' ,'*.pdb']).split('\n')[:-1]
        # print (filenames)
        
        index_of_backbone_atoms = ['2', '5', '7', '9', '15', '17', '19']
        
        for input_file in filenames:
            print ('generating coordinates of ' + input_file)
            output_file = input_file[:-4] + '_coordinates.txt'
        
            with open(input_file) as f_in:
                with open(output_file, 'w') as f_out:
                    for line in f_in:
                        fields = line.strip().split()
                        if (fields[0] == 'ATOM' and fields[1] in index_of_backbone_atoms):
                            f_out.write(reduce(lambda x,y: x + '\t' + y, fields[6:9]))
                            f_out.write('\t')
                        elif fields[0] == "MODEL" and fields[1] != "1":
                            f_out.write('\n')
        
                    f_out.write('\n')  # last line
        print("Done generating coordinates files\n")
        return


class iteration(object):
    def __init__(self, index, 
                 network=None, # if you want to start with existing network, assign value to "network"
                 num_of_simulation_steps = 3000
                 ):  
        self._index = index
        self._network = network
        self._num_of_simulation_steps = num_of_simulation_steps
        
    def train_network_and_save(self, training_interval=None, num_of_trainings = 5):
        '''num_of_trainings is the number of trainings that we are going to run, and 
        then pick one that has the largest Fraction of Variance Explained (FVE),
        by doing this, we might avoid network with very poor quality
        '''
        if training_interval is None: training_interval = self._index  # to avoid too much time on training
        my_file_list = coordinates_data_files(list_of_dir_of_coor_data_files=['../target'])._list_of_coor_data_files

        max_FVE = 0 #
        current_network = None
        for item in range(num_of_trainings):
            temp_network = neural_network_for_simulation(index=self._index,
                                                            list_of_coor_data_files = my_file_list,
                                                            training_data_interval=training_interval, 
                                                            )
            
            temp_network.train() 
            temp_network.get_PC_and_save_it_to_network()
            print("temp FVE = %f" % (temp_network.get_fraction_of_variance_explained()))
            if temp_network.get_fraction_of_variance_explained() > max_FVE:
                max_FVE = temp_network.get_fraction_of_variance_explained()
                print("max_FVE = %f" % max_FVE) 
                assert(max_FVE > 0)
                current_network = copy.deepcopy(temp_network)
        
        current_network.save_into_file()
        self._network = current_network 
        return
        
    def prepare_simulation(self):
        
        self._network.write_expression_into_file()
        
        manager = simulation_management(self._network)
        manager.create_sge_files_for_simulation(num_of_simulation_steps = self._num_of_simulation_steps)
        return
    
    def run_simulation(self):
        manager = simulation_management(self._network)
        manager.monitor_status_and_submit_periodically(num = 7, 
                                        num_of_running_jobs_when_allowed_to_stop = 1) 
        manager.generate_coordinates_from_pdb_files()
        return
    
    def get_plottings(self):
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
        one_iteration.run_simulation()
        print('running this iteration #index = %d' % one_iteration._index)
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
    def __init__(self, my_network, file_for_single_biased_simulation_coor):
        '''my_network is the corresponding network for this biased simulation'''
        self._file_for_single_biased_simulation_coor = file_for_single_biased_simulation_coor
        self._my_network = my_network
        self._potential_center = [float(file_for_single_biased_simulation_coor.split('_x1_')[1].split('_x2_')[0]), \
                                  float(file_for_single_biased_simulation_coor.split('_x2_')[1].split('_coordinates.txt')[0])]
        self._force_constant = float(file_for_single_biased_simulation_coor.split('biased_output_fc_')[1].split('_x1_')[0])
        self._number_of_data = float(subprocess.check_output(['wc', '-l', file_for_single_biased_simulation_coor]).split()[0])  
        return

    def get_offset_between_potential_center_and_data_cloud_center(self):
        '''see if the push in this biased simulation actually works, large offset means it
        does not work well
        '''
        cossin = self._my_network.get_many_cossin_from_coordiantes_in_file([self._file_for_single_biased_simulation_coor])
        temp_mid_result = self._my_network.get_mid_result(input_data = cossin)
        PCs = [item[1] for item in temp_mid_result]
        assert(len(PCs[0]) == 2)
        assert(len(PCs) == self._number_of_data)
        PCs_transpose = zip(*PCs)
        PCs_average = map(lambda x: sum(x) / len(x), PCs_transpose)
        offset = [PCs_average[0] - self._potential_center[0], PCs_average[1] - self._potential_center[1]]
        return offset

    