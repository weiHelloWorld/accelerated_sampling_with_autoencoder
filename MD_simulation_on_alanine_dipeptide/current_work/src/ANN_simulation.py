from config import * # configuration file
from cluster_management import *
from autoencoders import *

"""note that all configurations for a class should be in function __init__(), and take configuration parameters
from config.py
"""

class coordinates_data_files_list(object):
    def __init__(self,
                list_of_dir_of_coor_data_files = CONFIG_1, # this is the directory that holds corrdinates data files
                ):
        assert (isinstance(list_of_dir_of_coor_data_files, list))    # to avoid passing the string in the constructor
        self._list_of_dir_of_coor_data_files = list_of_dir_of_coor_data_files
        self._list_of_coor_data_files = []

        for item in self._list_of_dir_of_coor_data_files:
            self._list_of_coor_data_files += subprocess.check_output('find %s -name *coordinates.txt' % item, shell=True).strip().split('\n')

        self._list_of_coor_data_files = list(set(self._list_of_coor_data_files))  # remove duplicates
        self._list_of_coor_data_files = filter(lambda x: os.stat(x).st_size > 0, self._list_of_coor_data_files)   # remove empty files
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


class plotting(object):
    """this class implements different plottings
    """

    def __init__(self, network):
        assert isinstance(network, autoencoder)
        self._network = network
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
                                            smoothing_radius = 0.1,
                                            enable_mousing_clicking_event = False,
                                            related_coor_list_obj = None,
                                            saving_snapshot_mode = 'single_point'
                                      ):
        """
        by default, we are using training data, and we also allow external data input
        :param related_coor_list_obj,  this must be specified when enable_mousing_clicking_event == True
        """
        if enable_mousing_clicking_event and related_coor_list_obj is None:
            raise Exception('related_coor_list_obj not defined!')

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
            assert (isinstance(molecule_type, Alanine_dipeptide))
            temp_dihedrals = molecule_type.get_many_dihedrals_from_cossin(cossin_data)

            (x,y) = ([item[1] for item in temp_dihedrals], [item[2] for item in temp_dihedrals])
            labels = ["phi", "psi"]
        else:
            raise Exception('plotting_space not defined!')

        # coloring
        if color_option == 'pure':
            coloring = 'red'
        elif color_option == 'step':
            coloring = list(range(len(x)))
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
        else:
            raise Exception('color_option not defined!')

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
        if enable_mousing_clicking_event:
            folder_to_store_these_frames = 'temp_pdb'
            if not os.path.exists(folder_to_store_these_frames):
                subprocess.check_output(['mkdir', folder_to_store_these_frames])

            # should calculate step_interval
            total_num_of_lines_in_coor_files = sum(related_coor_list_obj.get_list_of_line_num_of_coor_data_file())
            step_interval = int(total_num_of_lines_in_coor_files / len(cossin_data))

            if saving_snapshot_mode == 'multiple_points':
                axis_object.text(-1.2, -1.2, 'save_frames', picker = True, fontsize=12)  # TODO: find better coordinates

                global temp_list_of_coor_index   # TODO: use better way instead of global variable
                temp_list_of_coor_index = []
                def onclick(event):
                    global temp_list_of_coor_index
                    if isinstance(event.artist, matplotlib.text.Text):
                        if event.artist.get_text() == 'save_frames':
                            print temp_list_of_coor_index
                            related_coor_list_obj.write_pdb_frames_into_file_with_list_of_coor_index(temp_list_of_coor_index,
                                                                            folder_to_store_these_frames + '/temp_frames.pdb')  # TODO: better naming

                            temp_list_of_coor_index = []  # output pdb file and clean up
                            print ('done saving frames!')
                    elif isinstance(event.artist, matplotlib.collections.PathCollection):
                        ind_list = list(event.ind)
                        print ('onclick:')
                        temp_list_of_coor_index += [item * step_interval for item in ind_list]  # should include step_interval

                        for item in ind_list:
                            print(item, x[item], y[item])
                    return

            elif saving_snapshot_mode == 'single_point':
                def onclick(event):
                    if isinstance(event.artist, matplotlib.collections.PathCollection):
                        ind_list = list(event.ind)
                        print ('onclick:')
                        for item in ind_list:
                            print(item, x[item], y[item])
                        temp_ind_list = [item * step_interval for item in ind_list]  # should include step_interval
                        average_x = sum([x[item] for item in ind_list]) / len(ind_list)
                        average_y = sum([y[item] for item in ind_list]) / len(ind_list)
                        out_file_name = folder_to_store_these_frames + '/temp_frames_[%f,%f].pdb' % (average_x, average_y)

                        related_coor_list_obj.write_pdb_frames_into_file_with_list_of_coor_index(temp_ind_list,
                            out_file_name=out_file_name)
                        # need to verify PCs generated from this output pdb file are consistent from those in the list selected
                        molecule_type.generate_coordinates_from_pdb_files(path_for_pdb=out_file_name)
                        cossin_data_selected = molecule_type.get_many_cossin_from_coordiantes_in_list_of_files(
                            list_of_files=[out_file_name.replace('.pdb', '_coordinates.txt')])
                        PCs_of_points_selected = network.get_PCs(input_data=cossin_data_selected)
                        assert (PCs_of_points_selected == [[x[item], y[item]] for item in ind_list])

                    return
            else:
                raise Exception('saving_snapshot_mode error')

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

    @staticmethod
    def plotting_potential_centers(fig_object, axis_object,
                                   list_of_coor_data_files, marker='x'):
        potential_centers = [single_biased_simulation_data(None, item)._potential_center for item in list_of_coor_data_files]
        [x, y] = list(zip(*potential_centers))

        axis_object.scatter(x, y, marker=marker)
        return fig_object, axis_object


class iteration(object):
    def __init__(self, index,
                 network=None # if you want to start with existing network, assign value to "network"
                 ):
        self._index = index
        self._network = network

    # def train_network_and_save(self, training_interval=None, num_of_trainings = CONFIG_13):
    #     """num_of_trainings is the number of trainings that we are going to run, and
    #     then pick one that has the largest Fraction of Variance Explained (FVE),
    #     by doing this, we might avoid network with very poor quality
    #     """
    #     if training_interval is None: training_interval = self._index  # to avoid too much time on training
    #     my_file_list = coordinates_data_files_list(list_of_dir_of_coor_data_files=['../target/' + CONFIG_30]).get_list_of_coor_data_files()
    #     data_set = molecule_type.get_many_cossin_from_coordiantes_in_list_of_files(my_file_list)
    #
    #     parallelize_training = CONFIG_43
    #     if parallelize_training:
    #         from multiprocessing import Process
    #         training_and_saving_task = lambda x: neural_network_for_simulation(index=self._index,
    #                                                                        data_set_for_training=data_set,
    #                                                                        training_data_interval=training_interval,
    #                                                                        ).train().save_into_file(x)
    #         task_list = list(range(num_of_trainings))
    #         temp_intermediate_result_file_list = ['/tmp/%d.pkl' % item for item in range(num_of_trainings)]
    #         for item in range(num_of_trainings):
    #             task_list[item] = Process(target = training_and_saving_task, args = (temp_intermediate_result_file_list[item], ))   # train and save intermediate result in /tmp folder
    #             task_list[item].start()
    #
    #         map(lambda x: x.join(), task_list)
    #         temp_networks = [Sutils.load_object_from_pkl_file(item) for item in temp_intermediate_result_file_list]
    #
    #     else:
    #         temp_networks = [neural_network_for_simulation(index=self._index,
    #                                                        data_set_for_training=data_set,
    #                                                        training_data_interval=training_interval,
    #                                                        )
    #                          for _ in range(num_of_trainings)]
    #         for item in temp_networks:
    #             item.train()
    #
    #     temp_FVE_list = [item.get_fraction_of_variance_explained() for item in temp_networks]
    #     max_FVE = max(temp_FVE_list)
    #     print("temp_FVE_list = %s, max_FVE = %s" % (str(temp_FVE_list), str(max_FVE)))
    #
    #     select_network_manually = False
    #     if select_network_manually:
    #         network_index = int(raw_input('select a network:'))
    #         current_network = temp_networks[network_index]
    #     else:
    #         current_network = temp_networks[temp_FVE_list.index(max_FVE)]
    #
    #     current_network.save_into_file()
    #     self._network = current_network
    #     return

    def train_network_and_save(self, training_interval=None, num_of_trainings=CONFIG_13):
        """num_of_trainings is the number of trainings that we are going to run, and
        then pick one that has the largest Fraction of Variance Explained (FVE),
        by doing this, we might avoid network with very poor quality
        """
        if training_interval is None: training_interval = self._index  # to avoid too much time on training
        my_file_list = coordinates_data_files_list(
            list_of_dir_of_coor_data_files=['../target/' + CONFIG_30]).get_list_of_coor_data_files()
        data_set = molecule_type.get_many_cossin_from_coordiantes_in_list_of_files(my_file_list)

        max_FVE = 0
        current_network = None

        for _ in range(num_of_trainings):
            if CONFIG_45 == 'pybrain':
                temp_network = neural_network_for_simulation(index=self._index,
                                                             data_set_for_training=data_set,
                                                             training_data_interval=training_interval,
                                                             )
            elif CONFIG_45 == 'keras':
                temp_network = autoencoder_Keras(index=self._index,
                                                 data_set_for_training=data_set,
                                                 training_data_interval=training_interval,
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
            procs_to_run_commands = list(range(len(commands)))
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
        for _ in range(num):
            self.run_one_iteration(current_iter)
            next_index = current_iter._index + 1
            current_iter = iteration(next_index, None)

        return

class single_biased_simulation_data(object):
    def __init__(self, my_network, file_for_single_biased_simulation_coor):
        """my_network is the corresponding network for this biased simulation"""
        self._file_for_single_biased_simulation_coor = file_for_single_biased_simulation_coor
        self._my_network = my_network
        temp_potential_center_string = file_for_single_biased_simulation_coor.split('_pc_[')[1].split(']')[0]
        self._potential_center = [float(item) for item in temp_potential_center_string.split(',')]
        self._force_constant = float(file_for_single_biased_simulation_coor.split('output_fc_')[1].split('_pc_')[0])
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
        PCs_transpose = list(zip(*PCs))
        center_of_data_cloud = map(lambda x: sum(x) / len(x), PCs_transpose)
        return center_of_data_cloud

    def get_offset_between_potential_center_and_data_cloud_center(self):
        """see if the push in this biased simulation actually works, large offset means it
        does not work well
        """
        PCs_average = self.get_center_of_data_cloud_in_this_biased_simulation()
        offset = [PCs_average[item] - self._potential_center[item] for item in range(self._dimension_of_PCs)]
        return offset

