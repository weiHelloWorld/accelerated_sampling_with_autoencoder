from config import * # configuration file
from cluster_management import *
from autoencoders import *

"""note that all configurations for a class should be in function __init__(), and take configuration parameters
from config.py
"""

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
                                            input_data_for_plotting=None,   # input could be cossin or Cartesian
                                            color_option='pure',
                                            other_coloring=None,
                                            contain_title=True,
                                            title=None,
                                            axis_ranges=None,
                                            contain_colorbar=True,
                                            colorbar_label=None,
                                            smoothing_using_RNR = False,  # smooth the coloring values for data points using RadiusNeighborsRegressor()
                                            variance_using_RNR = False,  # get variance of coloring values over space using RNR
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
        if input_data_for_plotting is None:
            input_data = self._network._data_set
        else:
            input_data = input_data_for_plotting

        if plotting_space == "PC":
            PCs_to_plot = network.get_PCs(input_data= input_data)

            (x, y) = ([item[0] for item in PCs_to_plot], [item[1] for item in PCs_to_plot])
            labels = ["PC1", "PC2"]

        elif plotting_space == "phipsi":
            assert (isinstance(molecule_type, Alanine_dipeptide))
            temp_dihedrals = molecule_type.get_many_dihedrals_from_cossin(input_data)

            (x,y) = ([item[1] for item in temp_dihedrals], [item[2] for item in temp_dihedrals])
            labels = ["phi", "psi"]
        elif plotting_space == "1st_4th_dihedrals":
            assert (isinstance(molecule_type, Alanine_dipeptide))
            temp_dihedrals = molecule_type.get_many_dihedrals_from_cossin(input_data)

            (x,y) = ([item[0] for item in temp_dihedrals], [item[3] for item in temp_dihedrals])
            labels = ["dihedral_1", "dihedral_4"]
        else:
            raise Exception('plotting_space not defined!')

        # coloring
        if color_option == 'pure':
            coloring = 'red'
        elif color_option == 'step':
            coloring = list(range(len(x)))
        elif color_option == 'phi':
            assert (isinstance(molecule_type, Alanine_dipeptide))
            coloring = [item[1] for item in molecule_type.get_many_dihedrals_from_cossin(input_data)]
        elif color_option == 'psi':
            assert (isinstance(molecule_type, Alanine_dipeptide))
            coloring = [item[2] for item in molecule_type.get_many_dihedrals_from_cossin(input_data)]
        elif color_option == '1st_dihedral':
            assert (isinstance(molecule_type, Alanine_dipeptide))
            coloring = [item[0] for item in molecule_type.get_many_dihedrals_from_cossin(input_data)]
        elif color_option == '4th_dihedral':
            assert (isinstance(molecule_type, Alanine_dipeptide))
            coloring = [item[3] for item in molecule_type.get_many_dihedrals_from_cossin(input_data)]
        elif color_option == 'other':
            assert (len(other_coloring) == len(x)), (len(other_coloring), len(x))
            coloring = other_coloring
            if smoothing_using_RNR:    # smooth coloring using RNR
                r_neigh = RadiusNeighborsRegressor(radius=smoothing_radius, weights='uniform')
                temp_coors = [list(item) for item in zip(x, y)]
                r_neigh.fit(temp_coors, coloring)
                coloring = r_neigh.predict(temp_coors)
            elif variance_using_RNR:  # get variance of the coloring values over space, using RNR
                r_neigh = RadiusNeighborsRegressor(radius=smoothing_radius, weights='uniform')
                temp_coors = [list(item) for item in zip(x, y)]
                r_neigh.fit(temp_coors, coloring)
                coloring_mean = r_neigh.predict(temp_coors)
                r_neigh.fit(temp_coors, np.multiply(np.array(coloring), np.array(coloring)))
                coloring_square_mean = r_neigh.predict(temp_coors)
                coloring = coloring_square_mean - np.multiply(coloring_mean, coloring_mean)
        else:
            raise Exception('color_option not defined!')

        im = axis_object.scatter(x,y,s=4, c=coloring, cmap='gist_rainbow', picker=True)
        axis_object.set_xlabel(labels[0])
        axis_object.set_ylabel(labels[1])
        if contain_title:
            axis_object.set_title(title)

        if not axis_ranges is None:
            axis_object.set_xlim(axis_ranges[0])
            axis_object.set_ylim(axis_ranges[1])

        if contain_colorbar:
            temp_colorbar = fig_object.colorbar(im, ax=axis_object)
            if not colorbar_label is None:
                temp_colorbar.set_label(str(colorbar_label))

        # mouse clicking event
        if enable_mousing_clicking_event:
            folder_to_store_these_frames = 'temp_pdb'
            if not os.path.exists(folder_to_store_these_frames):
                subprocess.check_output(['mkdir', folder_to_store_these_frames])

            # should calculate step_interval
            total_num_of_lines_in_coor_files = sum(related_coor_list_obj.get_list_of_line_num_of_coor_data_file())
            step_interval = int(total_num_of_lines_in_coor_files / len(input_data))

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
                global temp_global_index_click
                temp_global_index_click = 0
                def onclick(event):
                    global temp_global_index_click
                    if isinstance(event.artist, matplotlib.collections.PathCollection):
                        ind_list = list(event.ind)
                        print ('onclick:')
                        for item in ind_list:
                            print(item, x[item], y[item])

                        temp_ind_list = [item * step_interval for item in ind_list]  # should include step_interval
                        average_x = sum([x[item] for item in ind_list]) / len(ind_list)
                        average_y = sum([y[item] for item in ind_list]) / len(ind_list)
                        # notation on the graph
                        axis_object.scatter([average_x], [average_y], s=50, marker='s')
                        axis_object.text(average_x, average_y, '%d' % temp_global_index_click, picker = False, fontsize=15)
                        out_file_name = folder_to_store_these_frames + '/%02d_temp_frames_[%f,%f].pdb' % \
                                                                    (temp_global_index_click, average_x, average_y)

                        temp_global_index_click += 1
                        related_coor_list_obj.write_pdb_frames_into_file_with_list_of_coor_index(temp_ind_list,
                            out_file_name=out_file_name)
                        # need to verify PCs generated from this output pdb file are consistent from those in the list selected
                        molecule_type.generate_coordinates_from_pdb_files(path_for_pdb=out_file_name)
                        if CONFIG_48 == "cossin":
                            temp_input_data = molecule_type.get_many_cossin_from_coordinates_in_list_of_files(
                                list_of_files=[out_file_name.replace('.pdb', '_coordinates.txt')])
                        elif CONFIG_48 == "Cartesian" or 'pairwise_distance':
                            scaling_factor = CONFIG_49
                            temp_input_data = np.loadtxt(out_file_name.replace('.pdb', '_coordinates.txt')) / scaling_factor
                            temp_input_data = Sutils.remove_translation(temp_input_data)
                        else:
                            raise Exception("input data type error")

                        PCs_of_points_selected = network.get_PCs(input_data=temp_input_data)
                        assert_almost_equal(PCs_of_points_selected, np.array([[x[item], y[item]] for item in ind_list]), decimal=4)

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

    def equilibration_check(self, coor_file_folder,
                            scaling_factor, num_of_splits, save_fig=True,
                            starting_index_of_last_few_frames=0
                            ):
        """this function checks equilibration by plotting each individual runs in PC space, colored with 'step',
        note: inputs should be Cartesian coordinates, the case with input using cossin is not implemented
        """
        import scipy
        ks_stats_list = []
        temp_arrow_list = []
        potential_centers_list = []
        temp_arrow_start_list = []
        _1 = coordinates_data_files_list([coor_file_folder])
        for item in _1.get_list_of_coor_data_files():
            data = np.loadtxt(item)[starting_index_of_last_few_frames:] / scaling_factor
            data = Sutils.remove_translation(data)
            potential_centers_list.append([float(item_1) for item_1 in item.split('_pc_[')[1].split(']')[0].split(',')])
            # do analysis using K-S test
            PCs = self._network.get_PCs(data)
            dim_of_PCs = PCs.shape[1]
            PCs = PCs[:int(PCs.shape[0]) / num_of_splits * num_of_splits]   # in case that PCs cannot be splitted evenly
            samples_for_KS_testing = np.split(PCs, num_of_splits)
            ks_stats = max([
                sum(
                    [scipy.stats.ks_2samp(samples_for_KS_testing[xx][:,subindex], samples_for_KS_testing[yy][:,subindex])[0]
                        for subindex in range(dim_of_PCs) 
                    ]) / float(dim_of_PCs)
                 for xx in range(num_of_splits) for yy in range(xx + 1, num_of_splits)] 
            )
            ks_stats_list.append(ks_stats)
            # plot arrow from center of first split to last split
            temp_arrow_start = np.average(samples_for_KS_testing[0], axis=0)
            temp_arrow_end = np.average(samples_for_KS_testing[-1], axis=0)
            temp_arrow = (temp_arrow_end - temp_arrow_start)
            assert (temp_arrow.shape[0] == 2), temp_arrow.shape[0]
            temp_arrow_list.append(temp_arrow)
            temp_arrow_start_list.append(temp_arrow_start)
            
            fig, ax = plt.subplots()
            self.plotting_with_coloring_option("PC", fig, ax, input_data_for_plotting=data, color_option='step',
                                            title=item.strip().split('/')[-1])
            ax.quiver([temp_arrow_start[0]], [temp_arrow_start[1]], [temp_arrow[0]], [temp_arrow[1]],
                      units="xy", scale=1)
            if save_fig:
                fig.savefig(ax.get_title() + '.png')

        # plotting K-S stats
        potential_centers_list = np.array(potential_centers_list)
        temp_arrow_list = np.array(temp_arrow_list)
        temp_arrow_start_list = np.array(temp_arrow_start_list)
        fig, ax = plt.subplots()
        im = ax.scatter(potential_centers_list[:,0], potential_centers_list[:,1],  c=ks_stats_list, cmap="Blues")
        col_bar = fig.colorbar(im, ax=ax)
        col_bar.set_label("KS value")
        for pc, arr_start in zip(potential_centers_list, temp_arrow_start_list):
            # connect potential center to starting point of arrow with dashed line
            ax.plot([pc[0], arr_start[0]], [pc[1], arr_start[1]], linestyle='dotted')

        ax.quiver(temp_arrow_start_list[:,0], temp_arrow_start_list[:,1],
                  temp_arrow_list[:,0], temp_arrow_list[:,1],
                  units = 'xy', scale=1)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.set_size_inches((10, 10))
        fig.savefig("temp_harmonic_centers_and_stats.png")

        return


class iteration(object):
    def __init__(self, index,
                 network=None # if you want to start with existing network, assign value to "network"
                 ):
        self._index = index
        self._network = network

    @staticmethod
    def preprocessing(machine_to_run_simulations = CONFIG_24, target_folder=None):
        """
        1. aligned structure
        2. generate coordinate files
        """
        reference_suffix_list = CONFIG_63
        reference_configs = CONFIG_62
        atom_selection_list = CONFIG_64
        assert (len(reference_configs) == len(reference_suffix_list)), (
        len(reference_configs), len(reference_suffix_list))
        num_of_reference_configs = len(reference_configs)
        if not target_folder is None:
            temp_target_folder = target_folder
        else:
            if isinstance(molecule_type, Trp_cage):
                temp_target_folder = '../target/Trp_cage'
            elif isinstance(molecule_type, BetaHairpin):
                temp_target_folder = '../target/BetaHairpin'
            elif isinstance(molecule_type, Alanine_dipeptide):
                temp_target_folder = '../target/Alanine_dipeptide'
            elif isinstance(molecule_type, Src_kinase):
                temp_target_folder = '../target/Src_kinase'
            else:
                raise Exception("molecule type error")

        for _1 in range(num_of_reference_configs):
            temp_command_list = ['python', 'structural_alignment.py', temp_target_folder,
                                 '--ref', reference_configs[_1], '--suffix', reference_suffix_list[_1],
                                 '--atom_selection', atom_selection_list[_1]
                                 ]
            if machine_to_run_simulations == 'local':
                subprocess.check_output(temp_command_list)
            elif machine_to_run_simulations == 'cluster':
                temp_command = ' '.join(['"%s"' % item for item in temp_command_list]) + ' 2> /dev/null '  # TODO: does it work by adding quotation marks to everything
                cluster_management.run_a_command_and_wait_on_cluster(command=temp_command)
            else:
                raise Exception('machine type error')

        molecule_type.generate_coordinates_from_pdb_files(path_for_pdb=temp_target_folder)
        return

    def train_network_and_save(self, machine_to_run_simulations = CONFIG_24,
                               training_interval=1, num_of_trainings=CONFIG_13):
        """num_of_trainings is the number of trainings that we are going to run, and
        then pick one that has the largest Fraction of Variance Explained (FVE),
        by doing this, we might avoid network with very poor quality
        """
        command = 'python ../src/train_network_and_save_for_iter.py %d --training_interval %d --num_of_trainings %d' %\
                  (self._index, training_interval, num_of_trainings)
        if machine_to_run_simulations == 'local':
            print command
            temp_output = subprocess.check_output(command.strip().split(' '))
            autoencoder_filename = temp_output.strip().split('\n')[-1]
        elif machine_to_run_simulations == 'cluster':
            command = 'OMP_NUM_THREADS=6  ' + command
            job_id = cluster_management.run_a_command_and_wait_on_cluster(command=command)
            output_file, _ = cluster_management.get_output_and_err_with_job_id(job_id=job_id)
            temp_output = subprocess.check_output(['cat', output_file])
            assert (temp_output.strip().split('\n')[-1] == 'This job is DONE!')
            autoencoder_filename = temp_output.strip().split('\n')[-2]
        else:
            raise Exception('machine type error')

        print temp_output
        self._network = autoencoder.load_from_pkl_file(autoencoder_filename)
        return

    def prepare_simulation(self):
        if CONFIG_28 == "CustomManyParticleForce":
            self._network.write_expression_into_file()
        elif CONFIG_28 == "ANN_Force":
            self._network.write_coefficients_of_connections_into_file()
        else:
            raise Exception("force type not defined!")
        return

    def run_simulation(self, machine_to_run_simulations = CONFIG_24, commands = None, cuda=None):
        if cuda is None:
            cuda = (CONFIG_23 == 'CUDA')
        if commands is None:
            commands = self._network.get_commands_for_further_biased_simulations()
        if machine_to_run_simulations == 'cluster':
            cluster_management.create_sge_files_for_commands(list_of_commands_to_run=commands,
                                                             run_on_gpu=cuda)
            cluster_management.monitor_status_and_submit_periodically(num = CONFIG_14,
                            monitor_mode='normal',
                            num_of_running_jobs_when_allowed_to_stop = 500)  # should not loop forever
        elif machine_to_run_simulations == 'local':
            num_of_simulations_run_in_parallel = CONFIG_56
            total_num_failed_jobs = 0
            for item in range(int(len(commands) / num_of_simulations_run_in_parallel) + 1):
                temp_commands_parallel = commands[item * num_of_simulations_run_in_parallel: (item + 1) * num_of_simulations_run_in_parallel]
                print ("running: \t" + '\n'.join(temp_commands_parallel))
                procs_to_run_commands = [subprocess.Popen(_1.strip().split()) for _1 in temp_commands_parallel]
                exit_codes = [p.wait() for p in procs_to_run_commands]
                total_num_failed_jobs += sum(exit_codes)

            assert (total_num_failed_jobs < CONFIG_31)  # we could not have more than CONFIG_31 simulations failed in each iteration
        else:
            raise Exception('machine type error')

        # next line only when the jobs are done, check this
        if CONFIG_29:
            molecule_type.remove_water_mol_and_Cl_from_pdb_file(preserve_original_file = CONFIG_50)

        return


class simulation_with_ANN_main(object):
    def __init__(self, num_of_iterations = 1,
                 initial_iteration=None,  # this is where we start with
                 training_interval = None,
                 ):
        self._num_of_iterations = num_of_iterations
        self._initial_iteration = initial_iteration
        self._training_interval = training_interval
        print "running iterations for system: %s" % CONFIG_30
        return

    def run_one_iteration(self, one_iteration):
        one_iteration.preprocessing()
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

    def get_center_of_data_cloud_in_this_biased_simulation(self, input_data_type):
        if input_data_type == 'cossin':
            PCs = self._my_network.get_PCs(molecule_type.get_many_cossin_from_coordinates_in_list_of_files(
                [self._file_for_single_biased_simulation_coor]))
        elif input_data_type == 'Cartesian':
            scaling_factor = CONFIG_49
            temp_data = np.loadtxt(self._file_for_single_biased_simulation_coor) / scaling_factor
            temp_data = Sutils.remove_translation(temp_data)
            PCs = self._my_network.get_PCs(temp_data)
        else:
            raise Exception('error input_data_type')

        assert(len(PCs[0]) == self._dimension_of_PCs)
        assert(len(PCs) == self._number_of_data)
        PCs_transpose = list(zip(*PCs))
        center_of_data_cloud = map(lambda x: sum(x) / len(x), PCs_transpose)
        return center_of_data_cloud

    def get_offset_between_potential_center_and_data_cloud_center(self, input_data_type):
        """see if the push in this biased simulation actually works, large offset means it
        does not work well
        """
        PCs_average = self.get_center_of_data_cloud_in_this_biased_simulation(input_data_type)
        offset = [PCs_average[item] - self._potential_center[item] for item in range(self._dimension_of_PCs)]
        return offset

