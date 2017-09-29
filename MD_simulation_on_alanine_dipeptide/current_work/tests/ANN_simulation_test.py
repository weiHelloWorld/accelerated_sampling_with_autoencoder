'''This is a test for functionality of ANN_simulation.py
'''

import sys, os, math, subprocess, matplotlib
matplotlib.use('agg')

sys.path.append('../src/')  # add the source file folder

from ANN_simulation import *
from numpy.testing import assert_almost_equal, assert_equal


class test_Sutils(object):
    @staticmethod
    def test_mark_and_modify_pdb_for_calculating_RMSD_for_plumed():
        temp_out = 'temp_out.pdb'
        Sutils.mark_and_modify_pdb_for_calculating_RMSD_for_plumed('../resources/1l2y.pdb', temp_out,
                                get_index_list_with_selection_statement('../resources/1l2y.pdb', 'name CA'))
        a = Universe(temp_out)
        b = a.select_atoms('name CA')
        assert np.all(b.tempfactors) and np.all(b.occupancies)
        b = a.select_atoms('not name CA')
        assert not (np.any(b.tempfactors) or np.any(b.occupancies))
        subprocess.check_output(['rm', temp_out])
        return

    @staticmethod
    def test__generate_coordinates_from_pdb_files():
        # TODO
        return

    @staticmethod
    def test_write_some_frames_into_a_new_file():
        input_pdb = 'dependency/temp_output_0.pdb'
        output_pdb = "dependency/temp_output_0_interval_3.pdb"
        output_coor = output_pdb.replace('.pdb', '_coordinates.txt')
        actual_output_coor = 'dependency/temp_output_0_coor.txt'
        for interval in range(3, 10):
            Sutils.write_some_frames_into_a_new_file(input_pdb, 0, 0, interval, output_pdb)
            if os.path.exists(output_coor):
                subprocess.check_output(['rm', output_coor])
            Alanine_dipeptide.generate_coordinates_from_pdb_files(output_pdb)
            assert_almost_equal(np.loadtxt(output_coor), np.loadtxt(actual_output_coor)[::interval])
            subprocess.check_output(['rm', output_coor, output_pdb])
        return

    @staticmethod
    def test_get_boundary_points():
        """generate plotting for tests"""
        cov = [[0.1, 0], [0, 0.1]]  # diagonal covariance
        get_points = lambda mean: np.random.multivariate_normal(mean, cov, 50)
        points = reduce(lambda x, y: np.concatenate((x, y)), map(get_points, [[0, 1], [0, -1]]))
        boundary_points = Sutils.get_boundary_points(points, preprocessing=True)
        x, y = zip(*points)
        x1, y1 = zip(*boundary_points)
        fig, ax = plt.subplots()
        ax.scatter(x, y, c='b')
        ax.scatter(x1, y1, c='r')
        fig.savefig('test_get_boundary_points_noncircular.png')

        points = reduce(lambda x, y: np.concatenate((x, y)), map(get_points, [[-.8, -.8]]))
        boundary_points = Sutils.get_boundary_points(points, preprocessing=True, is_circular_boundary=True,
                                                     range_of_PCs=[[-1, 1], [-1, 1]])
        x, y = zip(*points)
        x1, y1 = zip(*boundary_points)
        fig, ax = plt.subplots()
        ax.scatter(x, y, c='b')
        ax.scatter(x1, y1, c='r')
        fig.savefig('test_get_boundary_points_circular.png')
        return

    @staticmethod
    def test_get_boundary_points_2_diagram():
        """diagram for the find_boundary algorithm"""
        dimensionality = 2
        fig, axes = plt.subplots(2, 2)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
        fig.set_size_inches(15, 15)
        # hist_matrix = np.random.randint(1, 10, size=(size_of_grid, size_of_grid))
        hist_matrix = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 3, 5, 3, 2, 1, 0],
            [0, 0, 2, 9, 6, 2, 0, 0],
            [0, 0, 5, 1, 7, 2, 0, 0],
            [0, 1, 2, 9, 8, 1, 0, 0],
            [0, 0, 0, 1, 4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        hist_matrix = np.array(hist_matrix)
        hist_matrix_processed = map(lambda x: map(lambda y: - np.exp(- y), x), hist_matrix)  # preprocessing process

        diff_with_neighbors = hist_matrix_processed - 1.0 / (2 * dimensionality) * sum(
            map(lambda x: np.roll(hist_matrix_processed, 1, axis=x)
                          + np.roll(hist_matrix_processed, -1, axis=x),
                range(dimensionality)
                )
        )
        temp_fontsize = 25
        sns.heatmap(hist_matrix, ax=axes[0][0], annot=True, cbar=False)
        sns.heatmap(hist_matrix_processed, ax=axes[0][1], annot=True, cbar=False)
        sns.heatmap(diff_with_neighbors, ax=axes[1][0], annot=True, cbar=False)
        sns.heatmap(diff_with_neighbors < 0, ax=axes[1][1], annot=False, cbar=False)
        axes[0][0].set_title(r'number of data points $n_i$', fontsize=temp_fontsize)
        axes[0][1].set_title(r'$p_i = -\exp{(-n_i)}$', fontsize=temp_fontsize)
        axes[1][0].text(2, 8.5, r'$v_i = p_i-\frac{1}{| K_i |}\sum_{j \in K_i} p_j$', fontsize=temp_fontsize)
        axes[1][1].set_title('locations of selected cells', fontsize=temp_fontsize)
        temp_annotation = ['(a)', '(b)', '(c)', '(d)']
        index = 0
        for _1 in axes:
            for ax in _1:
                ax.set_xlabel('$\\xi_1$', fontsize=temp_fontsize)
                ax.set_ylabel('$\\xi_2$', fontsize=temp_fontsize)
                ax.text(-0.5, 8.4, temp_annotation[index], fontsize=temp_fontsize - 5)
                index += 1
                #     fig.tight_layout()
        fig.savefig('diagram_of_finding_boundary.pdf', format='pdf', bbox_inches='tight')
        return

    @staticmethod
    def test_L_method():
        evaluation_values = [0, 0.1, 0.5, 0.85, 0.9, 0.93]
        nums = list(range(len(evaluation_values)))
        opt_num, x_data, y_data_left, y_data_right = Sutils.L_method(evaluation_values, nums)
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data_left)
        ax.plot(x_data, y_data_right)
        ax.scatter(nums, evaluation_values)
        fig.savefig("L_method.png")
        assert (opt_num == 4), opt_num
        return

    @staticmethod
    def test_rotating_coordinates():
        data = np.loadtxt('dependency/temp_Trp_cage_data/1l2y_coordinates.txt').reshape((38, 60, 3))[0]
        actual = Sutils.rotating_coordinates(data, [0,0,0], [0,0,1], np.pi / 2)
        expected = np.array([data[:, 1], - data[:,0], data[:,2]]).T
        assert_almost_equal(expected, actual)
        return

    @staticmethod
    def test__get_expression_script_for_plumed():
        with open('temp_plumed_script.txt', 'w') as my_f:
            my_f.write(Trp_cage.get_expression_script_for_plumed())
        return


class test_Alanine_dipeptide(object):
    @staticmethod
    def test_get_many_cossin_from_coordiantes_in_list_of_files():
        list_of_files = ['dependency/biased_output_fc_1000_x1_0.7_x2_-1.07_coordinates.txt']
        actual = Alanine_dipeptide().get_many_cossin_from_coordinates_in_list_of_files(list_of_files)
        assert_equal(100, len(actual))
        assert_equal(8, len(actual[0]))
        expected = np.loadtxt('dependency/output_cossin.txt')
        assert_almost_equal(expected, actual)
        return

    @staticmethod
    def test_get_many_dihedrals_from_cossin():
        angle = [.4, -.7, math.pi, -.45]
        cossin = [[1, 0, -1, 0, 1, 0, -1, 0], [0, 1, 0, -1, 0, 1, 0, -1],
                  reduce(lambda x, y: x + y, map(lambda x: [cos(x), sin(x)], angle))
                  ]
        actual = Alanine_dipeptide().get_many_dihedrals_from_cossin(cossin)
        expected = [[0, 0, 0, 0], [math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2], angle]
        for item in range(len(actual)):
            for index in range(4):
                assert_almost_equal(actual[item][index], expected[item][index], 4)
        return

    @staticmethod
    def test_get_many_dihedrals_from_coordinates_in_file():
        list_of_files = ['dependency/biased_output_fc_1000_x1_0.7_x2_-1.07_coordinates.txt']
        actual = Alanine_dipeptide.get_many_dihedrals_from_coordinates_in_file(list_of_files)
        expected = np.loadtxt('dependency/output_dihedrals.txt')
        assert_almost_equal(actual, expected)
        return

    @staticmethod
    def test_generate_coordinates_from_pdb_files():
        pdb_file_name = 'dependency/temp_output_0.pdb'
        actual_output_file = pdb_file_name.replace('.pdb', '_coordinates.txt')
        expected_output_files = 'dependency/temp_output_0_coor.txt'
        for interval in range(1, 10):
            if interval != 1:
                actual_output_file = pdb_file_name.replace('.pdb', '_int_%d_coordinates.txt' % interval)
            if os.path.exists(actual_output_file):
                subprocess.check_output(['rm', actual_output_file])
            Alanine_dipeptide.generate_coordinates_from_pdb_files(pdb_file_name, step_interval=interval)
            assert_equal(np.loadtxt(actual_output_file), np.loadtxt(expected_output_files)[::interval])
            subprocess.check_output(['rm', actual_output_file])
        return


class test_Trp_cage(object):
    @staticmethod
    def test_get_many_cossin_from_coordiantes_in_list_of_files():
        # TODO
        return

    @staticmethod
    def test_get_many_dihedrals_from_coordinates_in_file():
        # TODO
        return

    @staticmethod
    def test_generate_coordinates_from_pdb_files():
        # TODO
        return

    @staticmethod
    def test_get_pairwise_distance_matrices_of_alpha_carbon():
        # TODO
        return

    @staticmethod
    def test_get_non_repeated_pairwise_distance_as_list_of_alpha_carbon():
        pdb_file_list = ['dependency/temp_Trp_cage_data/1l2y.pdb']
        a = Trp_cage.get_pairwise_distance_matrices_of_selected_atoms(pdb_file_list)
        a = [item.reshape(400, 1) for item in a]
        b = Trp_cage.get_non_repeated_pairwise_distance(pdb_file_list)
        assert (len(a) == len(b))
        for _1 in range(len(b)):
            for _2 in b[_1]:
                assert (_2 in a[_1])
        return

    @staticmethod
    def test_get_pairwise_distance_matrices_of_alpha_carbon():
        actual = Trp_cage.get_pairwise_distance_matrices_of_selected_atoms(['dependency/temp_Trp_cage_data/1l2y.pdb'])[0]
        expected = np.loadtxt("dependency/test_get_pairwise_distance_matrices_of_alpha_carbon.txt")
        assert_almost_equal(actual, expected)
        return

    @staticmethod
    def test_rotating_dihedral_angles_and_save_to_pdb():
        pdb_file = 'dependency/temp_Trp_cage_data/1l2y.pdb'
        output = 'temp_rotating_out.pdb'
        target_dihedrals_list = [np.ones((38, 38)), np.zeros((38, 38))]
        for target_dihedrals in target_dihedrals_list:
            Trp_cage.rotating_dihedral_angles_and_save_to_pdb(pdb_file, target_dihedrals, output)
            out_coor_file_list = Trp_cage.generate_coordinates_from_pdb_files(output)
            actual_dihedrals = Trp_cage.get_many_dihedrals_from_coordinates_in_file(out_coor_file_list)
            # print np.max(np.abs(actual_dihedrals - target_dihedrals))
            assert_almost_equal(actual_dihedrals, target_dihedrals, decimal=2)

        return


class test_cluster_management(object):
    @staticmethod
    def test_create_sge_files_from_a_file_containing_commands():
        input_file = 'dependency/command_file.txt'
        folder_to_store_sge_files = 'dependency/out_sge/'
        if os.path.exists(folder_to_store_sge_files):
            subprocess.check_output(['rm', '-rf', folder_to_store_sge_files])

        subprocess.check_output(['mkdir', folder_to_store_sge_files])

        temp = cluster_management()
        commands = temp.create_sge_files_from_a_file_containing_commands(input_file, folder_to_store_sge_files)
        commands = map(lambda x: x[:-1].strip(), commands)
        print commands
        for out_file in subprocess.check_output(['ls', folder_to_store_sge_files]).strip().split('\n'):
            with open(folder_to_store_sge_files + out_file, 'r') as temp_file:
                content = temp_file.readlines()
                content = map(lambda x: x.strip(), content)
                this_command = filter(lambda x: x.startswith('python'), content)
                print this_command[0]
                assert this_command[0] in commands

        subprocess.check_output(['rm', '-rf', folder_to_store_sge_files])
        return

    @staticmethod
    def test_generate_sge_filename_for_a_command():
        actual = cluster_management.generate_sge_filename_for_a_command('python main____work.py :::: && -- ../target')
        expected = 'python_main_work.py_target.sge'
        assert (actual == expected)
        return


class test_coordinates_data_files_list(object):
    @staticmethod
    def test__init__():
        folder = 'dependency/temp_data'
        num_of_coor_files = len(subprocess.check_output(['find', folder, '-name', "*_coordinates.txt"]).strip().split())
        a = coordinates_data_files_list([folder])
        assert len(a.get_list_of_coor_data_files()) == num_of_coor_files - 1      # one file is empty
        assert a.get_list_of_line_num_of_coor_data_file() == [100 for _ in range(num_of_coor_files - 1)]
        assert sorted(a.get_list_of_coor_data_files()) == a.get_list_of_coor_data_files()
        assert len(a.get_list_of_corresponding_pdb_files()) == num_of_coor_files - 1
        assert sorted(a.get_list_of_corresponding_pdb_files()) == a.get_list_of_corresponding_pdb_files()

    @staticmethod
    def test_create_sub_coor_data_files_list_using_filter_conditional():
        folder = 'dependency/temp_data'
        a = coordinates_data_files_list([folder])
        a_sub = a.create_sub_coor_data_files_list_using_filter_conditional(lambda x: '0.7' in x)
        for item in a_sub.get_list_of_coor_data_files():
            assert ('0.7' in item)
        return

    @staticmethod
    def test_get_pdb_name_and_corresponding_frame_index_with_global_coor_index():
        _1 = coordinates_data_files_list(['dependency/temp_data/'])
        pdb_files = _1.get_list_of_corresponding_pdb_files()
        for item in range(1, 602, 100):
            assert (_1.get_pdb_name_and_corresponding_frame_index_with_global_coor_index(item) == (pdb_files[item / 100], 1))
        return


class test_neural_network_for_simulation(object):
    def __init__(self):
        autoencoder_pkl_path = 'dependency/a_network_pkl_and_coef_file/network_1.pkl'
        self._network = Sutils.load_object_from_pkl_file(autoencoder_pkl_path)

    def test_get_mid_result(self):
        assert isinstance(self._network, neural_network_for_simulation)
        mid_result =  self._network.get_mid_result()
        # for _1 in range(4):
        #     np.savetxt('output_mid_result_%d.txt' % _1, [item[_1] for item in mid_result])

        for _1 in range(4):
            assert_almost_equal (np.loadtxt('dependency/out_mid_result/output_mid_result_%d.txt' % _1),
                                 [item[_1] for item in mid_result])
        return

    def test_write_coefficients_of_connections_into_file(self):
        # TODO: add testing for values, currently only tests basic functionality
        assert isinstance(self._network, neural_network_for_simulation)
        self._network.write_coefficients_of_connections_into_file('test_coeff.txt')
        return

    def test_write_expression_into_file(self):
        # TODO: add testing for values, currently only tests basic functionality
        assert isinstance(self._network, neural_network_for_simulation)
        self._network.write_expression_into_file('test_expression.txt')
        return

    def test_get_output_data(self):
        # TODO: add testing for values, currently only tests basic functionality
        assert isinstance(self._network, neural_network_for_simulation)
        np.savetxt('test_get_output_data.txt', self._network.get_output_data())
        return

    def test_get_training_error(self):
        # TODO: add testing for values, currently only tests basic functionality
        assert isinstance(self._network, neural_network_for_simulation)
        self._network.get_training_error()
        return

    def test_get_fraction_of_variance_explained(self):
        # TODO: add testing for values, currently only tests basic functionality
        assert isinstance(self._network, neural_network_for_simulation)
        self._network.get_fraction_of_variance_explained()
        return

    def test_get_proper_potential_centers_for_WHAM(self):
        # TODO: add testing for values, currently only tests basic functionality
        assert isinstance(self._network, neural_network_for_simulation)
        list_of_points = []
        for xx in np.linspace(-1, 1, 11):
            for yy in np.linspace(-1, 1, 11):
                list_of_points += [[xx, yy]]

        res = np.array(self._network.get_proper_potential_centers_for_WHAM(list_of_points, 0.1, 2))

        fig, ax = plt.subplots()
        ax.scatter(res.T[0], res.T[1])
        fig.savefig('test_get_proper_potential_centers_for_WHAM.png')


class test_autoencoder_Keras(object):
    def __init__(self):
        my_file_list = coordinates_data_files_list(['dependency/noncircular_alanine_exploration_data/'])
        self._data = np.array(Alanine_dipeptide.get_many_cossin_from_coordinates_in_list_of_files(
            my_file_list.get_list_of_coor_data_files()))

    def test_train(self):
        data = self._data
        dihedrals = Alanine_dipeptide.get_many_dihedrals_from_cossin(data)

        model = autoencoder_Keras(1447, data,
                                  node_num=[8, 15, 2, 15, 8],
                                  hidden_layers_types=[TanhLayer, TanhLayer, TanhLayer],
                                  network_parameters = [0.02, 0.9,0, True, [0.001]* 4],
                                  batch_size=100
                                  )
        model.train()

        PCs = model.get_PCs()
        [x, y] = zip(*PCs)

        psi = [item[2] for item in dihedrals]
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=psi, cmap='gist_rainbow')

        fig.savefig('try_keras_noncircular.png')
        return

    def test_train_2(self):
        data = self._data
        dihedrals = Alanine_dipeptide.get_many_dihedrals_from_cossin(data)
        model = autoencoder_Keras(1447, data,
                                  node_num=[8, 15, 4, 15, 8],
                                  hidden_layers_types=[TanhLayer, CircularLayer, TanhLayer],
                                  network_parameters = [0.1, 0.4,0, True, [0.001]* 4]
                                  )
        model.train()

        PCs = model.get_PCs()
        [x, y] = zip(*PCs)

        psi = [item[2] for item in dihedrals]
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=psi, cmap='gist_rainbow')

        fig.savefig('try_keras_circular.png')
        return

    def test_save_into_file(self):
        data = self._data
        model = autoencoder_Keras(1447, data,
                                  node_num=[8, 15, 2, 15, 8],
                                  hidden_layers_types=[TanhLayer, TanhLayer, TanhLayer],
                                  network_parameters=[0.02, 0.9,0, True, [0.001]* 4],
                                  batch_size=50
                                  )
        model.train().save_into_file('test_save_into_file.pkl')
        model.save_into_file('test_save_into_file_fraction.pkl', fraction_of_data_to_be_saved=0.5)
        return


class test_biased_simulation(object):
    @staticmethod
    def helper_biased_simulation_alanine_dipeptide(potential_center):
        autoencoder_coeff_file = 'dependency/test_biased_simulation/autoencoder_info_4.txt'
        autoencoder_pkl_file = 'dependency/test_biased_simulation/network_4.pkl'
        output_folder = 'temp_output_test_biased_simulation'

        if os.path.exists(output_folder):
            subprocess.check_output(['rm', '-rf', output_folder])

        subprocess.check_output(
            'python ../src/biased_simulation.py 50 5000 100 %s %s pc_%s --num_of_nodes %s --layer_types %s --platform CPU --data_type_in_input_layer 0'
            % (output_folder, autoencoder_coeff_file, potential_center, "8,15,4", "Tanh,Circular"),
            shell=True)

        Alanine_dipeptide.generate_coordinates_from_pdb_files(output_folder)
        fig, ax = plt.subplots()
        my_files = coordinates_data_files_list([output_folder]).get_list_of_coor_data_files()
        cossin_data = Alanine_dipeptide.get_many_cossin_from_coordinates_in_list_of_files(my_files)
        my_network = Sutils.load_object_from_pkl_file(autoencoder_pkl_file)
        assert (isinstance(my_network, autoencoder))
        PCs = my_network.get_PCs(cossin_data)
        x, y = zip(*PCs)
        ax.scatter(x, y)
        fig.savefig('test_biased_simulation_%s.png' % potential_center)
        subprocess.check_output(['rm', '-rf', output_folder])
        return

    @staticmethod
    def test_biased_simulation_alanine_dipeptide():
        for item in ['-1.57,-1.57', '0,0', '-0.9,0.9', '-2,2', '-2,1', '-2,-2']:
            test_biased_simulation.helper_biased_simulation_alanine_dipeptide(item.replace(' ',''))
        return

    @staticmethod
    def test_biased_simulation_alanine_dipeptide_with_metadynamics(use_well_tempered=0, biasfactor=-1):
        autoencoder_pkl_file = 'dependency/test_biased_simulation/temp_bias_with_MTD.pkl'
        output_folder = 'temp_output_test_biased_simulation'
        a = Sutils.load_object_from_pkl_file(autoencoder_pkl_file)
        a.write_expression_script_for_plumed('temp_info.txt')
        subprocess.check_output(
'python ../src/biased_simulation.py 50 50000 0 %s temp_info.txt pc_0,0 --MTD_pace 100 --platform CPU --bias_method MTD --MTD_biasfactor %f --MTD_WT %d --equilibration_steps 0 --data_type_in_input_layer 0'
                                % (output_folder, biasfactor, use_well_tempered), shell=True)
        subprocess.check_output(['python', '../src/generate_coordinates.py', 'Alanine_dipeptide', '--path', output_folder])
        fig, axes = plt.subplots(1, 3)
        data = np.loadtxt(
            output_folder + '/output_fc_0.000000_pc_[0.0,0.0]_coordinates.txt')
        data /= 5.0
        data = Sutils.remove_translation(data)
        PCs = a.get_PCs(data)
        ax = axes[0]
        ax.set_xlabel('CV1')
        ax.set_ylabel('CV2')
        ax.set_title('CV data generated by autoencoder')
        im = ax.scatter(PCs[:, 0], PCs[:, 1], c=range(PCs.shape[0]), cmap='gist_rainbow', s=4)
        fig.colorbar(im, ax=ax)

        out_data = np.loadtxt('temp_MTD_out.txt')

        ax = axes[1]
        im = ax.scatter(out_data[:, 1], out_data[:, 2], c=range(out_data.shape[0]), cmap='gist_rainbow', s=4)
        ax.set_xlabel('CV1')
        ax.set_ylabel('CV2')
        ax.set_title('CV data generated by PLUMED')
        fig.colorbar(im, ax=ax)

        ax = axes[2]
        dihedrals = Alanine_dipeptide.get_many_dihedrals_from_cossin(
            Alanine_dipeptide.get_many_cossin_from_coordinates(data))
        dihedrals = np.array(dihedrals)
        im = ax.scatter(dihedrals[:, 1], dihedrals[:, 2], c=range(len(dihedrals)), cmap="gist_rainbow", s=4)
        ax.set_xlabel('$\phi$')
        ax.set_ylabel('$\psi$')
        ax.set_title('data in phi-psi space')
        fig.colorbar(im, ax=ax)
        fig.set_size_inches((15, 5))
        fig.savefig('metadynamics_biasfactor_%f.png' % biasfactor)
        subprocess.check_output(['rm', '-rf', output_folder])
        return

    @staticmethod
    def test_biased_simulation_alanine_dipeptide_with_metadynamics_multiple():
        test_biased_simulation.test_biased_simulation_alanine_dipeptide_with_metadynamics(0, -1)
        for item in [5, 20, 100]:
            test_biased_simulation.test_biased_simulation_alanine_dipeptide_with_metadynamics(1, item)
        return


class test_get_and_save_cossin_and_metrics_from_a_data_folder():
    @staticmethod
    def test_get_and_save_cossin_and_metrics_from_a_data_folder():
        # TODO: add testing for values, currently only tests basic functionality
        subprocess.check_output(['python', '../src/get_and_save_cossin_and_metrics_from_a_data_folder.py', 'dependency/temp_Trp_cage_data'])
        for item in subprocess.check_output(['find', 'dependency/temp_Trp_cage_data', '-name', 'info*']).strip().split():
            temp = np.loadtxt(item)
            assert temp.shape[0] == 38
        return
