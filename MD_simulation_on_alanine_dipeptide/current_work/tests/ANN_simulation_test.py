'''This is a test for functionality of ANN_simulation.py
'''

import sys, os, math, subprocess, matplotlib
from functools import reduce
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
        input_pdb = '../tests/dependency/temp_output_0.pdb'
        output_pdb = "../tests/dependency/temp_output_0_interval_3.pdb"
        output_coor = output_pdb.replace('.pdb', '_coordinates.txt')
        actual_output_coor = '../tests/dependency/temp_output_0_coor.txt'
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
        points = reduce(lambda x, y: np.concatenate((x, y)), list(map(get_points, [[0, 1], [0, -1]])))
        boundary_points = Sutils.get_boundary_points(points, preprocessing=True)
        x, y = list(zip(*points))
        x1, y1 = list(zip(*boundary_points))
        fig, ax = plt.subplots()
        ax.scatter(x, y, c='b')
        ax.scatter(x1, y1, c='r')
        fig.savefig('test_get_boundary_points_noncircular.png')

        points = reduce(lambda x, y: np.concatenate((x, y)), list(map(get_points, [[-.8, -.8]])))
        boundary_points = Sutils.get_boundary_points(points, preprocessing=True, is_circular_boundary=True,
                                                     range_of_PCs=[[-1, 1], [-1, 1]])
        x, y = list(zip(*points))
        x1, y1 = list(zip(*boundary_points))
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
        hist_matrix_processed = [[- np.exp(- y) for y in x] for x in hist_matrix]  # preprocessing process

        diff_with_neighbors = hist_matrix_processed - 1.0 / (2 * dimensionality) * sum(
            [np.roll(hist_matrix_processed, 1, axis=x)
                          + np.roll(hist_matrix_processed, -1, axis=x) for x in range(dimensionality)]
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
        data = np.loadtxt('../tests/dependency/temp_Trp_cage_data/1l2y_coordinates.txt').reshape((38, 60, 3))[0]
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
        list_of_files = ['../tests/dependency/biased_output_fc_1000_x1_0.7_x2_-1.07_coordinates.txt']
        actual = Alanine_dipeptide().get_many_cossin_from_coordinates_in_list_of_files(list_of_files)
        assert_equal(100, len(actual))
        assert_equal(8, len(actual[0]))
        expected = np.loadtxt('../tests/dependency/output_cossin.txt')
        assert_almost_equal(expected, actual)
        return

    @staticmethod
    def test_get_many_dihedrals_from_cossin():
        angle = [.4, -.7, math.pi, -.45]
        cossin = [[1, 0, -1, 0, 1, 0, -1, 0], [0, 1, 0, -1, 0, 1, 0, -1],
                  reduce(lambda x, y: x + y, [[cos(x), sin(x)] for x in angle])
                  ]
        actual = Alanine_dipeptide().get_many_dihedrals_from_cossin(cossin)
        expected = [[0, 0, 0, 0], [math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2], angle]
        for item in range(len(actual)):
            for index in range(4):
                assert_almost_equal(actual[item][index], expected[item][index], 4)
        return

    @staticmethod
    def test_get_many_dihedrals_from_coordinates_in_file():
        list_of_files = ['../tests/dependency/biased_output_fc_1000_x1_0.7_x2_-1.07_coordinates.txt']
        actual = Alanine_dipeptide.get_many_dihedrals_from_coordinates_in_file(list_of_files)
        expected = np.loadtxt('../tests/dependency/output_dihedrals.txt')
        assert_almost_equal(actual, expected)
        return

    @staticmethod
    def test_generate_coordinates_from_pdb_files():
        pdb_file_name = '../tests/dependency/temp_output_0.pdb'
        actual_output_file = pdb_file_name.replace('.pdb', '_coordinates.txt')
        expected_output_files = '../tests/dependency/temp_output_0_coor.txt'
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
        pdb_file_list = ['../tests/dependency/temp_Trp_cage_data/1l2y.pdb']
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
        actual = Trp_cage.get_pairwise_distance_matrices_of_selected_atoms(['../tests/dependency/temp_Trp_cage_data/1l2y.pdb'])[0]
        expected = np.loadtxt("../tests/dependency/test_get_pairwise_distance_matrices_of_alpha_carbon.txt")
        assert_almost_equal(actual, expected)
        return

    @staticmethod
    def test_rotating_dihedral_angles_and_save_to_pdb():
        pdb_file = '../tests/dependency/temp_Trp_cage_data/1l2y.pdb'
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
        server_name, _ = cluster_management.get_server_and_user()
        if not server_name == 'kengyangyao-pc':
            input_file = '../tests/dependency/command_file.txt'
            folder_to_store_sge_files = '../tests/dependency/out_sge/'
            if os.path.exists(folder_to_store_sge_files):
                subprocess.check_output(['rm', '-rf', folder_to_store_sge_files])

            subprocess.check_output(['mkdir', folder_to_store_sge_files])

            temp = cluster_management()
            commands = temp.create_sge_files_from_a_file_containing_commands(input_file, folder_to_store_sge_files)
            commands = [x[:-1].strip() for x in commands]
            print(commands)
            for out_file in subprocess.check_output(['ls', folder_to_store_sge_files]).strip().split('\n'):
                with open(folder_to_store_sge_files + out_file, 'r') as temp_file:
                    content = temp_file.readlines()
                    content = [x.strip() for x in content]
                    this_command = [x for x in content if x.startswith('python')]
                    print(this_command[0])
                    assert this_command[0] in commands
            subprocess.check_output(['rm', '-rf', folder_to_store_sge_files])
        return

    @staticmethod
    def test_generate_sge_filename_for_a_command():
        actual = cluster_management.generate_sge_filename_for_a_command('python main____work.py :::: && -- ../target')
        expected = '_main_work.py_target.sge'
        assert (actual == expected), (actual, expected)
        return


class test_coordinates_data_files_list(object):
    @staticmethod
    def test__init__():
        folder = '../tests/dependency/temp_data'
        num_of_coor_files = len(subprocess.check_output(['find', folder, '-name', "*_coordinates.txt"]).strip().split())
        a = coordinates_data_files_list([folder])
        assert len(a.get_list_of_coor_data_files()) == num_of_coor_files - 1      # one file is empty
        assert a.get_list_of_line_num_of_coor_data_file() == [100 for _ in range(num_of_coor_files - 1)]
        assert sorted(a.get_list_of_coor_data_files()) == a.get_list_of_coor_data_files()
        assert len(a.get_list_of_corresponding_pdb_files()) == num_of_coor_files - 1
        assert sorted(a.get_list_of_corresponding_pdb_files()) == a.get_list_of_corresponding_pdb_files()

    @staticmethod
    def test_create_sub_coor_data_files_list_using_filter_conditional():
        folder = '../tests/dependency/temp_data'
        a = coordinates_data_files_list([folder])
        a_sub = a.create_sub_coor_data_files_list_using_filter_conditional(lambda x: '0.7' in x)
        for item in a_sub.get_list_of_coor_data_files():
            assert ('0.7' in item)
        return

    @staticmethod
    def test_get_pdb_name_and_corresponding_frame_index_with_global_coor_index():
        _1 = coordinates_data_files_list(['../tests/dependency/temp_data/'])
        pdb_files = _1.get_list_of_corresponding_pdb_files()
        for item in range(1, 602, 100):
            assert (_1.get_pdb_name_and_corresponding_frame_index_with_global_coor_index(item) == (pdb_files[item / 100], 1))
        return


class test_autoencoder_Keras(object):
    def __init__(self):
        my_file_list = coordinates_data_files_list(['../tests/dependency/noncircular_alanine_exploration_data/'])
        self._data = np.array(Alanine_dipeptide.get_many_cossin_from_coordinates_in_list_of_files(
            my_file_list.get_list_of_coor_data_files()))
        self._dihedrals = Alanine_dipeptide.get_many_dihedrals_from_cossin(self._data)

    def test_train(self):
        data, dihedrals = self._data, self._dihedrals
        hidden_layers_list = [["Tanh", "Tanh", "Tanh", "Tanh", "Tanh", "Tanh", "Tanh"],
                              ["Sigmoid", "Sigmoid", "Sigmoid", "Sigmoid", "Tanh", "Sigmoid", "Tanh"]]
        reg_list = [0.001, 0]
        for item_activation in range(2):
            for is_hi, hier_var in [(0, 0), (1,0), (1,1), (1,2)]:
                model = autoencoder_Keras(1447, data,
                                          data_files=['/tmp/train_in.npy', '/tmp/train_out.npy'],
                                          node_num=[8, 8, 15, 8, 2, 15, 8, 8, 8],
                                          hidden_layers_types=hidden_layers_list[item_activation],
                                          network_parameters = [0.02, 0.9,0, True, [reg_list[item_activation]]* 8],
                                          batch_size=100, hierarchical=is_hi, hi_variant=hier_var
                                          )
                _, history = model.train()
                PCs = model.get_PCs()
                [x, y] = list(zip(*PCs))
                psi = [item[2] for item in dihedrals]
                fig, ax = plt.subplots()
                ax.scatter(x, y, c=psi, cmap='gist_rainbow')
                ax.set_title("FVE = %f" % model.get_fraction_of_variance_explained())
                file_name = 'try_keras_noncircular_hierarchical_%d_%d_act_%d.pkl' % (is_hi, hier_var, item_activation)
                model.save_into_file(file_name)
                fig.savefig(file_name.replace('.pkl', '.png'))
        return history

    def test_train_with_different_mse_weights(self):
        data, dihedrals = self._data, self._dihedrals
        for _1, weights in enumerate([None, np.array([1,1,0,0,0,0,1,1]),
                                      np.array([0,0,1,1,1,1,0,0]), np.array([1,1,1,1,0,0,0,0])]):
            model = autoencoder_Keras(1447, data,
                                      node_num=[8, 8, 15, 8, 2, 15, 8, 8, 8],
                                      hidden_layers_types=["Tanh", "Tanh", "Tanh", "Tanh", "Tanh", "Tanh", "Tanh"],
                                      network_parameters=[0.02, 0.9, 0, True, [0.001] * 8],
                                      batch_size=100, hierarchical=0,
                                      mse_weights=weights
                                      )
            _, history = model.train()
            PCs = model.get_PCs()
            [x, y] = list(zip(*PCs))
            psi = [item[2] for item in dihedrals]
            fig, ax = plt.subplots()
            ax.scatter(x, y, c=psi, cmap='gist_rainbow')
            model.save_into_file('try_diff_weights_%02d.pkl' % _1)
            fig.savefig('try_diff_weights_%02d.png' % _1)
        return

    def test_train_2(self):
        data, dihedrals = self._data, self._dihedrals
        model = autoencoder_Keras(1447, data,
                                  node_num=[8, 15, 4, 15, 8],
                                  hidden_layers_types=["Tanh", "Circular", "Tanh"],
                                  network_parameters = [0.1, 0.4,0, True, [0.001]* 4],
                                  hierarchical=False
                                  )
        model.train()
        PCs = model.get_PCs()
        [x, y] = list(zip(*PCs))
        psi = [item[2] for item in dihedrals]
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=psi, cmap='gist_rainbow')

        fig.savefig('try_keras_circular.png')
        return

    def test_save_into_file_and_load(self):
        data = self._data
        model = autoencoder_Keras(1447, data,
                                  node_num=[8, 15, 2, 15, 8],
                                  hidden_layers_types=["Tanh", "Tanh", "Tanh"],
                                  network_parameters=[0.02, 0.9,0, True, [0.001]* 4],
                                  batch_size=50,
                                  data_files=['test_save_into_file.npy', 'test_save_into_file.npy']
                                  )
        model.train()
        model.save_into_file('test_save_into_file.pkl')
        model.save_into_file('test_save_into_file_fraction.pkl', fraction_of_data_to_be_saved=0.5)
        model.save_into_file('temp_save/complicated/path/temp.pkl')
        _ = autoencoder.load_from_pkl_file('test_save_into_file.pkl')
        return

    def check_two_plumed_strings_containing_floats(self, string_1, string_2):
        """due to precision issue, string literals may not be exactly the same for two plumed strings, so we
                need to explicitly compare the float values"""
        def is_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
        split_1 = re.split(' |\n|=|,', string_1)
        split_2 = re.split(' |\n|=|,', string_2)
        assert (len(split_1) == len(split_2)), (len(split_1), len(split_2))
        for _1, _2 in zip(split_1, split_2):
            if is_float(_1):
                assert_almost_equal(float(_1), float(_2), decimal=4)
            else:
                assert (_1 == _2), (_1, _2)
        return

    def test_get_plumed_script_for_biased_simulation_with_solute_pairwise_dis_and_solvent_cg_input_and_ANN(self):
        scaling_factor_v = 26.9704478916
        scaling_factor_u = 29.1703348377
        r_high = 5.5
        atom_indices = list(range(1, 25))
        water_index_string = '75-11421:3'

        ae = autoencoder.load_from_pkl_file('../tests/dependency/solute_plus_solvent_AE/temp_alpha_0.5.pkl')
        with open('../tests/dependency/solute_plus_solvent_AE/temp_plumed.txt', 'r') as my_f:
            expected_plumed = my_f.read().strip()
        plumed_string = ae.get_plumed_script_for_biased_simulation_with_solute_pairwise_dis_and_solvent_cg_input_and_ANN(
            list(range(1, 25)), scaling_factor_u, water_index_string, atom_indices, -5, r_high, scaling_factor_v)
        self.check_two_plumed_strings_containing_floats(plumed_string, expected_plumed)

        AE = autoencoder.load_from_pkl_file('../tests/dependency/solvent_AE/solvent_test.pkl')
        with open('../tests/dependency/solvent_AE/temp_plumed.txt', 'r') as my_f:
            expected_plumed = my_f.read().strip()
        plumed_string = AE.get_plumed_script_for_biased_simulation_with_INDUS_cg_input_and_ANN(
            water_index_string, atom_indices, -5, r_high, scaling_factor_v).strip()
        self.check_two_plumed_strings_containing_floats(plumed_string, expected_plumed)
        return

class test_autoencoder_torch(object):
    @staticmethod
    def test_general_train_save_and_load():
        data = np.random.rand(1000, 21)
        a = autoencoder_torch(1447, data,
                              output_data_set=data,
                              hierarchical=True, hi_variant=2,
                              batch_size=500,
                              node_num=[21, 100, 2, 100, 21],
                              hidden_layers_types=['tanh', 'tanh', 'tanh'], epochs=10)
        a.train(lag_time=10)
        a.save_into_file('/tmp/temp_save.pkl')
        torch.save(a._ae, '/tmp/temp.df')
        model_1 = torch.load('/tmp/temp.df')
        torch.save(a._ae.state_dict(), '/tmp/temp_2.df')
        model_2 = AE_net([21, 100, 2], [2, 100, 21], activations=a._hidden_layers_type + ['linear'],
                         hi_variant=2, hierarchical=True).cuda()
        model_2.load_state_dict(torch.load('/tmp/temp_2.df'))
        data_in = torch.rand(1000, 21).cuda()
        assert_almost_equal(model_1(data_in)[0].cpu().data.numpy(), a._ae(data_in)[0].cpu().data.numpy())
        assert_almost_equal(model_2(data_in)[0].cpu().data.numpy(), a._ae(data_in)[0].cpu().data.numpy())
        _ = autoencoder_torch.load_from_pkl_file('/tmp/temp_save.pkl')
        return

    @staticmethod
    def test_time_lagged_AE_stored_data_saving():
        data = np.random.rand(1000, 21)
        a = autoencoder_torch(1447, data,
                              output_data_set=data,
                              hierarchical=True, hi_variant=2,
                              batch_size=500,
                              node_num=[21, 100, 2, 100, 21],
                              hidden_layers_types=['tanh', 'tanh', 'tanh'], epochs=10)
        a.train(lag_time=10)
        assert_almost_equal(a._data_set, data[:-10])
        assert_almost_equal(a._output_data_set, data[10:])
        return

    @staticmethod
    def test_save_and_load_data():
        data = np.random.rand(1000, 21)
        a = autoencoder_torch(1447, data,
                              output_data_set=data,
                              hierarchical=True,
                              batch_size=500,
                              node_num=[21, 100, 2, 100, 21], epochs=10,
                              data_files=['data.npy', 'data.npy'])
        a.train(lag_time=0)
        a.save_into_file('temp_save_pytorch/temp_save.pkl')
        assert (os.path.isfile('temp_save_pytorch/data.npy'))
        b = autoencoder_torch.load_from_pkl_file('temp_save_pytorch/temp_save.pkl')
        assert_almost_equal(a._data_set, b._data_set)
        assert_almost_equal(a._output_data_set, b._output_data_set)
        return


class test_biased_simulation(object):
    @staticmethod
    def helper_biased_simulation_alanine_dipeptide(potential_center):
        autoencoder_coeff_file = 'autoencoder_info_9.txt'
        autoencoder_pkl_file = '../tests/dependency/test_biased_simulation/network_9.pkl'
        my_network = autoencoder.load_from_pkl_file(autoencoder_pkl_file)
        assert (isinstance(my_network, autoencoder))
        my_network.write_coefficients_of_connections_into_file(autoencoder_coeff_file)
        output_folder = 'temp_output_test_biased_simulation'

        if os.path.exists(output_folder):
            subprocess.check_output(['rm', '-rf', output_folder])

        subprocess.check_output(
            'python ../src/biased_simulation.py 50 5000 5000 %s %s pc_%s --num_of_nodes %s --layer_types %s --platform CPU --data_type_in_input_layer 1'
            % (output_folder, autoencoder_coeff_file, potential_center, "21,40,2", "Tanh,Tanh"),
            shell=True)

        Alanine_dipeptide.generate_coordinates_from_pdb_files(output_folder)
        fig, ax = plt.subplots()
        input_data = coordinates_data_files_list([output_folder]).get_coor_data(5.0)
        input_data = Sutils.remove_translation(input_data)
        PCs = my_network.get_PCs(input_data)
        x, y = list(zip(*PCs))
        ax.scatter(x, y, c=list(range(len(x))), cmap='gist_rainbow')
        fig.savefig('test_biased_simulation_%s.png' % potential_center)
        subprocess.check_output(['rm', '-rf', output_folder])
        return

    @staticmethod
    def test_biased_simulation_alanine_dipeptide():
        for item in ['-0.4,0.7', '-0.5,0.7', '-0.6,0.7', '-0.7,0.3', '-0.7,0.4','-0.7,0.7','-0.5,0.4']:
            test_biased_simulation.helper_biased_simulation_alanine_dipeptide(item.replace(' ',''))
        return

    @staticmethod
    def test_biased_simulation_alanine_dipeptide_with_metadynamics(use_well_tempered=0, biasfactor=-1):
        autoencoder_pkl_file = '../tests/dependency/test_biased_simulation/network_9.pkl'
        output_folder = 'temp_output_test_biased_simulation'
        a = autoencoder.load_from_pkl_file(autoencoder_pkl_file)
        a.write_expression_script_for_plumed('temp_info.txt', mode='ANN')
        subprocess.check_output(
'python ../src/biased_simulation.py 50 50000 0 %s temp_info.txt pc_0,0 --MTD_pace 100 --platform CPU --bias_method MTD --MTD_biasfactor %f --MTD_WT %d --equilibration_steps 0'
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
        im = ax.scatter(PCs[:, 0], PCs[:, 1], c=list(range(PCs.shape[0])), cmap='gist_rainbow', s=4)
        fig.colorbar(im, ax=ax)

        out_data = np.loadtxt('temp_MTD_out.txt')

        ax = axes[1]
        im = ax.scatter(out_data[:, 1], out_data[:, 2], c=list(range(out_data.shape[0])), cmap='gist_rainbow', s=4)
        ax.set_xlabel('CV1')
        ax.set_ylabel('CV2')
        ax.set_title('CV data generated by PLUMED')
        fig.colorbar(im, ax=ax)

        ax = axes[2]
        dihedrals = Alanine_dipeptide.get_many_dihedrals_from_cossin(
            Alanine_dipeptide.get_many_cossin_from_coordinates(data))
        dihedrals = np.array(dihedrals)
        im = ax.scatter(dihedrals[:, 1], dihedrals[:, 2], c=list(range(len(dihedrals))), cmap="gist_rainbow", s=4)
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
        

class test_Helper_func(object):
    @staticmethod
    def test_compute_distances_min_image_convention():
        output_pdb = 'out_for_computing_distances.pdb'
        subprocess.check_output(['python', '../src/biased_simulation_general.py', 'Trp_cage', '50', '1000', '0', 'temp_out_12345',
                             'none', 'pc_0,0', 'explicit', 'NPT', '--platform', 'CUDA', '--device', '0', '--output_pdb', output_pdb])
        import mdtraj as md
        box_length = 4.5  # in nm
        temp_t = md.load(output_pdb)
        temp_t.unitcell_lengths, temp_t.unitcell_angles = box_length * np.ones((20, 3)), 90 * np.ones((20, 3))
        temp_u = Universe(output_pdb)
        a_sel = temp_u.select_atoms('name N')
        b_sel = temp_u.select_atoms('name O and resname HOH')
        absolute_index = b_sel.atoms.indices[30]
        b_positions = np.array([b_sel.positions for _ in temp_u.trajectory])
        b_positions = b_positions.reshape(20, b_positions.shape[1] * b_positions.shape[2])
        a_positions = np.array([a_sel.positions for _ in temp_u.trajectory])
        a_positions = a_positions.reshape(20, a_positions.shape[1] * a_positions.shape[2])
        result = Helper_func.compute_distances_min_image_convention(a_positions, b_positions, 10 * box_length)
        assert_almost_equal(md.compute_distances(temp_t, [[0, absolute_index]]).flatten(), result[:, 0, 30] / 10, decimal=4)
        subprocess.check_output(['rm', '-rf', output_pdb, 'temp_out_12345'])
        return

    @staticmethod
    def test_shuffle_multiple_arrays():
        a = np.random.rand(10, 2)
        b1, b2 =Helper_func.shuffle_multiple_arrays([a[:, 0], a[:, 1]])
        for item in range(10):
            assert( [b1[item], b2[item]] in a)
        return

    @staticmethod
    def test_attempt_to_save_npy():
        import shutil
        def get_num_files_in_folder(temp_folder): return len(os.listdir(temp_folder))
        a = 2 * np.eye(3, 3)
        folder = 'temp_test_save_npy'
        if os.path.exists(folder): shutil.rmtree(folder)
        os.mkdir(folder)
        filename = folder + '/1.npy'
        Helper_func.attempt_to_save_npy(filename, a)
        assert (os.path.isfile(filename))
        assert (get_num_files_in_folder(folder) == 1)
        Helper_func.attempt_to_save_npy(filename, a)
        assert (get_num_files_in_folder(folder) == 1)
        for item in range(10):
            Helper_func.attempt_to_save_npy(filename, a+item)
            assert (get_num_files_in_folder(folder) == item + 1)
        shutil.rmtree(folder)
        return


class test_others(object):
    @staticmethod
    def test_SphSh_INDUS_PLUMED_plugin():
        # test for original version of SphSh_INDUS_PLUMED_plugin provided by Prof. Amish Patel
        num_frames = 20
        potential_center = np.random.random(size=3) * 3
        with open('temp_plumed.txt', 'w') as my_f:
            my_f.write('''
SPHSH ATOMS=306-11390:4 XCEN=%f YCEN=%f ZCEN=%f RLOW=-0.5 RHIGH=0.311 SIGMA=0.01 CUTOFF=0.02 LABEL=sph
SPHSH ATOMS=306-11390:4 XCEN=%f YCEN=%f ZCEN=%f RLOW=0.05 RHIGH=0.311 SIGMA=0.01 CUTOFF=0.02 LABEL=sph_2
RESTRAINT ARG=sph.Ntw AT=5 KAPPA=5 SLOPE=0 LABEL=mypotential
PRINT STRIDE=50 ARG=sph.N,sph.Ntw,sph_2.N,sph_2.Ntw FILE=NDATA''' \
    % (potential_center[0], potential_center[1], potential_center[2], 
       potential_center[0], potential_center[1], potential_center[2]))  # since there is "TER" separating solute and solvent in pdb file, so index should start with 306, not 307
        subprocess.check_output(['python', '../src/biased_simulation_general.py', 'Trp_cage', '50', '1000', '0',
                                 'temp_plumed', 'none', 'pc_0,0', 'explicit', 'NPT', '--platform', 'CUDA',
                                 '--bias_method', 'plumed_other', '--plumed_file', 'temp_plumed.txt'])
        out_pdb = 'temp_plumed/output_fc_0.0_pc_[0.0,0.0]_T_300_explicit_NPT.pdb'
        temp_u = Universe(out_pdb)
        reporter_file = out_pdb.replace('output', 'report').replace('.pdb', '.txt')
        box_length_list = Helper_func.get_box_length_list_fom_reporter_file(reporter_file, unit='A')
        O_sel = temp_u.select_atoms('name O and resname HOH')
        O_coords = np.array([O_sel.positions for _ in temp_u.trajectory]).reshape(num_frames, 2772 * 3)
        distances = Helper_func.compute_distances_min_image_convention(
            10 * np.array([potential_center for _ in range(num_frames)]), O_coords, box_length_list)
        coarse_count, actual_count = Helper_func.get_cg_count_in_sphere(distances, 3.11, 0.2, .1)
        plumed_count = np.loadtxt('NDATA')
        assert_almost_equal(plumed_count[-num_frames:, 1], actual_count.flatten())
        assert_almost_equal(plumed_count[-num_frames:, 2], coarse_count.flatten(), decimal=2)
        coarse_count_1, actual_count_1 = Helper_func.get_cg_count_in_shell(distances, 0.5, 3.11, 0.2, .1)
        assert_almost_equal(plumed_count[-num_frames:, 3], actual_count_1.flatten())
        assert_almost_equal(plumed_count[-num_frames:, 4], coarse_count_1.flatten(), decimal=2)
        subprocess.check_output(['rm', '-rf', 'temp_plumed', 'NDATA', 'temp_plumed.txt'])
        return

    @staticmethod
    def test_SphShMod_INDUS_PLUMED_plugin():
        num_frames = 20
        with open('temp_plumed.txt', 'w') as my_f:
            my_f.write('''
SPHSHMOD ATOMS=306-11390:4 ATOMREF=1 RLOW=-0.5 RHIGH=0.311 SIGMA=0.01 CUTOFF=0.02 LABEL=sph
RESTRAINT ARG=sph.Ntw AT=10 KAPPA=5 SLOPE=0 LABEL=mypotential
PRINT STRIDE=50 ARG=sph.N,sph.Ntw FILE=NDATA''' )
        subprocess.check_output(['python', '../src/biased_simulation_general.py', 'Trp_cage', '50', '1000', '0',
                                 'temp_plumed', 'none', 'pc_0,0', 'explicit', 'NPT', '--platform', 'CUDA',
                                 '--bias_method', 'plumed_other', '--plumed_file', 'temp_plumed.txt'])
        out_pdb = 'temp_plumed/output_fc_0.0_pc_[0.0,0.0]_T_300_explicit_NPT.pdb'
        temp_u = Universe(out_pdb)
        reporter_file = out_pdb.replace('output', 'report').replace('.pdb', '.txt')
        box_length_list = Helper_func.get_box_length_list_fom_reporter_file(reporter_file, unit='A')
        print(box_length_list)
        O_sel = temp_u.select_atoms('name O and resname HOH')
        N_sel = temp_u.select_atoms('resnum 1 and name N')
        O_coords = np.array([O_sel.positions for _ in temp_u.trajectory]).reshape(num_frames, 2772 * 3)
        N_coords = np.array([N_sel.positions for _ in temp_u.trajectory]).reshape(num_frames, 3)
        distances = Helper_func.compute_distances_min_image_convention(N_coords, O_coords, box_length_list)
        coarse_count, actual_count = Helper_func.get_cg_count_in_sphere(distances, 3.11, 0.2, .1)
        plumed_count = np.loadtxt('NDATA')
        assert_almost_equal(plumed_count[-num_frames:, 1], actual_count.flatten())
        assert_almost_equal(plumed_count[-num_frames:, 2], coarse_count.flatten(), decimal=2)
        subprocess.check_output(['rm', '-rf', 'temp_plumed', 'NDATA', 'temp_plumed.txt'])
        return
