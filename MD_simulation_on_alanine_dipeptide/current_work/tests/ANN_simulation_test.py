'''This is a test for functionality of ANN_simulation.py
'''

import sys, os, math, subprocess, matplotlib.pyplot as plt

sys.path.append('../src/')  # add the source file folder

from ANN_simulation import *
from numpy.testing import assert_almost_equal, assert_equal


class test_Sutils(object):
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

        sns.heatmap(hist_matrix, ax=axes[0][0], annot=True, cbar=False)
        sns.heatmap(hist_matrix_processed, ax=axes[0][1], annot=True, cbar=False)
        sns.heatmap(diff_with_neighbors, ax=axes[1][0], annot=True, cbar=False)
        sns.heatmap(diff_with_neighbors < 0, ax=axes[1][1], annot=False, cbar=False)
        axes[0][0].set_title('histogram of number of data points')
        axes[0][1].set_title('histogram after preprocessing using function $p_i = \exp{(-n_i)}$')
        axes[1][0].set_title('difference of $p_i$ value of a grid with its neighbors')
        axes[1][1].set_title('location of new potential centers')
        fig.savefig('diagram_of_finding_boundary.pdf', format='pdf', bbox_inches='tight')
        return


class test_Alanine_dipeptide(object):
    @staticmethod
    def test_get_many_cossin_from_coordiantes_in_list_of_files():
        list_of_files = ['dependency/biased_output_fc_1000_x1_0.7_x2_-1.07_coordinates.txt']
        actual = Alanine_dipeptide().get_many_cossin_from_coordiantes_in_list_of_files(list_of_files)
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
        pdb_file_list = ['dependency/1l2y.pdb']
        a = Trp_cage.get_pairwise_distance_matrices_of_alpha_carbon(pdb_file_list)
        a = [item.reshape(400, 1) for item in a]
        b = Trp_cage.get_non_repeated_pairwise_distance_as_list_of_alpha_carbon(pdb_file_list)
        assert (len(a) == len(b))
        for _1 in range(len(b)):
            for _2 in b[_1]:
                assert (_2 in a[_1])
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


class test_neural_network_for_simulation(object):
    @staticmethod
    def test_get_mid_result():
        autoencoder_pkl_path = 'dependency/a_network_pkl_and_coef_file/network_1.pkl'
        coef_file = 'dependency/a_network_pkl_and_coef_file/autoencoder_info_1.txt'
        a = Sutils.load_object_from_pkl_file(autoencoder_pkl_path)
        assert isinstance(a, neural_network_for_simulation)
        mid_result =  a.get_mid_result()
        for _1 in range(4):
            assert_almost_equal (np.loadtxt('dependency/out_mid_result/output_mid_result_%d.txt' % _1), [item[_1] for item in mid_result])
        return

