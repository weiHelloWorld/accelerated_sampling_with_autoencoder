'''This is a test for functionality of ANN_simulation.py
'''

import sys
import math

sys.path.append('../src/')  # add the source file folder

from ANN_simulation_trp_cage import *
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal


class test_coordinates_file(object):


    @staticmethod
    def test_coor_data():
        filename = 'dependency/biased_output_fc_1000_x1_0.7_x2_-1.07_coordinates.txt'
        my_file = single_simulation_coordinates_file(filename)
        expected = np.array([1.875  , 2.791  , -0.215 , 3.338  , 2.897 ,  0.193   ,
            4.032  , 3.862 ,  -0.414 , 5.350  , 4.301  , 0.000   ,
            5.314 ,  5.734  , 0.511  , 5.119  , 6.718  , -0.370  ,
            4.974  , 8.120  , -0.034  ])

        assert_array_almost_equal(my_file._coor_data[0], expected)
        assert_equal(my_file._coor_data.shape[0], 100)
        return

class test_simulation_utils(object):
    @staticmethod
    def test_get_many_cossin_from_coordiantes_in_list_of_files():
        list_of_files = ['dependency/biased_output_fc_1000_x1_0.7_x2_-1.07_coordinates.txt']
        actual = sutils.get_many_cossin_from_coordiantes_in_list_of_files(list_of_files)
        assert_equal(100, len(actual))
        assert_equal(8, len(actual[0]))
        expected = [-0.97750379745637539, -0.40651951247361273, 0.32798972133405019, -0.99736251135112719,
                    0.21091781802011281, -0.91364209949969788, -0.94468129160008874, 0.07258113357734837]

        assert_almost_equal(expected, actual[0])
        return

    @staticmethod
    def test_get_many_dihedrals_from_cossin():
        angle = [.4, -.7, math.pi, -.45]
        cossin = [[1, -1, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, 1, -1],
                  map(cos, angle) + map(sin, angle)]
        actual = sutils.get_many_dihedrals_from_cossin(cossin)
        expected = [[0,0,0,0],[math.pi / 2, -math.pi/2, math.pi/2,-math.pi/2], angle]
        for item in range(len(actual)):
            for index in range(4):
                assert_almost_equal(actual[item][index], expected[item][index], 4)

        return

    @staticmethod
    def test_get_cossin_of_a_dihedral_from_four_atoms():
        coords = [[0,0,0], [0,1,1], [0,1,2], [1,1,3]]
        actual = np.array(sutils.get_cossin_of_a_dihedral_from_four_atoms(coords[0], coords[1], coords[2], coords[3]))
        expected = np.array([0, 1])
        assert_array_almost_equal(expected, actual)
        return

class test_neural_network_for_simulation(object):

    pass
