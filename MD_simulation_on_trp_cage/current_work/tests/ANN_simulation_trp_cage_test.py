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
    def test_get_many_dihedrals_from_cossin():
        a_coordinate = [35.411, 5.612,  43.440, 35.693, 4.170,  43.353, 37.163, 3.786,  43.442, 38.096, 4.568,  42.894, 39.509, 4.290,  42.738, 40.305, 3.589,  43.830, 40.264, 3.952,  45.114, 40.983, 3.246,  46.156, 40.703, 1.751,  46.221, 39.485, 1.287,  45.929, 38.942, -0.054, 45.855, 39.604, -0.769, 44.687, 39.864, -0.107, 43.557, 40.529, -0.808, 42.478, 41.947, -1.212, 42.856, 42.686, -0.229, 43.374, 44.015, -0.315, 43.946, 44.175, -1.380, 45.021, 43.157, -1.482, 45.880, 42.941, -2.454, 46.932, 42.763, -3.848, 46.347, 41.911, -3.997, 45.330, 41.821, -5.282, 44.667, 42.908, -5.700, 43.686, 43.892, -4.881, 43.307, 45.167, -5.240, 42.720, 46.149, -5.878, 43.693, 45.883, -5.673, 44.985, 46.652, -6.200, 46.094, 47.203, -5.130, 47.025, 46.837, -3.869, 46.782, 47.314, -2.779, 47.609, 48.708, -2.335, 47.190, 49.636, -2.164, 48.134, 51.033, -1.887, 47.867, 51.772, -2.980, 47.109, 51.523, -4.262, 47.389, 52.391, -5.306, 46.883, 52.237, -5.553, 45.389, 51.264, -5.083, 44.605, 51.143, -5.146, 43.163, 51.969, -4.055, 42.496, 53.004, -3.539, 43.163, 53.965, -2.740, 42.431, 53.556, -1.291, 42.205, 52.706, -0.654, 43.015, 52.198, 0.688,  42.817, 51.765, 1.412,  44.083, 52.033, 2.713,  44.221, 51.517, 3.539,  45.293, 50.038, 3.795,  45.041, 49.318, 4.204,  46.088, 47.891, 4.436,  45.991, 47.467, 5.619,  45.133, 46.425, 5.373,  44.336, 45.660, 6.510,  43.864, 45.021, 7.244,  45.034, 44.862, 8.561,  44.883, 44.272, 9.420,  45.890, 43.281, 10.508, 45.502]
        cossin = sutils.get_cossin_from_a_coordinate(a_coordinate)
        dih = sutils.get_many_dihedrals_from_cossin([cossin])
        assert (len(dih[0]) == 38)
        reconstruct_cossin = reduce(lambda x, y: x + y,
                                    map(lambda x: [cos(x), sin(x)],
                                        dih[0])
                                    )
        assert_array_almost_equal(reconstruct_cossin, cossin)
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
