'''This is a test for functionality of ANN_simulation.py
'''

import sys

sys.path.append('../src/')  # add the source file folder

from ANN_simulation import *
from nose.tools import assert_equal


class test_coordinates_file(object):
    def test_coor_data(self):
        filename = 'dependency/biased_output_fc_1000_x1_0.7_x2_-1.07_coordinates.txt'
        my_file = coordinates_file(filename)
        expected = np.array([1.875  , 2.791  , -0.215 , 3.338  , 2.897 ,  0.193   ,
            4.032  , 3.862 ,  -0.414 , 5.350  , 4.301  , 0.000   ,
            5.314 ,  5.734  , 0.511  , 5.119  , 6.718  , -0.370  ,
            4.974  , 8.120  , -0.034  ])
        for item in range(21):
            assert_equal(my_file._coor_data[0][item], expected[item])

        assert_equal(my_file._coor_data.shape[0], 100)
        return


class test_ANN_simulation(object):
    pass
