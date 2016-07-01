'''This is a test for functionality of ANN_simulation.py
'''

import sys
import math

sys.path.append('../src/')  # add the source file folder

from ANN_simulation import *
from nose.tools import assert_equal, assert_almost_equal


class test_molecule_spec_sutils(object):
    @staticmethod
    def test_get_many_cossin_from_coordiantes_in_list_of_files():
        list_of_files = ['dependency/biased_output_fc_1000_x1_0.7_x2_-1.07_coordinates.txt']
        actual = Alanine_dipeptide().get_many_cossin_from_coordiantes_in_list_of_files(list_of_files)
        assert_equal(100, len(actual))
        assert_equal(8, len(actual[0]))
        expected = [-0.97750379745637539, 0.21091781802011281, -0.40651951247361273, -0.91364209949969788,
                    0.32798972133405019, -0.94468129160008874,  -0.99736251135112719, 0.07258113357734837]
        for item in range(8):
            assert_almost_equal(expected[item], actual[0][item])
        return

    @staticmethod
    def test_get_many_dihedrals_from_cossin():
        angle = [.4, -.7, math.pi, -.45]
        cossin = [[1, 0, -1, 0, 1, 0, -1, 0], [0, 1, 0, -1, 0, 1, 0, -1],
                  reduce(lambda x,y: x+y, map(lambda x: [cos(x), sin(x)], angle))
                  ]
        actual = Alanine_dipeptide().get_many_dihedrals_from_cossin(cossin)
        expected = [[0,0,0,0],[math.pi / 2, -math.pi/2, math.pi/2,-math.pi/2], angle]
        for item in range(len(actual)):
            for index in range(4):
                assert_almost_equal(actual[item][index], expected[item][index], 4)
        return



