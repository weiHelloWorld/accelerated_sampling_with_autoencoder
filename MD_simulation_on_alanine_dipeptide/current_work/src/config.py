import copy, pickle, re, os, time, subprocess, datetime, itertools
from scipy import io as sciio
import numpy as np
from math import *
from pybrain.structure import *
from pybrain.structure.modules.circularlayer import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
import matplotlib.pyplot as plt

'''This is the configuration file for all Python code in this directory,
it configures all default values/global parameters for constructors/functions
'''

############################################################################

'''config for ANN_simulation.py'''

'''class coordinates_data_files:'''

'''def __init__'''

CONFIG_1 = ['../target/'] # list of directories that contains all coordinates files

'''class neural_network_for_simulation:'''

'''def __init__'''

CONFIG_2 = 1, # training data interval
CONFIG_3 = [8, 12, 2, 12, 8],  # the structure of ANN: number of nodes in each layer
CONFIG_4 = [0.002, 0.4, 0.1, 1],  # network parameters, includes [learningrate,momentum, weightdecay, lrdecay]
CONFIG_5 = 100, # max number of training steps

'''def save_into_file'''

CONFIG_6 = None # filename to save this network

'''def train'''

CONFIG_7 = [TanhLayer, CircularLayer, TanhLayer] 
# hidden layers of this network

'''class simulation_management'''

'''def __init__'''

'''def create_sge_files_for_simulation'''

CONFIG_8 = 2000, # num of simulation steps
CONFIG_9 = 500  # force constant for biased simulations

'''def get_boundary_points'''
'''def get_boundary_points_2'''

CONFIG_10 = 5   # num of bins
CONFIG_11 = 10  # num of boundary points

'''def generate_coordinates_from_pdb_files'''

CONFIG_12 = '../target'  # folder that contains all pdb files

'''class iteration'''

'''def train_network_and_save'''

CONFIG_13 = 5  # num of network trainings we are going to run, and pick the one with least FVE from them

'''def run_simulation'''

CONFIG_14 = 7  # max number of jobs submitted each time
CONFIG_15 = 1  # num of running jobs when the program is allowed to stop

'''def get_plottings'''
'''class simulation_with_ANN_main'''
'''def __init__'''
'''def run_one_iteration'''
'''def run_mult_iterations'''
'''class single_biased_simulation_data'''
'''def __init__'''
'''def get_center_of_data_cloud_in_this_biased_simulation'''
'''def get_offset_between_potential_center_and_data_cloud_center'''