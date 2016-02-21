from pybrain.structure import *
from pybrain.structure.modules.circularlayer import *

'''This is the configuration file for all Python code in this directory,
it configures all default values/global parameters for constructors/functions
'''

this_is_used_to_count_how_many_config_options_I_am_using = 17

############   config for ANN_simulation.py  ##########################

'''class coordinates_data_files_list:'''

CONFIG_1 = ['../target/'] # list of directories that contains all coordinates files

'''class neural_network_for_simulation:'''
CONFIG_17 = [TanhLayer, CircularLayer, TanhLayer]  # types of hidden layers
CONFIG_2 = 1 # training data interval
CONFIG_3 = [8, 12, 4, 12, 8]  # the structure of ANN: number of nodes in each layer
CONFIG_4 = [0.002, 0.4, 0.1, 1]  # network parameters, includes [learningrate,momentum, weightdecay, lrdecay]
CONFIG_5 = 50 # max number of training steps
CONFIG_6 = None # filename to save this network

'''class simulation_management'''

CONFIG_8 = 5000 # num of simulation steps
CONFIG_9 = 500  # force constant for biased simulations
CONFIG_16 = 50  # record interval (the frequency of writing system state into the file)

'''class sutils'''

CONFIG_10 = 5   # num of bins for get_boundary_points()
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