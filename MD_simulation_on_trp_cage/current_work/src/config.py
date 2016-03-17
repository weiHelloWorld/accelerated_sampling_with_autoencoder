from pybrain.structure import *
from pybrain.structure.modules.circularlayer import *

'''This is the configuration file for all Python code in this directory,
it configures all default values/global parameters for constructors/functions
'''

#######################################################################
############   config for ANN_simulation.py  ##########################
#######################################################################

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

CONFIG_8 = 10000 # num of simulation steps
CONFIG_9 = 50   # force constant for biased simulations
CONFIG_16 = 50  # record interval (the frequency of writing system state into the file)
CONFIG_19 = '24:00:00'  # max running time for the sge job

'''class sutils'''

CONFIG_10 = 5   # num of bins for get_boundary_points()
CONFIG_11 = 15  # num of boundary points
CONFIG_18 = True  # whether we limit the boundary points to be between [-pi, pi], typically works for circularLayer

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

##########################################################################
############   config for biased_simulation.py  ##########################
##########################################################################

CONFIG_20 = True  # whether the PC space is periodic in [- pi, pi], True for circular network, False for Tanh network, this affect the form of potential function
CONFIG_21 = 300   # simulation temperature
CONFIG_22 = 0.002   # simulation time step, in ps

CONFIG_23 = 'CPU'   # simulation platform