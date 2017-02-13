import copy, pickle, re, os, time, subprocess, datetime, itertools, sys, abc, argparse
from scipy import io as sciio
import numpy as np, pandas as pd, seaborn as sns
from numpy.testing import assert_almost_equal
from math import *
from pybrain.structure import *
from pybrain.structure.modules.circularlayer import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure.connections.shared import MotherConnection,SharedFullConnection
from pybrain.structure.moduleslice import ModuleSlice
import matplotlib.pyplot as plt
from sklearn.neighbors import RadiusNeighborsRegressor
import matplotlib
from Bio import PDB
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from MDAnalysis import Universe
from MDAnalysis.analysis.align import *
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis.distances import distance_array
from keras.models import Sequential
from keras.optimizers import *
from keras.layers import Dense, Activation, Lambda, Reshape
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

'''This is the configuration file for all Python code in this directory,
it configures all default values/global parameters for constructors/functions
'''

#######################################################################
######################   some global variables  #######################
#######################################################################

layer_type_to_name_mapping = {TanhLayer: "Tanh", CircularLayer: "Circular", LinearLayer: "Linear", ReluLayer: "Relu"}

#######################################################################
############   config for ANN_simulation.py  ##########################
#######################################################################

CONFIG_30 = "Alanine_dipeptide"     # the type of molecule we are studying, Alanine_dipeptide, or Trp_cage
CONFIG_45 = 'keras'                         # training backend: "pybrain", "keras"
CONFIG_48 = 'Cartesian'       # input data type, could be 'cossin' or 'Cartesian'

if CONFIG_30 == "Alanine_dipeptide":
    CONFIG_49 = 5.0                # scaling factor for output for Cartesian coordinates
elif CONFIG_30 == "Trp_cage":
    CONFIG_49 = 20.0
else:
    raise Exception('molecule type error')

'''class coordinates_data_files_list:'''

CONFIG_1 = ['../target/' + CONFIG_30] # list of directories that contains all coordinates files

'''class autoencoder:'''
CONFIG_17 = [TanhLayer, TanhLayer, TanhLayer]  # types of hidden layers
CONFIG_2 = 1     # training data interval
if CONFIG_45 == 'pybrain':
    CONFIG_4 = [0.002, 0.4, 0.1, 1]  # network parameters, includes [learningrate,momentum, weightdecay, lrdecay]
elif CONFIG_45 == 'keras':
    CONFIG_4 = [.5, 0.5, 0, True, [0.00, 0.0000, 0.00, 0.00]]      # [learning rates, momentum, learning rate decay, nesterov, regularization coeff], note that the definition of these parameters are different from those in Pybrain
else:
    raise Exception('training backend not implemented')

CONFIG_5 = 200                   # max number of training steps
CONFIG_6 = None # filename to save this network
CONFIG_36 = 2              #   dimensionality
if CONFIG_17[1] == CircularLayer:
    CONFIG_37 = 2 * CONFIG_36              # number of nodes in bottleneck layer
elif CONFIG_17[1] == TanhLayer or CONFIG_17[1] == ReluLayer:
    CONFIG_37 = CONFIG_36
else:
    raise Exception('Layer not defined')

if CONFIG_30 == "Alanine_dipeptide":
    if CONFIG_48 == 'cossin':
        CONFIG_3 = [8, 15, CONFIG_37, 15, 8]  # the structure of ANN: number of nodes in each layer
    elif CONFIG_48 == 'Cartesian':
        CONFIG_3 = [21, 40, CONFIG_37, 40, 21]
elif CONFIG_30 == "Trp_cage":
    if CONFIG_48 == 'cossin':
        CONFIG_3 = [76, 50, CONFIG_37, 50, 76]
    elif CONFIG_48 == 'Cartesian':
        CONFIG_3 = [180, 50, CONFIG_37, 50, 180]
    else:
        raise Exception('error input data type')
else:
    raise Exception('molecule type error')

CONFIG_40 = 'implicit'                  # whether to include water molecules, option: "with_water" or "without_water"
CONFIG_42 = False                             # whether to enable force constant adjustable mode
CONFIG_44 = False                             # whether to use hierarchical autoencoder
CONFIG_46 = False                             # whether to enable verbose mode (print training status)
CONFIG_47 = False                        # whether to set the output layer as circular layer

'''class iteration'''

'''def train_network_and_save'''

CONFIG_13 = 3  # num of network trainings we are going to run, and pick the one with least FVE from them
CONFIG_43 = False    # whether we need to parallelize training part, not recommended for single-core computers

'''def prepare_simulation'''
CONFIG_24 = 'local'  # machine to run the simulations
CONFIG_31 = 10        # maximum number of failed simulations allowed in each iteration

'''def run_simulation'''

CONFIG_14 = 7  # max number of jobs submitted each time
CONFIG_15 = 1  # num of running jobs when the program is allowed to stop
CONFIG_29 = False  # whether we need to remove the water molecules from pdb files


##########################################################################
############   config for molecule_spec_sutils.py  #######################
##########################################################################

'''class Sutils'''

CONFIG_10 = 10   # num of bins for get_boundary_points()
CONFIG_11 = 15  # num of boundary points

CONFIG_39 = False    #  set the range of histogram automatically based on min,max values in each dimension
CONFIG_41 = False    # whether we reverse the order of sorting of diff_with_neighbors values in get_boundary algorithm

if CONFIG_17[1] == CircularLayer:
    CONFIG_18 = True  # whether we limit the boundary points to be between [-pi, pi], typically works for circularLayer
    CONFIG_26 = [[-np.pi, np.pi] for item in range(CONFIG_36)]    # range of PCs, for circular case, it is typically [[-np.pi, np.pi],[-np.pi, np.pi]]
elif CONFIG_17[1] == TanhLayer:
    CONFIG_18 = False
    CONFIG_26 = [[-1, 1] for item in range(CONFIG_36)]
elif CONFIG_17[1] == ReluLayer:
    CONFIG_18 = False
    CONFIG_26 = [[-1, 1] for item in range(CONFIG_36)]   # FIXME: modify this later
else:
    raise Exception('Layer not defined')


CONFIG_33 = CONFIG_3[0]   # length of list of cos/sin values, equal to the number of nodes in input layer
CONFIG_12 = '../target/' + CONFIG_30  # folder that contains all pdb files

##########################################################################
############   config for cluster_management.py  #########################
##########################################################################

'''class cluster_management'''

CONFIG_8 = 5000 # num of simulation steps
CONFIG_9 = 3000   # force constant for biased simulations
CONFIG_16 = 50  # record interval (the frequency of writing system state into the file)
CONFIG_19 = '24:00:00'  # max running time for the sge job

##########################################################################
############   config for biased_simulation{,_Trp_cage}.py  ##############
##########################################################################

CONFIG_21 = 300   # simulation temperature
CONFIG_22 = 0.002   # simulation time step, in ps

CONFIG_23 = 'CPU'   # simulation platform
CONFIG_25 = '/home/fisiksnju/.anaconda2/lib/plugins'  # this is the directory where the plugin is installed
CONFIG_27 =  map(lambda x: layer_type_to_name_mapping[x], CONFIG_17[:2]) # layer_types for ANN_Force, it should be consistent with autoencoder
CONFIG_28 = "ANN_Force"    # the mode of biased force, it could be either "CustomManyParticleForce" (provided in the package) or "ANN_Force" (I wrote)
CONFIG_32 = 5000           # maximum force constant allowed (for force constant adjustable mode)
CONFIG_34 = 500            # force constant step, the value by which the force constant is increased each time (for force constant adjustable mode)
CONFIG_35 = 0.1            # distance tolerance, max distance allowed between center of data cloud and potential center (for force_constant_adjustable mode)

