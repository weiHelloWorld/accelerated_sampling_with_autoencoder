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

'''This is the configuration file for all Python code in this directory,
it configures all default values/global parameters for constructors/functions
'''

#######################################################################
############   some global variables and helper functions  ############
#######################################################################

layer_type_to_name_mapping = {TanhLayer: "Tanh", CircularLayer: "Circular", LinearLayer: "Linear", ReluLayer: "Relu"}
CONFIG_30 = "Alanine_dipeptide"     # the type of molecule we are studying, Alanine_dipeptide, or Trp_cage
WARNING_INFO = "Comment out this line to continue."

def get_mol_param(parameter_list, molecule_name=CONFIG_30):   # get molecule specific parameter using a parameter list
    if molecule_name == "Alanine_dipeptide": return parameter_list[0]
    elif molecule_name == "Trp_cage": return parameter_list[1]
    else: raise Exception("molecule not defined!")

#######################################################################
############   config for ANN_simulation.py  ##########################
#######################################################################

CONFIG_45 = 'keras'                         # training backend: "pybrain", "keras"
CONFIG_48 = 'Cartesian'       # input data type, could be 'cossin' or 'Cartesian'
CONFIG_52 = 16                # number of copies we generate for data augmentation

if CONFIG_48 == 'Cartesian':
    CONFIG_49 = get_mol_param([5.0, 20.0]) # scaling factor for output for Cartesian coordinates

'''class coordinates_data_files_list:'''

CONFIG_1 = ['../target/' + CONFIG_30] # list of directories that contains all coordinates files

'''class autoencoder:'''
CONFIG_57 = [
    [2,5,7,9,15,17,19],
    [1, 2, 3, 17, 18, 19, 36, 37, 38, 57, 58, 59, 76, 77, 78, 93, 94, 95,
    117, 118, 119, 136, 137, 138, 158, 159, 160, 170, 171, 172, 177, 178, 179, 184,
    185, 186, 198, 199, 200, 209, 210, 211, 220, 221, 222, 227, 228, 229, 251, 252,
    253, 265, 266, 267, 279, 280, 281, 293, 294, 295]
]                                          # index of atoms for training and biased simulations
CONFIG_17 = [TanhLayer, TanhLayer, TanhLayer]  # types of hidden layers
CONFIG_2 = 1     # training data interval
if CONFIG_45 == 'pybrain':
    CONFIG_4 = [0.002, 0.4, 0.1, 1]  # network parameters, includes [learningrate,momentum, weightdecay, lrdecay]
    raise Exception("Warning: PyBrain is no longer supported!  " + WARNING_INFO)
elif CONFIG_45 == 'keras':
    CONFIG_4 = get_mol_param([
        [.5, 0.5, 0, True, [0.00, 0.0000, 0.00, 0.00]],
        [0.3, 0.9, 0, True, [0.00, 0.0000, 0.00, 0.00]]
        ])   # [learning rates, momentum, learning rate decay, nesterov, regularization coeff]
else:
    raise Exception('training backend not implemented')

CONFIG_5 = 200                   # max number of training steps
CONFIG_6 = None                # filename to save this network
CONFIG_36 = 2                  #   dimensionality
if CONFIG_17[1] == CircularLayer:
    CONFIG_37 = 2 * CONFIG_36              # number of nodes in bottleneck layer
elif CONFIG_17[1] == TanhLayer or CONFIG_17[1] == ReluLayer:
    CONFIG_37 = CONFIG_36
else:
    raise Exception('Layer not defined')

CONFIG_55 = get_mol_param([2,2])       # number of reference configurations used in training

if CONFIG_48 == 'cossin':
    CONFIG_3 = get_mol_param([
         [8, 15, CONFIG_37, 15, 8],  # the structure of ANN: number of nodes in each layer
         [76, 50, CONFIG_37, 50, 76]
        ])
    raise Exception("Warning: it is not a good idea to use cossin as inputs!  " + WARNING_INFO)
elif CONFIG_48 == 'Cartesian':
    CONFIG_3 = get_mol_param([
         [3 * len(CONFIG_57[0]), 40, CONFIG_37, 40, 3 * len(CONFIG_57[0]) * CONFIG_55],  # the structure of ANN: number of nodes in each layer
         [3 * len(CONFIG_57[1]), 50, CONFIG_37, 50, 3 * len(CONFIG_57[1]) * CONFIG_55]
         ])
else:
    raise Exception('error input data type')

CONFIG_40 = 'explicit'                  # whether to include water molecules, option: explicit, implicit, water_already_included, no_water
CONFIG_51 = 'NPT'                  # simulation ensemble type (for Trp-cage only)
CONFIG_42 = False                             # whether to enable force constant adjustable mode
CONFIG_44 = False                             # whether to use hierarchical autoencoder
if CONFIG_44:
    raise Exception("Warning: no longer supported (used for backward compatibility)!  " + WARNING_INFO)
CONFIG_46 = False                             # whether to enable verbose mode (print training status)
CONFIG_47 = False                        # whether to set the output layer as circular layer
if CONFIG_47:
    raise Exception("Warning: this is a bad choice!  " + WARNING_INFO)

'''class iteration'''

'''def train_network_and_save'''

CONFIG_13 = get_mol_param([5,3])  # num of network trainings we are going to run, and pick the one with least FVE from them
CONFIG_43 = False    # whether we need to parallelize training part, not recommended for single-core computers
if CONFIG_43:
    raise Exception("Warning: parallelization of training is not well tested!  " + WARNING_INFO)

'''def prepare_simulation'''
CONFIG_24 = 'local'  # machine to run the simulations
if CONFIG_24 == "cluster":
    raise Exception("Warning: it has not been tested on the cluster for relatively long time, not recommended!  " + WARNING_INFO)
CONFIG_31 = 10        # maximum number of failed simulations allowed in each iteration

'''def run_simulation'''

CONFIG_56 = get_mol_param([20, 8])    # number of biased simulations running in parallel
CONFIG_14 = 7  # max number of jobs submitted each time
CONFIG_15 = 1  # num of running jobs when the program is allowed to stop
CONFIG_29 = True  if CONFIG_40 == 'explicit' else False   # whether we need to remove the water molecules from pdb files
CONFIG_50 = False   # whether we need to preserve original file if water molecules are removed


##########################################################################
############   config for molecule_spec_sutils.py  #######################
##########################################################################

'''class Sutils'''

CONFIG_10 = get_mol_param([10,10])   # num of bins for get_boundary_points()
CONFIG_11 = get_mol_param([15,20])  # num of boundary points

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
    raise Exception("Warning: very few tests are done for ReLu layer, this is not recommended!  " + WARNING_INFO)
else:
    raise Exception('Layer not defined')


CONFIG_33 = CONFIG_3[0]   # length of list of cos/sin values, equal to the number of nodes in input layer
CONFIG_12 = '../target/' + CONFIG_30  # folder that contains all pdb files

##########################################################################
############   config for cluster_management.py  #########################
##########################################################################

'''class cluster_management'''

CONFIG_8 = get_mol_param([50000, 200000])                  # num of simulation steps
CONFIG_9 = get_mol_param([3000, 5000])                     # force constant for biased simulations
CONFIG_53 = get_mol_param(['flexible', 'flexible'])          # use fixed/flexible force constants for biased simulation for each iteration
CONFIG_54 = 2.47 * get_mol_param([30.0, 15.0])             # max external potential energy allowed (in k_BT)
CONFIG_16 = get_mol_param([500, 2000])                     # record interval (the frequency of writing system state into the file)
CONFIG_19 = '48:00:00'                                    # max running time for the sge job

##########################################################################
############   config for biased_simulation{,_Trp_cage}.py  ##############
##########################################################################

CONFIG_21 = 300   # simulation temperature
CONFIG_22 = 0.002   # simulation time step, in ps

CONFIG_23 = get_mol_param(['CPU', 'CUDA'])              # simulation platform

temp_home_directory = subprocess.check_output('echo $HOME', shell=True).strip()
if temp_home_directory == "/home/fisiksnju":
    CONFIG_25 = '/home/fisiksnju/.anaconda2/lib/plugins'  # this is the directory where the plugin is installed
elif temp_home_directory == "/home/weichen9":
    CONFIG_25 = '/home/weichen9/.my_softwares/openmm7/lib/plugins'
else:
    raise Exception('unknown user directory: %s' % temp_home_directory)

CONFIG_27 =  map(lambda x: layer_type_to_name_mapping[x], CONFIG_17[:2]) # layer_types for ANN_Force, it should be consistent with autoencoder
CONFIG_28 = "ANN_Force"    # the mode of biased force, it could be either "CustomManyParticleForce" (provided in the package) or "ANN_Force" (I wrote)

CONFIG_32 = 5000           # maximum force constant allowed (for force constant adjustable mode)
CONFIG_34 = 500            # force constant step, the value by which the force constant is increased each time (for force constant adjustable mode)
CONFIG_35 = 0.1            # distance tolerance, max distance allowed between center of data cloud and potential center (for force_constant_adjustable mode)

