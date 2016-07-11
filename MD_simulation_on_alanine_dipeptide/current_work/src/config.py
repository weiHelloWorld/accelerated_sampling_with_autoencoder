from pybrain.structure import *
from pybrain.structure.modules.circularlayer import *
import numpy as np

'''This is the configuration file for all Python code in this directory,
it configures all default values/global parameters for constructors/functions
'''

#######################################################################
######################   some global variables  #######################
#######################################################################

layer_type_to_name_mapping = {TanhLayer: "Tanh", CircularLayer: "Circular", LinearLayer: "Linear"}

#######################################################################
############   config for ANN_simulation.py  ##########################
#######################################################################

CONFIG_30 = "Trp_cage"     # the type of molecule we are studying, Alanine_dipeptide, or Trp_cage

'''class coordinates_data_files_list:'''

CONFIG_1 = ['../target/' + CONFIG_30] # list of directories that contains all coordinates files

'''class neural_network_for_simulation:'''
CONFIG_17 = [TanhLayer, TanhLayer, TanhLayer]  # types of hidden layers
CONFIG_2 = 2     # training data interval
CONFIG_4 = [0.002, 0.4, 0.1, 1]  # network parameters, includes [learningrate,momentum, weightdecay, lrdecay]
CONFIG_5 = 50 # max number of training steps
CONFIG_6 = None # filename to save this network
if CONFIG_30 == "Alanine_dipeptide":
    CONFIG_3 = [8, 15, 2, 15, 8]  # the structure of ANN: number of nodes in each layer
elif CONFIG_30 == "Trp_cage":
    CONFIG_3 = [76, 10, 2, 10, 76]
else:
    raise Exception('molecule type error')

'''class iteration'''

'''def train_network_and_save'''

CONFIG_13 = 5  # num of network trainings we are going to run, and pick the one with least FVE from them

'''def prepare_simulation'''
CONFIG_24 = 'local'  # machine to run the simulations

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
CONFIG_18 = False  # whether we limit the boundary points to be between [-pi, pi], typically works for circularLayer
CONFIG_26 = [[-1, 1],[-1, 1]]    # range of PCs, for circular case, it is typically [[-np.pi, np.pi],[-np.pi, np.pi]]
CONFIG_25 = CONFIG_3[0]   # length of list of cos/sin values, equal to the number of nodes in input layer
CONFIG_12 = '../target/' + CONFIG_30  # folder that contains all pdb files

##########################################################################
############   config for cluster_management.py  #########################
##########################################################################

'''class cluster_management'''

CONFIG_8 = 5000 # num of simulation steps
CONFIG_9 = 5000   # force constant for biased simulations
CONFIG_16 = 50  # record interval (the frequency of writing system state into the file)
CONFIG_19 = '24:00:00'  # max running time for the sge job

##########################################################################
############   config for biased_simulation.py  ##########################
##########################################################################

CONFIG_20 = False  # whether the PC space is periodic in [-pi, pi], True for circular network, False for Tanh network, this affect the form of potential function
CONFIG_21 = 300   # simulation temperature
CONFIG_22 = 0.002   # simulation time step, in ps

CONFIG_23 = 'Reference'   # simulation platform
CONFIG_25 = '/usr/local/openmm/lib/plugins'  # this is the directory where the plugin is installed
CONFIG_27 =  map(lambda x: layer_type_to_name_mapping[x], CONFIG_17[:2]) # layer_types for ANN_Force, it should be consistent with autoencoder
CONFIG_28 = "ANN_Force"    # the mode of biased force, it could be either "CustomManyParticleForce" (provided in the package) or "ANN_Force" (I wrote)

