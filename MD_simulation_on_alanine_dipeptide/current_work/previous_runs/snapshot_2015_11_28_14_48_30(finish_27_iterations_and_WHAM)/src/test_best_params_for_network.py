import sys

from ANN_simulation import *

file_list = coordinates_data_files_list(['../target/'])._list_of_coor_data_files

myindex = int(sys.argv[1])
learningrate = float(sys.argv[2])
momentum = float(sys.argv[3])
weightdecay = float(sys.argv[4])
lrdecay = float(sys.argv[5])

parameters = [learningrate, momentum, weightdecay, lrdecay]

my_network = neural_network_for_simulation(index = myindex, list_of_coor_data_files=file_list, 
					training_data_interval=2, 
					network_parameters=parameters
					)

my_network.train()
my_network.save_into_file()
