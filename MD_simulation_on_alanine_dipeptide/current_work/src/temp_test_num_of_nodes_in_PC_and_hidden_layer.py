from ANN_simulation import *
import sys

my_file_list = coordinates_data_files_list(['../target/'])._list_of_coor_data_files
# my_file_list = filter(lambda x: x.find('network_27') == -1, my_file_list)

num_of_hidden_layer_nodes = int(sys.argv[1])
num_of_PC_nodes = int(sys.argv[2])

for item in range(0, 5):
	current_network = neural_network_for_simulation(index = item, 
				list_of_coor_data_files = my_file_list,
								training_data_interval = 2,
							  node_num = [8, num_of_hidden_layer_nodes, num_of_PC_nodes, num_of_hidden_layer_nodes, 8], 
							  network_parameters = None, max_num_of_training = 100)
	current_network.train() 
	current_network.save_into_file('../resources/network_numPC_%d_numHiddenLayerNode_%d_index_%d.pkl' \
									% (num_of_PC_nodes, num_of_hidden_layer_nodes, item) )
