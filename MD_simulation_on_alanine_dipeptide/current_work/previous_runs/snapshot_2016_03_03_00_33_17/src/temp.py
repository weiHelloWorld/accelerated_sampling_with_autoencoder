from ANN_simulation import *

sutils.generate_coordinates_from_pdb_files(folder_for_pdb = '../target')

staring_index = 2
init_iter = iteration(index = staring_index, network = None)

init_iter.train_network_and_save()   # train it if it is empty
init_iter.prepare_simulation()

print("Done temp work!")
