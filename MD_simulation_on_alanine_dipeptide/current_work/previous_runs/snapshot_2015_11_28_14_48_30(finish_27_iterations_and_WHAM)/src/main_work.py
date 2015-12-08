from ANN_simulation import *

staring_index = 21
init_iter = iteration(index = staring_index, network = None, num_of_simulation_steps = 1000)
a = simulation_with_ANN_main(num_of_iterations = 10, initial_iteration = init_iter, training_interval=3)
a.run_mult_iterations()
print("Done!")