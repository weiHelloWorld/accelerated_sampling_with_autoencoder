from ANN_simulation import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--starting_index", type=int, default=1, help="index of starting iteration")
parser.add_argument("--num_of_iterations", type=int, default=10, help="number of iterations to run")
parser.add_argument("--starting_network_file", type=str, default=None, help="the network to start with")
parser.add_argument("--training_interval", type=int, default=1, help="training interval")
args = parser.parse_args()

if args.starting_network_file is None:
    starting_network = None
else:
    starting_network = Sutils.load_object_from_pkl_file(args.starting_network_file)

init_iter = iteration(index = args.starting_index, network = starting_network)

a = simulation_with_ANN_main(num_of_iterations = args.num_of_iterations, initial_iteration = init_iter, training_interval=args.training_interval)
a.run_mult_iterations()

print("Done main work!")
