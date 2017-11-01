from ANN_simulation import *

parser = argparse.ArgumentParser()
parser.add_argument("iter_index_range", type=str,
                    help="range of iteration index, format='begin_iter,end_iter' where end_iter is not included")
parser.add_argument("max_num_CVs", type=int, help="max number of CVs for L method analysis")
parser.add_argument("--num_autoencoders", type=int, default=5,
                    help="number of autoencoders to train for each (iter_index, num_CVs) pair")
parser.add_argument('--output_folder', type=str, default='temp_L_method', help="output folder containing models")
parser.add_argument("--circular", type=int, default=0, help='is circular autoencoder?')
args = parser.parse_args()

start_index, end_index = [int(item) for item in args.iter_index_range.strip().split(',')]
if not os.path.exists(args.output_folder):
    subprocess.check_output(['mkdir', args.output_folder])

for iter_index in range(start_index, end_index):
    for num_CVs in range(1, args.max_num_CVs + 1):
        for item in range(args.num_autoencoders):
            pkl_file = args.output_folder + "/temp_iter_%02d_CV_%02d_index_%02d.pkl" % (iter_index, num_CVs, item)
            num_nodes_per_CV = 2 if args.circular else 1
            command = "python train_network_and_save_for_iter.py 1447 --num_of_trainings 1 --num_PCs %d " % (
                num_nodes_per_CV * num_CVs) + "--output_file %s --data_folder up_to_iter_%d" % (pkl_file, iter_index)
            print command
