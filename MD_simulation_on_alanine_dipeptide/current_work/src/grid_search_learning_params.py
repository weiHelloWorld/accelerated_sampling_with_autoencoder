from ANN_simulation import *

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=str, default='0.1,1.0,0.2', help="learning rate range: start, end (not included), step")
parser.add_argument("--momentum", type=str, default='0.1,1.0,0.2', help="momentum range: start, end (not included), step")
parser.add_argument("--num_each_param", type=int, default=5, help="number of autoencoders for each set of parameters")
parser.add_argument('--in_data', type=str, default=None, help="npy file containing pre-computed input data")
parser.add_argument('--out_data', type=str, default=None, help="npy file containing pre-computed output data")
args = parser.parse_args()

lr_list = [float(item) for item in args.lr.split(',')]
lr_list = np.arange(lr_list[0], lr_list[1], lr_list[2])
momentum_list = [float(item) for item in args.momentum.split(',')]
momentum_list = np.arange(momentum_list[0], momentum_list[1], momentum_list[2])

for index in range(args.num_each_param):
    for lr in lr_list:
        for momentum in momentum_list:
            print "OMP_NUM_THREADS=6 python train_network_and_save_for_iter.py 1447 --num_of_trainings 1 --lr_m %.2f,%.2f --output_file temp_%.2f_%.2f_%d.pkl --in_data %s --out_data %s" % (
                lr, momentum, lr, momentum, index, args.in_data, args.out_data
            )
