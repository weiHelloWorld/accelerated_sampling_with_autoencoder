from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=str, default='1e-3,10', help="learning rate search range")
parser.add_argument('--lr_log_scale', type=int, default=True, help='whether to search lr in log scale')
parser.add_argument("--momentum", type=str, default='0.1,0.99', help="momentum search range")
parser.add_argument("--num_params", type=int, default=30, help="number of sets of parameters to test")
parser.add_argument("--num_each_param", type=int, default=5, help="number of autoencoders for each set of parameters")
parser.add_argument('--in_data', type=str, default=None, help="npy file containing pre-computed input data")
parser.add_argument('--out_data', type=str, default=None, help="npy file containing pre-computed output data")
args = parser.parse_args()

lr_range = [float(item) for item in args.lr.split(',')]
momentum_range = [float(item) for item in args.momentum.split(',')]

params = np.random.uniform(size=(args.num_params, 2))
params[:, 1] = params[:, 1] * (momentum_range[1] - momentum_range[0]) + momentum_range[0]
if args.lr_log_scale:
    params[:, 0] = np.exp(params[:, 0] * (np.log(lr_range[1]) - np.log(lr_range[0])) + np.log(lr_range[0]))
else:
    params[:, 0] = params[:, 0] * (lr_range[1] - lr_range[0]) + lr_range[0]

assert (np.all(lr_range[0] <= params[:, 0]) and np.all(params[:, 0] <= lr_range[1]))
assert (np.all(momentum_range[0] <= params[:, 1]) and np.all(params[:, 1] <= momentum_range[1]))

for index in range(args.num_each_param):
    for each_param in params:
        print("OMP_NUM_THREADS=6 python train_network_and_save_for_iter.py 1447 --num_of_trainings 1 --lr_m %f,%f --output_file temp_%f_%f_%02d.pkl --in_data %s --out_data %s" % (
            each_param[0], each_param[1], each_param[0], each_param[1], index, args.in_data, args.out_data
        ))
