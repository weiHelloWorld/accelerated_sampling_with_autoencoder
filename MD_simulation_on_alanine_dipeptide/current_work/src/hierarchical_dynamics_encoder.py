from autoencoders import autoencoder_torch
from ANN_simulation import machine_independent_run
from helper_func import *
import glob

class HDE(object):
    def __init__(self, data_file, n_iter, lag_time, lr_list, batch_size, epochs, n_models):
        self._data_file = data_file           # training data file
        self._n_iter = n_iter                 # number of iterations (i.e. number of slow modes to learn)
        self._lag_time = lag_time
        self._lr_list = lr_list               # learning rates to use in each iteration, and then pick the best one
        self._batch_size = batch_size
        self._epochs = epochs
        self._n_models = n_models             # number of models per learning rate
        return

    def run_one_iter(self, iter_index, save_CV_for_best_model=True,
                     previous_CVs=None       # previous CVs to be orthogonal to
                     ):
        folder_for_model = "../target/iter_%02d/" % iter_index
        commands = ["python ../src/train_network_and_save_for_iter.py 1447 --num_of_trainings 1 " \
                  "--in_data %s --output_file %s/batch_%d_lagtime_%d_lr_%s_index_%02d.pkl " \
                  "--auto_scale 1 --lr_m %s,0 --node_num 0,100,100,1,100,100,0 --lag_time %d " \
                  "--rec_loss_type 2 --autocorr_weight 1.0 --save_to_data_files data.npy,data.npy " \
                  "--batch_size %d" % (
            self._data_file, folder_for_model, self._batch_size, self._lag_time, item_lr, item_index, item_lr,
            self._lag_time, self._batch_size) for item_lr in self._lr_list for item_index in range(self._n_models)]
        if not previous_CVs is None:
            commands = [item + ' --previous_CVs %s' % previous_CVs for item in commands]
        machine_independent_run.run_commands(machine_to_run_simulations='local',
                                             commands=commands, cuda=True, max_num_failed_jobs=1)
        model_pkls = glob.glob("%s/*.pkl" % folder_for_model)
        best_CV, max_autocorr = None, 0
        for item in model_pkls:
            model = autoencoder_torch.load_from_pkl_file(item)
            CVs = model.get_PCs().flatten()
            if not previous_CVs is None:
                for item_previous in previous_CVs.strip().split(','):   # remove contribution of previous CVs (gram-schmidt)
                    item_previous_CV = np.load(item_previous).flatten()
                    CVs -= (np.mean(item_previous_CV * CVs) * item_previous_CV / np.mean(
                        item_previous_CV * item_previous_CV)).flatten()
                    assert (np.mean(CVs * item_previous_CV) < 1.0e-5), np.mean(CVs * item_previous_CV)
            autocorr_current = Helper_func.get_autocorr(CVs.flatten(), self._lag_time)
            if not np.isnan(autocorr_current) and autocorr_current > max_autocorr:
                max_autocorr = autocorr_current
                best_CV = CVs
        if save_CV_for_best_model:
            best_CV -= best_CV.mean()
            best_CV /= np.std(best_CV)
            np.save('CV%02d' % iter_index, best_CV.reshape(best_CV.shape[0], 1))
        return commands, best_CV

    def run_many_iters(self):
        for item in range(1, self._n_iter + 1):
            if item > 1:
                previous_CVs = ['CV%02d.npy' % index_CV for index_CV in range(1, item)]
                previous_CVs = ','.join(previous_CVs)
            else:
                previous_CVs = None
            self.run_one_iter(item, previous_CVs=previous_CVs)
        return

if __name__ == "__main__":
    hde = HDE('pairwise_dis.npy', 3, 1000, ['5e-4', '1e-4'], 500, 100, 3)
    # hde.run_one_iter(1)
    hde.run_many_iters()
