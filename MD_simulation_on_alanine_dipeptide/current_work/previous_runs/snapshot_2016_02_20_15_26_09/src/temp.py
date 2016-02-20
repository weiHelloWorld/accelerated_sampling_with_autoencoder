# try training on the complete dialanine dataset

from ANN_simulation import *

def train_on_complete_dataset_using_tanh(index):
    complete_raw_data = np.loadtxt('complete_data.txt')
    complete_angle_data = complete_raw_data / 180 * np.pi

    complete_data = np.concatenate((np.cos(complete_angle_data).T, np.sin(complete_angle_data).T)).T

    a = neural_network_for_simulation(index=1, 
                                      training_data_interval = 5,
                                      data_set_for_training= complete_data,
                                      node_num = [8, 12, 2, 12, 8],   
                                      hidden_layers_types=[TanhLayer, TanhLayer, TanhLayer],
                                      max_num_of_training=30
                                     )
    a.train()
    a.save_into_file('complete_dataset_%d.pkl' % index)

    # a = pickle.load(open('complete_dataset_%d.pkl' % index,'rb'))
    # b = plotting(a)
    # temp_fig, temp_ax_1, _ = b.plotting_with_coloring_option(plotting_space = "PC",
    #                                 color_option = 'phi'
    #                                )

    # temp_fig, temp_ax_2, _ = b.plotting_with_coloring_option(plotting_space = "PC",
    #                                 color_option = 'psi'
    #                                )
    # return (temp_ax_1, temp_ax_2)

from multiprocessing import Pool

num = 5

if __name__ == '__main__':
    p = Pool(num)
    p.map(train_on_complete_dataset_using_tanh, range(num))