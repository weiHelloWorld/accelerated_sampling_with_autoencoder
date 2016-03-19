import matplotlib
matplotlib.use('Agg') 

from ANN_simulation_trp_cage import *

import seaborn as sb
import pandas as pd
import sys

temperature = int(sys.argv[1])
num_of_PC = int(sys.argv[2])
max_num_of_training = int(sys.argv[3])
# In[24]:

my_file_list = coordinates_data_files_list(['../target/unbiased_%dK_output_coordinates.txt' % temperature])._list_of_coor_data_files
data = sutils.get_many_cossin_from_coordiantes_in_list_of_files(my_file_list)

a = neural_network_for_simulation(index=1447, 
                                  training_data_interval = 1,
                                  data_set_for_training=data,
                                  node_num = [76, 152, num_of_PC, 152, 76],
                                  max_num_of_training = max_num_of_training,
                                  network_verbose = False,
                                  network_parameters = [0.002, 0.4, 0.1, 1]
                                 )
a.train()
a.save_into_file('network_1447.pkl')

FVE = a.get_fraction_of_variance_explained()

print(FVE)



result = [item[3] for item in a.get_mid_result()]


plt.figure(figsize=(20, 15))

plt.subplot(2,1,1)
result = np.array(result)

result_with_mean_removed = result - np.tile(result.mean(axis=0), (result.shape[0], 1))

df = pd.DataFrame(result_with_mean_removed)

sb.boxplot(df)
plt.title('output of autoencoder, FVE = %f' % FVE)
plt.xlabel('index')
plt.ylabel('value (with mean removed)')

plt.subplot(2,1,2)
data = np.array(data)
data_with_mean_removed = data - np.tile(data.mean(axis=0), (data.shape[0], 1))

df = pd.DataFrame(data_with_mean_removed)

sb.boxplot(df)

plt.title('input of autoencoder')
plt.xlabel('index')
plt.ylabel('value (with mean removed)')

plt.savefig('%dK_numPC_%d.png' % (temperature, num_of_PC))




