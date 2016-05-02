from ANN_simulation import *
import sys

version = sys.argv[1]

directory_containing_coor_files = '../target/wham/'

a=pickle.load(open('../resources/network_1447.pkl','rb'))

if version == "Bayes":
	a.generate_files_for_Bayes_WHAM(directory_containing_coor_files)
elif version == "Standard":
	a.generate_mat_file_for_WHAM_reweighting(directory_containing_coor_files)
else:
	print ('error!')
