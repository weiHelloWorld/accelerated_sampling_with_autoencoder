from ANN_simulation import *

a=pickle.load(open('../resources/network_6.pkl','rb'))

a.generate_files_for_Bayes_WHAM('../target/wham_6/')
