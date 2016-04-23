from ANN_simulation import *

a=pickle.load(open('../resources/network_2.pkl','rb'))

a.generate_files_for_Bayes_WHAM('../target/wham/')
