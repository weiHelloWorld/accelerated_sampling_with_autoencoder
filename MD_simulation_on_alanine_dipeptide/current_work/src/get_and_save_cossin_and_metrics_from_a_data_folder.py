from ANN_simulation import *

parser = argparse.ArgumentParser()
parser.add_argument("data_folder", type=str, help="folder containing simulation data")
parser.add_argument("--save_cossin", type=int, default=1, help="save cossin data into file")
parser.add_argument("--save_CA_RMSD", type=int, default=1, help="save CA RMSD into file")
parser.add_argument("--save_radius_of_gyration", type=int, default=1, help="save radius of gyration into file")
parser.add_argument("--save_residue_9_16_distance", type=int, default=1, help="save residue 9-16 distance into file")
parser.add_argument("--save_residue_1_20_distance", type=int, default=1, help="save residue 1-20 distance into file")
args = parser.parse_args()

data_folder = args.data_folder
data_folder = data_folder[:-1] if data_folder[-1] == '/' else data_folder

my_coor = coordinates_data_files_list([data_folder])
my_coor_file_list = my_coor.get_list_of_coor_data_files()
my_pdb_file_list = my_coor.get_list_of_corresponding_pdb_files()

info_cossin_file = data_folder + '/info_cossin.txt'
info_CA_RMSD_file = data_folder + '/info_CA_RMSD.txt'
info_radius_of_gyration_file = data_folder + '/info_radius_of_gyration.txt'
info_residue_9_16_distance_file = data_folder + '/info_residue_9_16_distance.txt'
info_residue_1_20_distance_file = data_folder + '/info_residue_1_20_distance.txt'

if args.save_cossin:
    np.savetxt(info_cossin_file, Trp_cage.get_many_cossin_from_coordinates_in_list_of_files(my_coor_file_list))

if args.save_CA_RMSD:
    np.savetxt(info_CA_RMSD_file, Trp_cage.metric_RMSD_of_atoms(my_pdb_file_list))

if args.save_radius_of_gyration:
    np.savetxt(info_radius_of_gyration_file, Trp_cage.metric_radius_of_gyration(my_pdb_file_list))

distances_list = Trp_cage.get_pairwise_distance_matrices_of_alpha_carbon(my_pdb_file_list)

if args.save_residue_9_16_distance:
    np.savetxt(info_residue_9_16_distance_file, np.array([item[8][15] for item in distances_list]))

if args.save_residue_1_20_distance:
    np.savetxt(info_residue_1_20_distance_file, np.array([item[0][19] for item in distances_list]))
