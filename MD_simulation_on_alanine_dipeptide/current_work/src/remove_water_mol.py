from ANN_simulation import *
import argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="", help="specify the directory/file containing the pdb files")
args = parser.parse_args()


Trp_cage().remove_water_mol_and_Cl_from_pdb_file(folder_for_pdb = args.path)
