from ANN_simulation import *
import argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="", help="specify the directory/file containing the pdb files")
parser.add_argument("--remove_original", help="remove original pdb files", action="store_true")
args = parser.parse_args()

if args.remove_original:
    Trp_cage().remove_water_mol_and_Cl_from_pdb_file(folder_for_pdb = args.path, preserve_original_file=False)
else:
    Trp_cage().remove_water_mol_and_Cl_from_pdb_file(folder_for_pdb = args.path, preserve_original_file=True)
