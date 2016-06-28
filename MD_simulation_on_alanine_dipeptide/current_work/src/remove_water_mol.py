from ANN_simulation_trp_cage import *
import argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="specify which pdb file of which the water molecules need to be removed")
args = parser.parse_args()


sutils.remove_water_mol_from_pdb_file(folder_for_pdb = '../target/' + args.file)
