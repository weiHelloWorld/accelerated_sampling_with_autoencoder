from ANN_simulation_trp_cage import *
import argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="specify which pdb file to generate coordinates from")
args = parser.parse_args()


sutils.generate_coordinates_from_pdb_files(folder_for_pdb = '../target/' + args.file)

