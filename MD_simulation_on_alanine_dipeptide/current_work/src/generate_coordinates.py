from ANN_simulation import *
import argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="specify which pdb file to generate coordinates from")
args = parser.parse_args()


Alanine_dipeptide().generate_coordinates_from_pdb_files(folder_for_pdb = '../target/Alanine_dipeptide/' + args.file)
Trp_cage().generate_coordinates_from_pdb_files(folder_for_pdb = '../target/Trp_cage/' + args.file)
