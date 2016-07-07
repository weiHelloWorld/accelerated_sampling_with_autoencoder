from ANN_simulation import *
import argparse, subprocess, os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="", help="specify the directory/file containing the pdb files")
args = parser.parse_args()

temp_path = args.path
if os.path.exists(temp_path):
    Trp_cage().generate_coordinates_from_pdb_files(folder_for_pdb = temp_path)
else:
    print "%s not existed!" % temp_path
