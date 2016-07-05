from ANN_simulation import *
import argparse, subprocess, os

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="specify which pdb file to generate coordinates from")
parser.add_argument('--exclude_type', type=str, help='specify the molecule type of which the pdb files should be excluded')
args = parser.parse_args()

if not args.exclude_type == 'Alanine_dipeptide':
    temp_path = '../target/Alanine_dipeptide/' + args.file
    if os.path.exists(temp_path):
        Alanine_dipeptide().generate_coordinates_from_pdb_files(folder_for_pdb = temp_path)
    else:
        print "%s not existed!" % temp_path
if not args.exclude_type == "Trp_cage":
    temp_path = '../target/Trp_cage/' + args.file
    if os.path.exists(temp_path):
        Trp_cage().generate_coordinates_from_pdb_files(folder_for_pdb = temp_path)
    else:
        print "%s not existed!" % temp_path
