"""
modified from the code: https://gist.github.com/andersx/6354971
"""

import Bio.PDB, argparse, subprocess, os
from MDAnalysis import *
from MDAnalysis.analysis.align import *
from MDAnalysis.analysis.rms import rmsd

parser = argparse.ArgumentParser()
parser.add_argument("sample_path", type=str, help="path (file or folder) of pdb file(s) to be aligned")
parser.add_argument("--ignore_aligned_file",type=int, default=1)
parser.add_argument("--ref", type=str, default="../resources/1l2y.pdb", help="reference pdb file")
parser.add_argument("--name", type=str, default=None, help='name of the aligned pdb file')
parser.add_argument('--remove_original', help='remove original pdb file after doing structural alignment', action="store_true")
parser.add_argument('--suffix', type=str, default="", help="string that appends at the end of filename")
parser.add_argument('--atom_selection', type=str, default='backbone', help='atom_selection_statement for alignment')
args = parser.parse_args()

ref_structure_pdb_file = args.ref

pdb_files = subprocess.check_output(['find', args.sample_path, '-name', "*.pdb"]).decode("utf-8").strip().split('\n')
if args.ignore_aligned_file:
    pdb_files = [x for x in pdb_files if not '_aligned' in x]

for sample_structure_pdb_file in pdb_files:
    print("doing structural alignment for %s" % sample_structure_pdb_file)

    if args.name is None:
        output_pdb_file = sample_structure_pdb_file.split('.pdb')[0] + '_aligned%s.pdb' % (args.suffix)
    else:
        output_pdb_file = parser.name

    if os.path.exists(output_pdb_file) and os.path.getmtime(sample_structure_pdb_file) < os.path.getmtime(output_pdb_file):
        print("aligned file already exists: %s (remove previous one if needed)" % output_pdb_file)
    else:
        ref = Universe(ref_structure_pdb_file) 
        trj = Universe(sample_structure_pdb_file) 
        predefined_filename = rms_fit_trj(trj, ref, select=args.atom_selection)
        subprocess.check_output(['mv', predefined_filename, output_pdb_file])

        print("done structural alignment for %s" % sample_structure_pdb_file)

        if args.remove_original:
            subprocess.check_output(['rm', sample_structure_pdb_file])
            print("%s removed!" % sample_structure_pdb_file)
    