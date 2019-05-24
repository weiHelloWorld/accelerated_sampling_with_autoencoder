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
parser.add_argument("--ref", type=str, help="reference pdb file")
parser.add_argument("--name", type=str, default=None, help='name of the aligned pdb file')
parser.add_argument('--remove_original', help='remove original pdb file after doing structural alignment', action="store_true")
parser.add_argument('--suffix', type=str, default="", help="string that appends at the end of filename")
parser.add_argument('--atom_selection', type=str, default='backbone', help='atom_selection_statement for alignment')
args = parser.parse_args()

ref_pdb = args.ref

traj_files = subprocess.check_output([
    'find', args.sample_path, '-name', "*.pdb", '-o', '-name', '*.dcd']).decode("utf-8").strip().split('\n')
if args.ignore_aligned_file:
    traj_files = [x for x in traj_files if not '_aligned' in x]

for sample_traj in traj_files:
    print("doing structural alignment for %s" % sample_traj)

    if args.name is None:
        output_pdb_file = sample_traj[:-4] + '_aligned%s.pdb' % (args.suffix)
    else:
        output_pdb_file = parser.name

    if os.path.exists(output_pdb_file) and os.path.getmtime(sample_traj) < os.path.getmtime(output_pdb_file):
        print("aligned file already exists: %s (remove previous one if needed)" % output_pdb_file)
    else:
        ref = Universe(ref_pdb)
        m_traj = Universe(ref_pdb, sample_traj)
        AlignTraj(m_traj, reference=ref, filename=output_pdb_file, select=args.atom_selection).run()
        print("done structural alignment for %s" % sample_traj)

        if args.remove_original:
            subprocess.check_output(['rm', sample_traj])
            print("%s removed!" % sample_traj)
    