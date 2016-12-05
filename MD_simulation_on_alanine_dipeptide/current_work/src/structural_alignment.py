"""
modified from the code: https://gist.github.com/andersx/6354971
"""

import Bio.PDB, argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("sample_path", type=str, help="path (file or folder) of pdb file(s) to be aligned")
parser.add_argument("--ignore_aligned_file",type=int, default=1)
parser.add_argument("--ref", type=str, default="../resources/1l2y.pdb", help="reference pdb file")
parser.add_argument("--name", type=str, default=None, help='name of the aligned pdb file')
parser.add_argument('--remove_original', help='remove original pdb file after doing structural alignment', action="store_true")
args = parser.parse_args()

ref_structure_pdb_file = args.ref

pdb_files = subprocess.check_output(['find', args.sample_path, '-name', "*.pdb"]).strip().split('\n')
if args.ignore_aligned_file:
    pdb_files = filter(lambda x: not '_aligned.pdb' in x, pdb_files)

for sample_structure_pdb_file in pdb_files:
    print "doing structural alignment for %s" % sample_structure_pdb_file

    if args.name is None:
        output_pdb_file = sample_structure_pdb_file.split('.pdb')[0] + '_aligned.pdb'
    else:
        output_pdb_file = parser.name

    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    ref_structure = pdb_parser.get_structure("reference", ref_structure_pdb_file)
    sample_structure = pdb_parser.get_structure("sample", sample_structure_pdb_file)

    ref_atoms = [item for item in ref_structure[0].get_atoms()]

    for sample_model in sample_structure:
        sample_atoms = [item for item in sample_model.get_atoms()]
        super_imposer = Bio.PDB.Superimposer()
        super_imposer.set_atoms(ref_atoms, sample_atoms)
        super_imposer.apply(sample_model.get_atoms())
        # print super_imposer.rms

    # Save the aligned version
    io = Bio.PDB.PDBIO()
    io.set_structure(sample_structure)
    io.save(output_pdb_file)

    print "done structural alignment for %s" % sample_structure_pdb_file

    if args.remove_original:
        subprocess.check_output(['rm', sample_structure_pdb_file])
        print "%s removed!" % sample_structure_pdb_file
    