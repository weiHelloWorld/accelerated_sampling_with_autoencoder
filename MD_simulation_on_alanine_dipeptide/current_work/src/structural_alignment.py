"""
modified from the code: https://gist.github.com/andersx/6354971
"""

import Bio.PDB, sys

ref_structure_pdb_file = sys.argv[1]
sample_structure_pdb_file = sys.argv[2]

if len(sys.argv) == 4:
    output_pdb_file = sys.argv[3]
elif len(sys.argv) == 3:
    output_pdb_file = sample_structure_pdb_file.split('.pdb')[0] + '_aligned.pdb'
else:
    raise Exception('parameter num error')

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
