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

# Select what residues numbers you wish to align
# and put them in a list
start_id = 1
end_id   = 20
atoms_to_be_aligned = range(start_id, end_id + 1)

# Start the parser
pdb_parser = Bio.PDB.PDBParser(QUIET = True)

# Get the structures
ref_structure = pdb_parser.get_structure("reference", ref_structure_pdb_file)
sample_structure = pdb_parser.get_structure("sample", sample_structure_pdb_file)

# Use the first model in the pdb-files for alignment
# Change the number 0 if you want to align to another structure
ref_model    = ref_structure[0]
sample_model = sample_structure[0]

# Make a list of the atoms (in the structures) you wish to align.
# In this case we use CA atoms whose index is in the specified range
ref_atoms = []
sample_atoms = []

# Iterate of all chains in the model in order to find all residues
for ref_chain in ref_model:
    for ref_res in ref_chain:
    # Check if residue number ( .get_id() ) is in the list
        if ref_res.get_id()[1] in atoms_to_be_aligned:
            ref_atoms.append(ref_res['CA'])

# Do the same for the sample structure
for sample_chain in sample_model:
    for sample_res in sample_chain:
        if sample_res.get_id()[1] in atoms_to_be_aligned:
            sample_atoms.append(sample_res['CA'])

# Now we initiate the superimposer:
super_imposer = Bio.PDB.Superimposer()
super_imposer.set_atoms(ref_atoms, sample_atoms)
super_imposer.apply(sample_model.get_atoms())

# Print RMSD:
print super_imposer.rms

# Save the aligned version
io = Bio.PDB.PDBIO()
io.set_structure(sample_structure)
io.save(output_pdb_file)
