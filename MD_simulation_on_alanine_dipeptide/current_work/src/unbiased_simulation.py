
# no biased potential

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

############################ PARAMETERS BEGIN ###############################################################
record_interval = 500
total_number_of_steps = 400000

input_pdb_file_of_molecule = '../resources/alanine_dipeptide.pdb'
force_field_file = 'amber99sb.xml'
water_field_file = 'amber99_obc.xml'

pdb_reporter_file = '../target/unbiased_output.pdb'
state_data_reporter_file = '../target/unbiased_report.txt'


############################ PARAMETERS END ###############################################################


pdb = PDBFile(input_pdb_file_of_molecule) 
forcefield = ForceField(force_field_file) # without water
system = forcefield.createSystem(pdb.topology,  nonbondedMethod=NoCutoff, \
                                 constraints=AllBonds)  # what does it mean by topology? 

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName("CUDA")
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)


simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter(pdb_reporter_file, record_interval))
simulation.reporters.append(StateDataReporter(state_data_reporter_file, record_interval, step=True, potentialEnergy=True, temperature=True))
simulation.step(total_number_of_steps)

print('Done!')
