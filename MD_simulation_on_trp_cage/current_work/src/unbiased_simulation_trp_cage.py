
# no biased potential

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

############################ PARAMETERS BEGIN ###############################################################

record_interval = 500
total_number_of_steps = 400000

temperature = 300   

input_pdb_file_of_molecule = '../resources/1l2y.pdb'
force_field_file = 'amber03.xml'
water_field_file = 'tip4pew.xml'

pdb_reporter_file = '../target/temp_output.pdb'
state_data_reporter_file = '../target/temp_report.txt'

box_size = 4.5
neg_ion = "Cl-"

time_step = 0.002        # simulation time step, in ps


############################ PARAMETERS END ###############################################################


pdb = PDBFile(input_pdb_file_of_molecule)
modeller = Modeller(pdb.topology, pdb.positions)

forcefield = ForceField(force_field_file, water_field_file)
modeller.addSolvent(forcefield, boxSize=Vec3(box_size, box_size, box_size)*nanometers, negativeIon = neg_ion)   # By default, addSolvent() creates TIP3P water molecules
modeller.addExtraParticles(forcefield)    # no idea what it is doing, but it works?

platform = Platform.getPlatformByName('CPU')

system = forcefield.createSystem(modeller.topology,  nonbondedMethod=Ewald, nonbondedCutoff = 1.0*nanometers, \
                                 constraints=AllBonds, ewaldErrorTolerance=0.0005)

system.addForce(AndersenThermostat(temperature*kelvin, 1/picosecond))

integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, time_step*picoseconds)
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)


print('begin Minimizing energy...')
simulation.minimizeEnergy()
print('Done minimizing energy.')

simulation.reporters.append(PDBReporter(pdb_reporter_file, record_interval))
simulation.reporters.append(StateDataReporter(state_data_reporter_file, record_interval, \
								step=True, potentialEnergy=True, temperature=True))
simulation.step(total_number_of_steps)

print('Done!')
