
# no biased potential

import datetime, os
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

from config import *

############################ PARAMETERS BEGIN ###############################################################

record_interval = int(sys.argv[1])
total_number_of_steps = int(sys.argv[2])

temperature = int(sys.argv[3])   # in Kelvin

folder_to_store_output_files = '../target/' + sys.argv[4] # this is used to separate outputs for different networks into different folders
energy_expression_file = '../resources/' + sys.argv[5]

force_constant = sys.argv[6]

xi_1_0 = sys.argv[7]
xi_2_0 = sys.argv[8]

input_pdb_file_of_molecule = '../resources/1l2y.pdb'
force_field_file = 'amber03.xml'
water_field_file = 'tip4pew.xml'

pdb_reporter_file = '../target/unbiased_%dK_output.pdb' % temperature
state_data_reporter_file = '../target/unbiased_%dK_report.txt' % temperature

if os.path.isfile(pdb_reporter_file):
    os.rename(pdb_reporter_file, pdb_reporter_file.split('.pdb')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pdb") # ensure the file extension stays the same

if os.path.isfile(state_data_reporter_file):
    os.rename(state_data_reporter_file, state_data_reporter_file.split('.txt')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt")

k1 = force_constant
k2 = force_constant

with open(energy_expression_file, 'r') as f_in:
    energy_expression = f_in.read()

if CONFIG_20:   # whether the PC space is periodic in [- pi, pi], True for circular network, False for Tanh network, this affect the form of potential function
    energy_expression = '''
    %s * d1_square + %s * d2_square;
    d1_square = min( min( (PC0 - %s)^2, (PC0 - %s + 6.2832)^2 ), (PC0 - %s - 6.2832)^2 );
    d2_square = min( min( (PC1 - %s)^2, (PC1 - %s + 6.2832)^2 ), (PC1 - %s - 6.2832)^2 );
    ''' % (k1, k2, xi_1_0, xi_1_0, xi_1_0, xi_2_0, xi_2_0, xi_2_0) + energy_expression

else:
    energy_expression = '''
    %s * (PC0 - %s)^2 + %s * (PC1 - %s)^2;

    ''' %(k1, xi_1_0, k2, xi_2_0) + energy_expression

flag_random_seed = 0 # whether we need to fix this random seed

box_size = 4.5    # in nm
neg_ion = "Cl-"

flag_random_seed = 0 # whether we need to fix this random seed

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
system.addForce(MonteCarloBarostat(1*atmospheres, temperature*kelvin, 25))

# add custom force

customForce = CustomManyParticleForce(304, energy_expression)
for i in range(system.getNumParticles()):
    customForce.addParticle("",0)  # what kinds of types should we specify here for each atom?
system.addForce(customForce)
# end add custom force

integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, time_step*picoseconds)

if flag_random_seed:
    integrator.setRandomNumberSeed(1)  # set random seed

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
