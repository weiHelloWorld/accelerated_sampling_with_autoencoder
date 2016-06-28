
# no biased potential

import datetime, os
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import ast

from ANN import *
from config import *

############################ PARAMETERS BEGIN ###############################################################


record_interval = int(sys.argv[1])
total_number_of_steps = int(sys.argv[2])

force_constant = sys.argv[3] 

folder_to_store_output_files = '../target/' + sys.argv[4] # this is used to separate outputs for different networks into different folders
autoencoder_info_file = '../resources/' + sys.argv[5]

potential_center = list(map(lambda x: float(x), sys.argv[6].replace('"','').split(',')))   # this API is the generalization for higher-dimensional cases

if len(sys.argv) == 8:  # temperature is optional, it is 300 K by default
    temperature = int(sys.argv[7])   # in kelvin
else:
    temperature = 300

if not os.path.exists(folder_to_store_output_files):
    try:
        os.makedirs(folder_to_store_output_files)
    except:
        pass
        

assert(os.path.exists(folder_to_store_output_files))

input_pdb_file_of_molecule = '../resources/1l2y.pdb'
force_field_file = 'amber03.xml'
water_field_file = 'tip4pew.xml'

if force_constant == '0':   # unbiased case
    pdb_reporter_file = '%s/unbiased_%dK_output.pdb' % (folder_to_store_output_files, temperature)  # typically the folder for unbiased case is ../target/unbiased/
    state_data_reporter_file = '%s/unbiased_%dK_report.txt' % (folder_to_store_output_files, temperature)
else:
    pdb_reporter_file = '%s/biased_output_fc_%s_x1_%s_x2_%s.pdb' %(folder_to_store_output_files, force_constant, xi_1_0, xi_2_0)
    state_data_reporter_file = '%s/biased_report_fc_%s_x1_%s_x2_%s.txt' %(folder_to_store_output_files, force_constant, xi_1_0, xi_2_0)

if os.path.isfile(pdb_reporter_file):
    os.rename(pdb_reporter_file, pdb_reporter_file.split('.pdb')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pdb") # ensure the file extension stays the same

if os.path.isfile(state_data_reporter_file):
    os.rename(state_data_reporter_file, state_data_reporter_file.split('.txt')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt")

k1 = force_constant
k2 = force_constant

# with open(autoencoder_info_file, 'r') as f_in:
#     energy_expression = f_in.read()
#
# if CONFIG_20:   # whether the PC space is periodic in [- pi, pi], True for circular network, False for Tanh network, this affect the form of potential function
#     energy_expression = '''
#     %s * d1_square + %s * d2_square;
#     d1_square = min( min( (PC0 - %s)^2, (PC0 - %s + 6.2832)^2 ), (PC0 - %s - 6.2832)^2 );
#     d2_square = min( min( (PC1 - %s)^2, (PC1 - %s + 6.2832)^2 ), (PC1 - %s - 6.2832)^2 );
#     ''' % (k1, k2, xi_1_0, xi_1_0, xi_1_0, xi_2_0, xi_2_0, xi_2_0) + energy_expression
#
# else:
#     energy_expression = '''
#     %s * (PC0 - %s)^2 + %s * (PC1 - %s)^2;
#
#     ''' %(k1, xi_1_0, k2, xi_2_0) + energy_expression

flag_random_seed = 0 # whether we need to fix this random seed

box_size = 4.5    # in nm
neg_ion = "Cl-"

time_step = CONFIG_22       # simulation time step, in ps

index_of_backbone_atoms = [1, 2, 3, 17, 18, 19, 36, 37, 38, 57, 58, 59, 76, 77, 78, 93, 94, 95, \
        117, 118, 119, 136, 137, 138, 158, 159, 160, 170, 171, 172, 177, 178, 179, 184, \
        185, 186, 198, 199, 200, 209, 210, 211, 220, 221, 222, 227, 228, 229, 251, 252, \
        253, 265, 266, 267, 279, 280, 281, 293, 294, 295]

layer_types = ['Tanh', 'Tanh']


############################ PARAMETERS END ###############################################################


pdb = PDBFile(input_pdb_file_of_molecule)
modeller = Modeller(pdb.topology, pdb.positions)

forcefield = ForceField(force_field_file, water_field_file)
modeller.addSolvent(forcefield, boxSize=Vec3(box_size, box_size, box_size)*nanometers, negativeIon = neg_ion)   # By default, addSolvent() creates TIP3P water molecules
modeller.addExtraParticles(forcefield)    # no idea what it is doing, but it works?

platform = Platform.getPlatformByName(CONFIG_23)
platform.loadPluginsFromDirectory(CONFIG_25)  # load the plugin from specific directory

system = forcefield.createSystem(modeller.topology,  nonbondedMethod=Ewald, nonbondedCutoff = 1.0*nanometers, \
                                 constraints=AllBonds, ewaldErrorTolerance=0.0005)

system.addForce(AndersenThermostat(temperature*kelvin, 1/picosecond))
system.addForce(MonteCarloBarostat(1*atmospheres, temperature*kelvin, 25))

# add custom force (only for biased simulation)
if force_constant != '0':
    force = ANN_Force()

    force.set_layer_types(layer_types)
    

    force.set_list_of_index_of_atoms_forming_dihedrals_from_index_of_backbone_atoms(index_of_backbone_atoms)
    force.set_num_of_nodes(CONFIG_3[:3])
    force.set_potential_center(
        [float(xi_1_0), float(xi_2_0)] 
        )
    force.set_force_constant(float(force_constant))

    # TODO: parse coef_file
    with open(autoencoder_info_file, 'r') as f_in:
        content = f_in.readlines()


    force.set_coeffients_of_connections(
        [ast.literal_eval(content[0].strip())[0], ast.literal_eval(content[1].strip())[0]]
                                    )

    force.set_values_of_biased_nodes([
        ast.literal_eval(content[2].strip())[0], ast.literal_eval(content[3].strip())[0]
        ])

    system.addForce(force)
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
