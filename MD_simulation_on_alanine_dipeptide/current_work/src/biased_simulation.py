# simulation (add CustomManyParticleForce)

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from ANN import *
import ast

import os
import datetime

from config import *


############################ PARAMETERS BEGIN ###############################################################
record_interval = int(sys.argv[1])
total_number_of_steps = int(sys.argv[2])

force_constant = sys.argv[3] 

folder_to_store_output_files = '../target/Alanine_dipeptide/' + sys.argv[4] # this is used to separate outputs for different networks into different folders
autoencoder_info_file = '../resources/Alanine_dipeptide/' + sys.argv[5]

potential_center = list(map(lambda x: float(x), sys.argv[6].replace('"','').split(',')))   # this API is the generalization for higher-dimensional cases

if not os.path.exists(folder_to_store_output_files):
    try:
        os.makedirs(folder_to_store_output_files)
    except:
        pass
        

assert(os.path.exists(folder_to_store_output_files))

input_pdb_file_of_molecule = '../resources/alanine_dipeptide.pdb'

force_field_file = 'amber99sb.xml'

pdb_reporter_file = '%s/biased_output_fc_%s_pc_%s.pdb' %(folder_to_store_output_files, force_constant, str(potential_center).replace(' ',''))
state_data_reporter_file = '%s/biased_report_fc_%s_pc_%s.txt' %(folder_to_store_output_files, force_constant, str(potential_center).replace(' ',''))

# check if the file exist
if os.path.isfile(pdb_reporter_file):
    os.rename(pdb_reporter_file, pdb_reporter_file.split('.pdb')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pdb") # ensure the file extension stays the same

if os.path.isfile(state_data_reporter_file):
    os.rename(state_data_reporter_file, state_data_reporter_file.split('.txt')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt")

k1 = force_constant
k2 = force_constant

list_of_index_of_atoms_forming_dihedrals = [[2,5,7,9],
                                            [5,7,9,15],
                                            [7,9,15,17],
                                            [9,15,17,19]]

layer_types = CONFIG_27

with open(autoencoder_info_file, 'r') as f_in:
    energy_expression = f_in.read()

# FIXME: following expression is out-of-date due to the change in API for higher-dimensional cases
if CONFIG_28 == "CustomManyParticleForce":
    [xi_1_0, xi_2_0] = potential_center
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

simulation_temperature = CONFIG_21
time_step = CONFIG_22   # simulation time step, in ps

############################ PARAMETERS END ###############################################################

pdb = PDBFile(input_pdb_file_of_molecule) 
forcefield = ForceField(force_field_file) # without water
system = forcefield.createSystem(pdb.topology,  nonbondedMethod=NoCutoff, \
                                 constraints=AllBonds)  

# add biased force, could be either "CustomManyParticleForce" (provided in the package) or "ANN_Force" (I wrote)

if CONFIG_28 == "CustomManyParticleForce":
    force = CustomManyParticleForce(22, energy_expression) 
    for i in range(system.getNumParticles()):
        force.addParticle("",0)  # what kinds of types should we specify here for each atom?
    system.addForce(force)

if CONFIG_28 == "ANN_Force":
    if force_constant != '0':
        force = ANN_Force()
        force.set_layer_types(layer_types)
        force.set_list_of_index_of_atoms_forming_dihedrals(list_of_index_of_atoms_forming_dihedrals)
        force.set_num_of_nodes(CONFIG_3[:3])
        force.set_potential_center(
            potential_center
            )
        force.set_force_constant(float(force_constant))

        # set coefficient
        with open(autoencoder_info_file, 'r') as f_in:
            content = f_in.readlines()

        force.set_coeffients_of_connections(
            [ast.literal_eval(content[0].strip())[0], ast.literal_eval(content[1].strip())[0]]
                                        )

        force.set_values_of_biased_nodes([
            ast.literal_eval(content[2].strip())[0], ast.literal_eval(content[3].strip())[0]
            ])

        system.addForce(force)
# end of biased force

integrator = LangevinIntegrator(simulation_temperature*kelvin, 1/picosecond, time_step*picoseconds)
if flag_random_seed:
    integrator.setRandomNumberSeed(1)  # set random seed

platform = Platform.getPlatformByName(CONFIG_23)
platform.loadPluginsFromDirectory(CONFIG_25)  # load the plugin from specific directory

simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter(pdb_reporter_file, record_interval))
simulation.reporters.append(StateDataReporter(state_data_reporter_file, record_interval, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True))
simulation.step(total_number_of_steps)

print('Done biased simulation!')
