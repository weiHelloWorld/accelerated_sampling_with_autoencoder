
# no biased potential

import datetime, os, argparse
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import ast

from config import *

############################ PARAMETERS BEGIN ###############################################################

parser = argparse.ArgumentParser()
parser.add_argument("record_interval", type=int, help="interval to take snapshots")
parser.add_argument("total_num_of_steps", type=int, help="total number of simulation steps")
parser.add_argument("force_constant", type=str, help="force constants")
parser.add_argument("folder_to_store_output_files", type=str, help="folder to store the output pdb and report files")
parser.add_argument("autoencoder_info_file", type=str, help="file to store autoencoder information (coefficients)")
parser.add_argument("pc_potential_center", type=str, help="potential center (should include 'pc_' as prefix)")
parser.add_argument("whether_to_add_water_mol_opt", type=str, help='whether we need to add water molecules in the simulation')
parser.add_argument("ensemble_type", type=str, help='simulation ensemble type, either NVT or NPT')
parser.add_argument("--temperature", type=int, default= 300, help='simulation temperature')
parser.add_argument("--starting_pdb_file", type=str, default='../resources/1l2y.pdb', help='the input pdb file to start simulation')
parser.add_argument("--minimize_energy", type=int, default=1, help='whether to minimize energy (1 = yes, 0 = no)')
parser.add_argument("--platform", type=str, default=CONFIG_23, help='platform on which the simulation is run')
args = parser.parse_args()

record_interval = args.record_interval
total_number_of_steps = args.total_num_of_steps
force_constant = args.force_constant

folder_to_store_output_files = args.folder_to_store_output_files # this is used to separate outputs for different networks into different folders
autoencoder_info_file = args.autoencoder_info_file

potential_center = list(map(lambda x: float(x), args.pc_potential_center.replace('"','')\
                                .replace('pc_','').split(',')))   # this API is the generalization for higher-dimensional cases

if args.whether_to_add_water_mol_opt == 'with_water':
    whether_to_add_water_mol = True
elif args.whether_to_add_water_mol_opt == 'without_water' or args.whether_to_add_water_mol_opt == 'water_already_added':
    whether_to_add_water_mol = False
else:
    raise Exception('parameter error')

temperature = args.temperature

if not os.path.exists(folder_to_store_output_files):
    try:
        os.makedirs(folder_to_store_output_files)
    except:
        pass

assert(os.path.exists(folder_to_store_output_files))

input_pdb_file_of_molecule = args.starting_pdb_file
force_field_file = 'amber03.xml'
water_field_file = 'tip4pew.xml'

pdb_reporter_file = '%s/output_fc_%s_pc_%s_T_%d_%s.pdb' % (folder_to_store_output_files, force_constant,
                                                          str(potential_center).replace(' ', ''), temperature, args.whether_to_add_water_mol_opt)
state_data_reporter_file = '%s/report_fc_%s_pc_%s_T_%d_%s.txt' % (folder_to_store_output_files, force_constant,
                                                                 str(potential_center).replace(' ', ''), temperature, args.whether_to_add_water_mol_opt)

if args.starting_pdb_file != '../resources/1l2y.pdb':
    pdb_reporter_file = pdb_reporter_file.split('.pdb')[0] + '_sf_%s.pdb' % \
                            args.starting_pdb_file.split('.pdb')[0].split('/')[-1]   # 'sf' means 'starting_from'
    state_data_reporter_file = state_data_reporter_file.split('.txt')[0] + '_sf_%s.txt' % \
                            args.starting_pdb_file.split('.pdb')[0].split('/')[-1]

if os.path.isfile(pdb_reporter_file):
    os.rename(pdb_reporter_file, pdb_reporter_file.split('.pdb')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pdb") # ensure the file extension stays the same

if os.path.isfile(state_data_reporter_file):
    os.rename(state_data_reporter_file, state_data_reporter_file.split('.txt')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt")

k1 = force_constant
k2 = force_constant

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

if whether_to_add_water_mol:    # if we need to add water molecules in the simulation
    forcefield = ForceField(force_field_file, water_field_file)

    modeller.addSolvent(forcefield, boxSize=Vec3(box_size, box_size, box_size)*nanometers, negativeIon = neg_ion)   # By default, addSolvent() creates TIP3P water molecules
    modeller.addExtraParticles(forcefield)    # no idea what it is doing, but it works?
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=Ewald, nonbondedCutoff=1.0 * nanometers,
                                     constraints = AllBonds, ewaldErrorTolerance = 0.0005)

else:
    forcefield = ForceField(force_field_file, water_field_file)
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1.0 * nanometers,
                                     constraints=AllBonds)

platform = Platform.getPlatformByName(args.platform)
platform.loadPluginsFromDirectory(CONFIG_25)  # load the plugin from specific directory

system.addForce(AndersenThermostat(temperature*kelvin, 1/picosecond))
if args.ensemble_type == "NPT":
    system.addForce(MonteCarloBarostat(1*atmospheres, temperature*kelvin, 25))
elif args.ensemble_type == "NVT":
    pass
else:
    raise Exception("ensemble = %s not found!" % args.ensemble_type)

# add custom force (only for biased simulation)
if force_constant != '0':
    from ANN import *
    force = ANN_Force()

    force.set_layer_types(layer_types)
    

    force.set_list_of_index_of_atoms_forming_dihedrals_from_index_of_backbone_atoms(index_of_backbone_atoms)
    force.set_num_of_nodes(CONFIG_3[:3])
    force.set_potential_center(potential_center)
    force.set_force_constant(float(force_constant))

    with open(autoencoder_info_file, 'r') as f_in:
        content = f_in.readlines()

    force.set_coeffients_of_connections(
        [ast.literal_eval(content[0].strip())[0], ast.literal_eval(content[1].strip())[0]])

    force.set_values_of_biased_nodes([
        ast.literal_eval(content[2].strip())[0], ast.literal_eval(content[3].strip())[0]
        ])

    system.addForce(force)
# end add custom force

integrator = VerletIntegrator(time_step*picoseconds)

if flag_random_seed:
    integrator.setRandomNumberSeed(1)  # set random seed

simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

if args.minimize_energy:
    print('begin Minimizing energy...')
    simulation.minimizeEnergy()
    print('Done minimizing energy.')
else:
    print('energy minimization not required')

simulation.reporters.append(PDBReporter(pdb_reporter_file, record_interval))
simulation.reporters.append(StateDataReporter(state_data_reporter_file, record_interval, \
								step=True, potentialEnergy=True, temperature=True))
simulation.step(total_number_of_steps)

print('Done!')
