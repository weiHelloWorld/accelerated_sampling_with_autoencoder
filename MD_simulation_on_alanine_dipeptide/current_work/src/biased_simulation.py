# biased simulation

from ANN_simulation import *
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import ast, argparse

import os
import datetime

from config import *

parser = argparse.ArgumentParser()
parser.add_argument("record_interval", type=int, help="interval to take snapshots")
parser.add_argument("total_num_of_steps", type=int, help="total number of simulation steps")
parser.add_argument("force_constant", type=float, help="force constants")
parser.add_argument("folder_to_store_output_files", type=str, help="folder to store the output pdb and report files")
parser.add_argument("autoencoder_info_file", type=str, help="file to store autoencoder information (coefficients)")
parser.add_argument("pc_potential_center", type=str, help="potential center (should include 'pc_' as prefix)")
parser.add_argument("--layer_types", type=str, default=str(CONFIG_27), help='layer types')
parser.add_argument("--num_of_nodes", type=str, default=str(CONFIG_3[:3]), help='number of nodes in each layer')
parser.add_argument("--temperature", type=int, default= CONFIG_21, help='simulation temperature')
parser.add_argument("--starting_pdb_file", type=str, default='../resources/alanine_dipeptide.pdb', help='the input pdb file to start simulation')
# note on "force_constant_adjustable" mode:
# the simulation will stop if either:
# force constant is greater or equal to max_force_constant
# or distance between center of data cloud and potential center is smaller than distance_tolerance
parser.add_argument("--fc_adjustable", help="set the force constant to be adjustable", action="store_true")
parser.add_argument("--max_fc", type=float, default=CONFIG_32, help="max force constant (for force_constant_adjustable mode)")
parser.add_argument("--fc_step", type=float, default=CONFIG_34, help="the value by which the force constant is increased each time (for force_constant_adjustable mode)")
parser.add_argument("--distance_tolerance", type=float, default=CONFIG_35, help="max distance allowed between center of data cloud and potential center (for force_constant_adjustable mode)")
parser.add_argument("--autoencoder_file", type=str, help="pkl file that stores autoencoder (for force_constant_adjustable mode)")
parser.add_argument("--remove_previous", help="remove previous outputs while adjusting force constants", action="store_true")
args = parser.parse_args()

record_interval = args.record_interval
total_number_of_steps = args.total_num_of_steps
force_constant = args.force_constant
layer_types = re.sub("\[|\]|\"|\'| ",'', args.layer_types).split(',')
num_of_nodes = re.sub("\[|\]|\"|\'| ",'', args.num_of_nodes).split(',')
num_of_nodes = [int(item) for item in num_of_nodes]

if float(force_constant) != 0:
    from ANN import *

folder_to_store_output_files = args.folder_to_store_output_files # this is used to separate outputs for different networks into different folders
autoencoder_info_file = args.autoencoder_info_file

potential_center = list(map(lambda x: float(x), args.pc_potential_center.replace('"','')\
                                .replace('pc_','').split(',')))   # this API is the generalization for higher-dimensional cases

def run_simulation(force_constant):
    if not os.path.exists(folder_to_store_output_files):
        try:
            os.makedirs(folder_to_store_output_files)
        except:
            pass

    assert(os.path.exists(folder_to_store_output_files))

    input_pdb_file_of_molecule = args.starting_pdb_file

    force_field_file = 'amber99sb.xml'

    pdb_reporter_file = '%s/biased_output_fc_%f_pc_%s.pdb' %(folder_to_store_output_files, force_constant, str(potential_center).replace(' ',''))
    state_data_reporter_file = '%s/biased_report_fc_%f_pc_%s.txt' %(folder_to_store_output_files, force_constant, str(potential_center).replace(' ',''))

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


    with open(autoencoder_info_file, 'r') as f_in:
        energy_expression = f_in.read()

    # FIXME: following expression is out-of-date due to the change in API for higher-dimensional cases
    if CONFIG_28 == "CustomManyParticleForce":
        [xi_1_0, xi_2_0] = potential_center
        if CONFIG_20:   # whether the PC space is periodic in [- pi, pi], True for circular network, False for Tanh network, this affect the form of potential function
            energy_expression = '''
            %f * d1_square + %f * d2_square;
            d1_square = min( min( (PC0 - %s)^2, (PC0 - %s + 6.2832)^2 ), (PC0 - %s - 6.2832)^2 );
            d2_square = min( min( (PC1 - %s)^2, (PC1 - %s + 6.2832)^2 ), (PC1 - %s - 6.2832)^2 );
            ''' % (k1, k2, xi_1_0, xi_1_0, xi_1_0, xi_2_0, xi_2_0, xi_2_0) + energy_expression

        else:
            energy_expression = '''
            %f * (PC0 - %s)^2 + %f * (PC1 - %s)^2;

            ''' %(k1, xi_1_0, k2, xi_2_0) + energy_expression

    flag_random_seed = 0 # whether we need to fix this random seed

    simulation_temperature = args.temperature
    time_step = CONFIG_22   # simulation time step, in ps

    pdb = PDBFile(input_pdb_file_of_molecule)
    forcefield = ForceField(force_field_file) # without water
    system = forcefield.createSystem(pdb.topology,  nonbondedMethod=NoCutoff,
                                     constraints=AllBonds)

    # add biased force, could be either "CustomManyParticleForce" (provided in the package) or "ANN_Force" (I wrote)

    if CONFIG_28 == "CustomManyParticleForce":
        force = CustomManyParticleForce(22, energy_expression)
        for _ in range(system.getNumParticles()):
            force.addParticle("",0)  # what kinds of types should we specify here for each atom?
        system.addForce(force)

    elif CONFIG_28 == "ANN_Force":
        if force_constant != '0' and force_constant != 0:
            force = ANN_Force()
            force.set_layer_types(layer_types)
            force.set_list_of_index_of_atoms_forming_dihedrals(list_of_index_of_atoms_forming_dihedrals)

            force.set_num_of_nodes(num_of_nodes)
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
    return pdb_reporter_file

def get_distance_between_data_cloud_center_and_potential_center(pdb_file):
    coor_file = Alanine_dipeptide().generate_coordinates_from_pdb_files(pdb_file)[0]
    temp_network = pickle.load(open(args.autoencoder_file, 'rb'))
    this_simulation_data = single_biased_simulation_data(temp_network, coor_file)
    offset = this_simulation_data.get_offset_between_potential_center_and_data_cloud_center()
    if layer_types[1] == "Circular":
        offset = [min(abs(item), abs(item + 2 * np.pi), abs(item - 2 * np.pi)) for item in offset]
        print "circular offset"
    print 'offset = %s' % str(offset)
    distance = sqrt(sum([item * item for item in offset]))
    return distance

if __name__ == '__main__':
    if not args.fc_adjustable:
        run_simulation(args.force_constant)
    else:
        force_constant = args.force_constant
        distance_of_data_cloud_center = float("inf")
        while force_constant < args.max_fc and distance_of_data_cloud_center > args.distance_tolerance:
            if args.remove_previous:
                try:
                    command = 'rm %s/*%s*' % (folder_to_store_output_files, str(potential_center).replace(' ',''))
                    command = command.replace('[','').replace(']','')
                    subprocess.check_output(command, shell=True)
                    print "removing previous results..."
                except:
                    pass
            pdb_file = run_simulation(force_constant)
            distance_of_data_cloud_center = get_distance_between_data_cloud_center_and_potential_center(pdb_file)
            force_constant += args.fc_step
            print "distance_between_data_cloud_center_and_potential_center = %f" % distance_of_data_cloud_center

