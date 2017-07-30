from ANN_simulation import *
import datetime, os, argparse
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import ast
from config import *

############################ PARAMETERS BEGIN ###############################################################

parser = argparse.ArgumentParser()
parser.add_argument("molecule", type=str, help="type of molecule for the simulation")
parser.add_argument("record_interval", type=int, help="interval to take snapshots")
parser.add_argument("total_num_of_steps", type=int, help="total number of simulation steps")
parser.add_argument("force_constant", type=float, help="force constants")
parser.add_argument("folder_to_store_output_files", type=str, help="folder to store the output pdb and report files")
parser.add_argument("autoencoder_info_file", type=str, help="file to store autoencoder information (coefficients)")
parser.add_argument("pc_potential_center", type=str, help="potential center (should include 'pc_' as prefix)")
parser.add_argument("whether_to_add_water_mol_opt", type=str, help='whether to add water (options: explicit, implicit, water_already_included, no_water)')
parser.add_argument("ensemble_type", type=str, help='simulation ensemble type, either NVT or NPT')
parser.add_argument("--output_pdb", type=str, default=None, help="name of output pdb file")
parser.add_argument("--scaling_factor", type=float, default = CONFIG_49/10, help='scaling_factor for ANN_Force')
parser.add_argument("--temperature", type=int, default= 300, help='simulation temperature')
parser.add_argument("--starting_pdb_file", type=str, default='auto', help='the input pdb file to start simulation')
parser.add_argument("--starting_frame", type=int, default=0, help="index of starting frame in the starting pdb file")
parser.add_argument("--minimize_energy", type=int, default=1, help='whether to minimize energy (1 = yes, 0 = no)')
parser.add_argument("--data_type_in_input_layer", type=int, default=0, help='data_type_in_input_layer, 0 = cos/sin, 1 = Cartesian coordinates')
parser.add_argument("--platform", type=str, default=CONFIG_23, help='platform on which the simulation is run')
parser.add_argument("--device", type=str, default='none', help='device index to run simulation on')
parser.add_argument("--checkpoint", type=int, default=1, help="whether to save checkpoint at the end of the simulation")
parser.add_argument("--starting_checkpoint", type=str, default="auto", help='starting checkpoint file, to resume simulation ("none" means no starting checkpoint file is provided, "auto" means automatically)')
parser.add_argument("--equilibration_steps", type=int, default=1000, help="number of steps for the equilibration process")
parser.add_argument("--fast_equilibration", type=int, default=0, help="do fast equilibration by running biased simulation with larger force constant")
parser.add_argument("--remove_eq_file", type=int, default=1, help="remove equilibration pdb files associated with fast equilibration")
parser.add_argument("--auto_equilibration", help="enable auto equilibration so that it will run enough equilibration steps", action="store_true")
# next few options are for metadynamics
parser.add_argument("--bias_method", type=str, default='US', help="biasing method for enhanced sampling, US = umbrella sampling, MTD = metadynamics")
parser.add_argument("--MTD_pace", type=int, default=CONFIG_66, help="pace of metadynamics")
parser.add_argument("--MTD_height", type=float, default=CONFIG_67, help="height of metadynamics")
parser.add_argument("--MTD_sigma", type=float, default=CONFIG_68, help="sigma of metadynamics")
parser.add_argument("--MTD_WT", type=int, default=CONFIG_69, help="whether to use well-tempered version")
parser.add_argument("--MTD_biasfactor", type=float, default=CONFIG_70, help="biasfactor of well-tempered metadynamics")
# following is for plumed script
parser.add_argument("--plumed_file", type=str, default=None, help="plumed script for biasing force, used only when the bias_method == plumed_other")
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

print "start simulation at %s" % datetime.datetime.now()  # to calculate compile time

record_interval = args.record_interval
total_number_of_steps = args.total_num_of_steps
force_constant = args.force_constant
scaling_factor = args.scaling_factor

platform = Platform.getPlatformByName(args.platform)
temperature = args.temperature
input_data_type = ['cossin', 'Cartesian'][args.data_type_in_input_layer]

if float(force_constant) != 0:
    from ANN import *
    platform.loadPluginsFromDirectory(CONFIG_25)  # load the plugin from specific directory

folder_to_store_output_files = args.folder_to_store_output_files # this is used to separate outputs for different networks into different folders
autoencoder_info_file = args.autoencoder_info_file

potential_center = list(map(lambda x: float(x), args.pc_potential_center.replace('"','')\
                                .replace('pc_','').split(',')))   # this API is the generalization for higher-dimensional cases

def run_simulation(force_constant, number_of_simulation_steps):
    if not os.path.exists(folder_to_store_output_files):
        try:
            os.makedirs(folder_to_store_output_files)
        except:
            pass

    assert(os.path.exists(folder_to_store_output_files))

    force_field_file = {'Trp_cage': 'amber03.xml', '2src': 'amber03.xml', '1y57': 'amber03.xml'}[args.molecule]
    water_field_file = {'Trp_cage': 'tip4pew.xml', '2src': 'tip3p.xml', '1y57': 'tip3p.xml'}[args.molecule]
    water_model = {'Trp_cage': 'tip4pew', '2src': 'tip3p', '1y57': 'tip3p'}[args.molecule]
    ionic_strength = {'Trp_cage': 0 * molar, '2src': 0.5 * .15 * molar, '1y57': 0.5 * .15 * molar}[args.molecule]
    implicit_solvent_force_field = 'amber03_obc.xml'

    pdb_reporter_file = '%s/output_fc_%s_pc_%s_T_%d_%s.pdb' % (folder_to_store_output_files, force_constant,
                                                              str(potential_center).replace(' ', ''), temperature, args.whether_to_add_water_mol_opt)


    if args.starting_pdb_file == 'auto':
        input_pdb_file_of_molecule = {'Trp_cage': '../resources/1l2y.pdb',
                                      '2src': '../resources/2src.pdb',
                                      '1y57': '../resources/1y57.pdb'}[args.molecule]
    else:
        input_pdb_file_of_molecule = args.starting_pdb_file
        pdb_reporter_file = pdb_reporter_file.split('.pdb')[0] + '_sf_%s.pdb' % \
                                args.starting_pdb_file.split('.pdb')[0].split('/')[-1]   # 'sf' means 'starting_from'
        state_data_reporter_file = state_data_reporter_file.split('.txt')[0] + '_sf_%s.txt' % \
                                args.starting_pdb_file.split('.pdb')[0].split('/')[-1]

    print "start_pdb = %s" % input_pdb_file_of_molecule
    if args.starting_frame != 0:
        pdb_reporter_file = pdb_reporter_file.split('.pdb')[0] + '_ff_%d.pdb' % args.starting_frame   # 'ff' means 'from_frame'
        state_data_reporter_file = state_data_reporter_file.split('.txt')[0] + '_ff_%d.txt' % args.starting_frame

    if not args.output_pdb is None:
        pdb_reporter_file = args.output_pdb

    state_data_reporter_file = pdb_reporter_file.replace('output_fc', 'report_fc').replace('.pdb', '.txt')
    checkpoint_file = pdb_reporter_file.replace('output_fc', 'checkpoint_fc').replace('.pdb', '.chk')
    if args.fast_equilibration:
        checkpoint_file = checkpoint_file.replace(str(force_constant), str(args.force_constant))

    # check existence
    if os.path.isfile(pdb_reporter_file):
        os.rename(pdb_reporter_file, pdb_reporter_file.split('.pdb')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pdb") # ensure the file extension stays the same

    if os.path.isfile(state_data_reporter_file):
        os.rename(state_data_reporter_file, state_data_reporter_file.split('.txt')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt")

    flag_random_seed = 0 # whether we need to fix this random seed
    box_size = {'Trp_cage': 4.5, '2src': 8.0, '1y57': 8.0}[args.molecule]
    time_step = CONFIG_22       # simulation time step, in ps

    index_of_backbone_atoms = {'Trp_cage': CONFIG_57[1],
                               '2src': CONFIG_57[2], '1y57': CONFIG_57[2]}[args.molecule]

    layer_types = CONFIG_27
    simulation_constraints = HBonds

    pdb = PDBFile(input_pdb_file_of_molecule)
    modeller = Modeller(pdb.topology, pdb.getPositions(frame=args.starting_frame))

    if args.whether_to_add_water_mol_opt == 'explicit':
        forcefield = ForceField(force_field_file, water_field_file)
        modeller.addHydrogens(forcefield)
        modeller.addSolvent(forcefield, model=water_model, boxSize=Vec3(box_size, box_size, box_size)*nanometers,
                            ionicStrength=ionic_strength)
        modeller.addExtraParticles(forcefield)
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * nanometers,
                                         constraints = simulation_constraints, ewaldErrorTolerance = 0.0005)
    elif args.whether_to_add_water_mol_opt == 'implicit':
        forcefield = ForceField(force_field_file, implicit_solvent_force_field)
        modeller.addHydrogens(forcefield)
        modeller.addExtraParticles(forcefield)
        system = forcefield.createSystem(pdb.topology,nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=5 * nanometers,
                                         constraints=simulation_constraints, rigidWater=True, removeCMMotion=True)

    elif args.whether_to_add_water_mol_opt == 'no_water' or args.whether_to_add_water_mol_opt == 'water_already_included':
        forcefield = ForceField(force_field_file, water_field_file)
        modeller.addHydrogens(forcefield)
        modeller.addExtraParticles(forcefield)
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff,nonbondedCutoff=1.0 * nanometers,
                                         constraints = simulation_constraints)
    else:
        raise Exception("parameter error")

    system.addForce(AndersenThermostat(temperature*kelvin, 1/picosecond))
    if args.ensemble_type == "NPT" and args.whether_to_add_water_mol_opt == 'explicit':
        system.addForce(MonteCarloBarostat(1*atmospheres, temperature*kelvin, 25))

    # add custom force (only for biased simulation)
    if args.bias_method == "US":
        if float(force_constant) != 0:
            force = ANN_Force()

            force.set_layer_types(layer_types)
            force.set_data_type_in_input_layer(args.data_type_in_input_layer)
            force.set_list_of_index_of_atoms_forming_dihedrals_from_index_of_backbone_atoms(index_of_backbone_atoms)
            force.set_index_of_backbone_atoms(index_of_backbone_atoms)
            force.set_num_of_nodes(CONFIG_3[:3])
            force.set_potential_center(potential_center)
            force.set_force_constant(float(force_constant))
            force.set_scaling_factor(float(scaling_factor))

            with open(autoencoder_info_file, 'r') as f_in:
                content = f_in.readlines()

            force.set_coeffients_of_connections(
                [ast.literal_eval(content[0].strip())[0], ast.literal_eval(content[1].strip())[0]])

            force.set_values_of_biased_nodes([
                ast.literal_eval(content[2].strip())[0], ast.literal_eval(content[3].strip())[0]
                ])

            system.addForce(force)
    elif args.bias_method == "MTD":
        from openmmplumed import PlumedForce
        molecule_type = {'Trp_cage': Trp_cage, '2src': Src_kinase, '1y57': Src_kinase}[args.molecule]
        plumed_force_string = molecule_type.get_expression_script_for_plumed()
        with open(autoencoder_info_file, 'r') as f_in:
            plumed_force_string += f_in.read()

        # note that dimensionality of MTD is determined by potential_center string
        mtd_output_layer_string = ['l_2_out_%d' % item for item in range(len(potential_center))]
        mtd_output_layer_string = ','.join(mtd_output_layer_string)
        mtd_sigma_string = ','.join([str(args.MTD_sigma) for _ in range(len(potential_center))])
        if args.MTD_WT:
            mtd_well_tempered_string = 'TEMP=%d BIASFACTOR=%f' % (args.temperature, args.MTD_biasfactor)
        else:
            mtd_well_tempered_string = ""
        plumed_force_string += """
        metad: METAD ARG=%s PACE=%d HEIGHT=%f SIGMA=%s FILE=temp_MTD_hills.txt %s
        PRINT STRIDE=%d ARG=%s,metad.bias FILE=temp_MTD_out.txt
        """ % (mtd_output_layer_string, args.MTD_pace, args.MTD_height, mtd_sigma_string, mtd_well_tempered_string,
               record_interval, mtd_output_layer_string)
        system.addForce(PlumedForce(plumed_force_string))
    elif args.bias_method == "TMD":  # targeted MD
        # TODO: this is temporary version
        from openmmplumed import PlumedForce
        kappa_string = str(args.force_constant)
        plumed_force_string = """
rmsd: RMSD REFERENCE=../resources/1y57_TMD.pdb TYPE=OPTIMAL
restraint: MOVINGRESTRAINT ARG=rmsd AT0=0.4 STEP0=0 KAPPA0=%s AT1=0 STEP1=%d KAPPA1=%s
PRINT STRIDE=500 ARG=* FILE=COLVAR
            """ % (kappa_string, total_number_of_steps, kappa_string)
        system.addForce(PlumedForce(plumed_force_string))
    elif args.bias_method == "plumed_other":
        from openmmplumed import PlumedForce
        with open(args.plumed_file, 'r') as f_in:
            plumed_force_string = f_in.read()
        system.addForce(PlumedForce(plumed_force_string))
    else:
        raise Exception('bias method error')
    # end add custom force

    integrator = VerletIntegrator(time_step*picoseconds)

    if flag_random_seed:
        integrator.setRandomNumberSeed(1)  # set random seed

    if args.platform == "CUDA" and args.device != 'none':
        properties = {'CudaDeviceIndex': args.device}
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    else:
        simulation = Simulation(modeller.topology, system, integrator, platform)
    # print "positions = "
    # print (modeller.positions)
    simulation.context.setPositions(modeller.positions)
    print datetime.datetime.now()

    if args.starting_checkpoint != 'none':
        if args.starting_checkpoint == "auto":  # restart from checkpoint if it exists
            if os.path.isfile(checkpoint_file):
                print ("resume simulation from %s" % checkpoint_file)
                simulation.loadCheckpoint(checkpoint_file)
        else:
            print ("resume simulation from %s" % args.starting_checkpoint)
            simulation.loadCheckpoint(args.starting_checkpoint)     # the topology is already set by pdb file, and the positions in the pdb file will be overwritten by those in the starting_checkpoing file

    if args.minimize_energy:
        print('begin Minimizing energy...')
        print datetime.datetime.now()
        simulation.minimizeEnergy()
        print('Done minimizing energy.')
        print datetime.datetime.now()
    else:
        print('energy minimization not required')

    print("begin equilibrating...")
    print datetime.datetime.now()
    simulation.step(args.equilibration_steps)
    previous_distance_to_potential_center = 100
    current_distance_to_potential_center = 90
    if args.auto_equilibration:
        distance_change_tolerance = 0.05
        while abs(previous_distance_to_potential_center - current_distance_to_potential_center) > distance_change_tolerance:
            temp_pdb_reporter_file_for_auto_equilibration = pdb_reporter_file.replace('.pdb', '_temp.pdb')
            simulation.reporters.append(PDBReporter(temp_pdb_reporter_file_for_auto_equilibration, record_interval))
            simulation.step(args.equilibration_steps)
            previous_distance_to_potential_center = current_distance_to_potential_center
            current_distance_to_potential_center = get_distance_between_data_cloud_center_and_potential_center(
                            temp_pdb_reporter_file_for_auto_equilibration)
            subprocess.check_output(['rm', temp_pdb_reporter_file_for_auto_equilibration])
            print "previous_distance_to_potential_center =  %f\ncurrent_distance_to_potential_center = %f" % (
                previous_distance_to_potential_center, current_distance_to_potential_center
            )

    print("Done equilibration")
    print datetime.datetime.now()

    simulation.reporters.append(PDBReporter(pdb_reporter_file, record_interval))
    simulation.reporters.append(StateDataReporter(state_data_reporter_file, record_interval, time=True,
                                    step=True, potentialEnergy=True, kineticEnergy=True, speed=True,
                                                  temperature=True, progress=True, remainingTime=True,
                                                  totalSteps=number_of_simulation_steps + args.equilibration_steps,
                                                  ))
    simulation.step(number_of_simulation_steps)

    if args.checkpoint:
        if os.path.isfile(checkpoint_file):
            os.rename(checkpoint_file, checkpoint_file.split('.chk')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".chk")
        simulation.saveCheckpoint(checkpoint_file)

    print('Done!')
    print datetime.datetime.now()
    return pdb_reporter_file

def get_distance_between_data_cloud_center_and_potential_center(pdb_file):
    coor_file = Trp_cage().generate_coordinates_from_pdb_files(pdb_file)[0]
    temp_network = pickle.load(open(args.autoencoder_file, 'rb'))
    print coor_file
    this_simulation_data = single_biased_simulation_data(temp_network, coor_file)
    offset = this_simulation_data.get_offset_between_potential_center_and_data_cloud_center(input_data_type)
    if CONFIG_17[1] == CircularLayer:
        offset = [min(abs(item), abs(item + 2 * np.pi), abs(item - 2 * np.pi)) for item in offset]
        print "circular offset"
    print 'offset = %s' % str(offset)
    distance = sqrt(sum([item * item for item in offset]))
    return distance

if __name__ == '__main__':
    if not args.fc_adjustable:
        if args.fast_equilibration:
            temp_eq_force_constants = [args.force_constant * item for item in [5, 3, 2, 1.5, 1.2]]
            temp_eq_num_steps = [int(total_number_of_steps * item) for item in [0.02, 0.05, 0.05, 0.1, 0.1]]
            for item_1, item_2 in zip(temp_eq_force_constants, temp_eq_num_steps):
                temp_eq_pdb = run_simulation(item_1, item_2)
                if args.remove_eq_file:
                    subprocess.check_output(['rm', temp_eq_pdb])
        
        run_simulation(args.force_constant, total_number_of_steps)

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
            pdb_file = run_simulation(force_constant, total_number_of_steps)
            distance_of_data_cloud_center = get_distance_between_data_cloud_center_and_potential_center(pdb_file)
            force_constant += args.fc_step
            print "distance_between_data_cloud_center_and_potential_center = %f" % distance_of_data_cloud_center
