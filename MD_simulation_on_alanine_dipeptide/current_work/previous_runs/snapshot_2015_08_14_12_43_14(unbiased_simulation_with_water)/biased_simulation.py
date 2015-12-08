# simulation (add CustomManyParticleForce)


from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

############################ PARAMETERS BEGIN ###############################################################
record_interval = 5
total_number_of_steps = 4000

input_pdb_file_of_molecule = 'alanine_dipeptide.pdb'
force_field_file = 'amber99sb.xml'

pdb_reporter_file = 'biased_output.pdb'
state_data_reporter_file = 'biased_report.txt'

xi_1_0 = "0.25"  # they  are the coordinates of center of the potential well
xi_2_0 = "0.2"

k1 = '100'
k2 = '100'

energy_expression = '''
%s * (layer_3_unit_1 - %s)^2 + %s * (layer_3_unit_2 - %s)^2;

layer_2_unit_1 = tanh(  0.877037 * layer_1_unit_1 + 0.347990 * layer_1_unit_2 + 0.005693 * layer_1_unit_3 + 1.005140 * layer_1_unit_4 + 0.007232 * layer_1_unit_5 + 0.835003 * layer_1_unit_6 + -0.115248 * layer_1_unit_7 + 0.283534 * layer_1_unit_8 + -2.147890);
layer_2_unit_2 = tanh(  1.318027 * layer_1_unit_1 + 0.989178 * layer_1_unit_2 + -0.627503 * layer_1_unit_3 + 0.699056 * layer_1_unit_4 + 1.268530 * layer_1_unit_5 + -0.295129 * layer_1_unit_6 + 0.017457 * layer_1_unit_7 + -0.207536 * layer_1_unit_8 + -2.144287);
layer_2_unit_3 = tanh(  -0.915753 * layer_1_unit_1 + -0.294191 * layer_1_unit_2 + 0.010966 * layer_1_unit_3 + 0.372786 * layer_1_unit_4 + 0.832320 * layer_1_unit_5 + 0.279237 * layer_1_unit_6 + -0.663126 * layer_1_unit_7 + 0.731042 * layer_1_unit_8 + 0.988334);
layer_2_unit_4 = tanh(  -0.467286 * layer_1_unit_1 + -0.206760 * layer_1_unit_2 + -0.101939 * layer_1_unit_3 + -0.024593 * layer_1_unit_4 + -1.236918 * layer_1_unit_5 + 1.671020 * layer_1_unit_6 + -0.083779 * layer_1_unit_7 + 0.180281 * layer_1_unit_8 + 2.300970);
layer_2_unit_5 = tanh(  0.270532 * layer_1_unit_1 + 0.059102 * layer_1_unit_2 + 0.065795 * layer_1_unit_3 + -0.143065 * layer_1_unit_4 + 0.077700 * layer_1_unit_5 + 0.459661 * layer_1_unit_6 + 0.258937 * layer_1_unit_7 + -0.470990 * layer_1_unit_8 + 0.846883);
layer_2_unit_6 = tanh(  -0.014199 * layer_1_unit_1 + 0.072641 * layer_1_unit_2 + 0.255800 * layer_1_unit_3 + 0.210263 * layer_1_unit_4 + 0.428730 * layer_1_unit_5 + 0.457493 * layer_1_unit_6 + -0.280515 * layer_1_unit_7 + 0.127974 * layer_1_unit_8 + -0.101788);
layer_2_unit_7 = tanh(  -0.206002 * layer_1_unit_1 + 0.052775 * layer_1_unit_2 + -0.108654 * layer_1_unit_3 + 0.407912 * layer_1_unit_4 + 0.004792 * layer_1_unit_5 + 0.452431 * layer_1_unit_6 + 0.277407 * layer_1_unit_7 + -0.557300 * layer_1_unit_8 + -0.222791);
layer_2_unit_8 = tanh(  0.218365 * layer_1_unit_1 + 0.141467 * layer_1_unit_2 + 0.758856 * layer_1_unit_3 + 0.168453 * layer_1_unit_4 + 0.360643 * layer_1_unit_5 + -0.214640 * layer_1_unit_6 + -0.093805 * layer_1_unit_7 + -0.169296 * layer_1_unit_8 + 0.462329);
layer_2_unit_9 = tanh(  -0.036123 * layer_1_unit_1 + 0.077656 * layer_1_unit_2 + -1.012644 * layer_1_unit_3 + 0.131078 * layer_1_unit_4 + 0.099377 * layer_1_unit_5 + 0.145766 * layer_1_unit_6 + 0.237310 * layer_1_unit_7 + -0.869977 * layer_1_unit_8 + -1.471329);
layer_2_unit_10 = tanh(  -0.167880 * layer_1_unit_1 + -0.430843 * layer_1_unit_2 + -0.976190 * layer_1_unit_3 + 0.000643 * layer_1_unit_4 + 0.033738 * layer_1_unit_5 + -0.510786 * layer_1_unit_6 + 0.230375 * layer_1_unit_7 + 0.299908 * layer_1_unit_8 + -2.398451);


layer_3_unit_1 = tanh(  0.501512 * layer_2_unit_1 + 0.543757 * layer_2_unit_2 + 0.009760 * layer_2_unit_3 + 0.530117 * layer_2_unit_4 + 0.090065 * layer_2_unit_5 + 0.368213 * layer_2_unit_6 + -0.238092 * layer_2_unit_7 + -0.186269 * layer_2_unit_8 + 0.481446 * layer_2_unit_9 + -0.209328 * layer_2_unit_10 + 1.196981);
layer_3_unit_2 = tanh(  -0.167083 * layer_2_unit_1 + 0.731316 * layer_2_unit_2 + 0.026123 * layer_2_unit_3 + 0.590604 * layer_2_unit_4 + 0.292660 * layer_2_unit_5 + 0.108637 * layer_2_unit_6 + 0.149203 * layer_2_unit_7 + 0.088771 * layer_2_unit_8 + -0.325499 * layer_2_unit_9 + 0.690856 * layer_2_unit_10 + 0.831272);


dihedral_angle_1 = dihedral(p2, p5, p7, p9);
raw_layer_1_unit_1 = cos(dihedral_angle_1);
layer_1_unit_1 = (raw_layer_1_unit_1 - -1.000000) * 10.009593 - 1;
raw_layer_1_unit_5 = sin(dihedral_angle_1);
layer_1_unit_5 = (raw_layer_1_unit_5 - -0.599744) * 1.719538 - 1;
dihedral_angle_2 = dihedral(p5, p7, p9, p15);
raw_layer_1_unit_2 = cos(dihedral_angle_2);
layer_1_unit_2 = (raw_layer_1_unit_2 - -0.999997) * 2.420524 - 1;
raw_layer_1_unit_6 = sin(dihedral_angle_2);
layer_1_unit_6 = (raw_layer_1_unit_6 - -0.984793) * 2.035773 - 1;
dihedral_angle_3 = dihedral(p7, p9, p15, p17);
raw_layer_1_unit_3 = cos(dihedral_angle_3);
layer_1_unit_3 = (raw_layer_1_unit_3 - -1.000000) * 2.421335 - 1;
raw_layer_1_unit_7 = sin(dihedral_angle_3);
layer_1_unit_7 = (raw_layer_1_unit_7 - 0.000177) * 2.031351 - 1;
dihedral_angle_4 = dihedral(p9, p15, p17, p19);
raw_layer_1_unit_4 = cos(dihedral_angle_4);
layer_1_unit_4 = (raw_layer_1_unit_4 - -1.000000) * 9.232228 - 1;
raw_layer_1_unit_8 = sin(dihedral_angle_4);
layer_1_unit_8 = (raw_layer_1_unit_8 - -0.549316) * 1.708124 - 1;


''' %(k1, xi_1_0,k2, xi_2_0)


############################ PARAMETERS END ###############################################################


pdb = PDBFile(input_pdb_file_of_molecule) 
forcefield = ForceField(force_field_file) # vacumn, no water
system = forcefield.createSystem(pdb.topology,  nonbondedMethod=NoCutoff, \
                                 constraints=AllBonds)  # what does it mean by topology? 

# add custom force

force = CustomManyParticleForce(22, energy_expression) 
for i in range(system.getNumParticles()):
    force.addParticle("",0)  # what kinds of types should we specify here for each atom?
system.addForce(force)
# end add custom force
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds) 
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)


simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter(pdb_reporter_file, record_interval))
simulation.reporters.append(StateDataReporter(state_data_reporter_file, record_interval, step=True, potentialEnergy=True, temperature=True))
simulation.step(total_number_of_steps)

print('Done!')
