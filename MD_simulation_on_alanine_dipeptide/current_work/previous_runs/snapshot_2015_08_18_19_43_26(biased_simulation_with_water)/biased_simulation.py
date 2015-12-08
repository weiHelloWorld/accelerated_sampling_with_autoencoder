# simulation (add CustomManyParticleForce)


from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

############################ PARAMETERS BEGIN ###############################################################
record_interval = 10 
total_number_of_steps = 10000

input_pdb_file_of_molecule = 'alanine_dipeptide.pdb'
force_field_file = 'amber99sb.xml'
water_field_file = 'amber99_obc.xml'

pdb_reporter_file = 'biased_output.pdb'
state_data_reporter_file = 'biased_report.txt'

xi_1_0 = "-0.6"  # they  are the coordinates of center of the potential well
xi_2_0 = "-0.4"

k1 = '300'
k2 = '300'

energy_expression = '''
%s * (layer_3_unit_1 - %s)^2 + %s * (layer_3_unit_2 - %s)^2;

layer_2_unit_1 = tanh(  0.624823 * layer_1_unit_1 + 0.564856 * layer_1_unit_2 + -0.121220 * layer_1_unit_3 + 0.561187 * layer_1_unit_4 + 0.639146 * layer_1_unit_5 + 0.950128 * layer_1_unit_6 + -0.505366 * layer_1_unit_7 + -0.175374 * layer_1_unit_8 + -2.656059);
layer_2_unit_2 = tanh(  -0.402610 * layer_1_unit_1 + -0.085324 * layer_1_unit_2 + 0.328071 * layer_1_unit_3 + 0.686114 * layer_1_unit_4 + 0.978815 * layer_1_unit_5 + -0.358761 * layer_1_unit_6 + 0.673455 * layer_1_unit_7 + 0.994804 * layer_1_unit_8 + 1.730097);
layer_2_unit_3 = tanh(  -0.004866 * layer_1_unit_1 + 0.095556 * layer_1_unit_2 + -0.211108 * layer_1_unit_3 + -0.004823 * layer_1_unit_4 + -0.021056 * layer_1_unit_5 + 0.051738 * layer_1_unit_6 + -0.110294 * layer_1_unit_7 + 0.017910 * layer_1_unit_8 + -0.343393);
layer_2_unit_4 = tanh(  -0.011042 * layer_1_unit_1 + -0.054019 * layer_1_unit_2 + 1.133528 * layer_1_unit_3 + -0.009699 * layer_1_unit_4 + 0.032756 * layer_1_unit_5 + -0.020382 * layer_1_unit_6 + 0.045919 * layer_1_unit_7 + 0.002452 * layer_1_unit_8 + 1.702319);
layer_2_unit_5 = tanh(  0.371405 * layer_1_unit_1 + 0.568842 * layer_1_unit_2 + 0.252674 * layer_1_unit_3 + 0.275218 * layer_1_unit_4 + 0.564059 * layer_1_unit_5 + 0.974478 * layer_1_unit_6 + -1.037493 * layer_1_unit_7 + -0.549787 * layer_1_unit_8 + -1.002440);
layer_2_unit_6 = tanh(  0.020340 * layer_1_unit_1 + -0.051366 * layer_1_unit_2 + 0.486628 * layer_1_unit_3 + 0.028276 * layer_1_unit_4 + -0.103158 * layer_1_unit_5 + -0.068130 * layer_1_unit_6 + -0.441406 * layer_1_unit_7 + 0.023748 * layer_1_unit_8 + 0.432553);
layer_2_unit_7 = tanh(  0.005844 * layer_1_unit_1 + -0.211705 * layer_1_unit_2 + -0.028302 * layer_1_unit_3 + 0.005947 * layer_1_unit_4 + -0.061929 * layer_1_unit_5 + 0.084278 * layer_1_unit_6 + -0.009478 * layer_1_unit_7 + 0.023650 * layer_1_unit_8 + -0.919552);
layer_2_unit_8 = tanh(  -0.059057 * layer_1_unit_1 + 0.658267 * layer_1_unit_2 + 0.889431 * layer_1_unit_3 + -0.027870 * layer_1_unit_4 + -0.020208 * layer_1_unit_5 + -0.015610 * layer_1_unit_6 + -0.216993 * layer_1_unit_7 + 0.082465 * layer_1_unit_8 + 1.448562);
layer_2_unit_9 = tanh(  -0.006457 * layer_1_unit_1 + 0.436915 * layer_1_unit_2 + -0.151642 * layer_1_unit_3 + 0.000878 * layer_1_unit_4 + -0.126873 * layer_1_unit_5 + -0.026717 * layer_1_unit_6 + -0.024258 * layer_1_unit_7 + 0.076820 * layer_1_unit_8 + -0.913736);
layer_2_unit_10 = tanh(  0.010728 * layer_1_unit_1 + -0.328686 * layer_1_unit_2 + 2.617891 * layer_1_unit_3 + -0.001645 * layer_1_unit_4 + -0.020104 * layer_1_unit_5 + -0.075958 * layer_1_unit_6 + 1.975245 * layer_1_unit_7 + -0.004033 * layer_1_unit_8 + 3.958138);


layer_3_unit_1 = tanh(  0.734188 * layer_2_unit_1 + 0.003134 * layer_2_unit_2 + 1.538687 * layer_2_unit_3 + 1.358441 * layer_2_unit_4 + 0.042905 * layer_2_unit_5 + 0.440081 * layer_2_unit_6 + -0.005726 * layer_2_unit_7 + -0.087633 * layer_2_unit_8 + -0.790571 * layer_2_unit_9 + 1.650011 * layer_2_unit_10 + -2.362960);
layer_3_unit_2 = tanh(  -0.067291 * layer_2_unit_1 + -0.003023 * layer_2_unit_2 + 0.092474 * layer_2_unit_3 + 0.220742 * layer_2_unit_4 + 0.047979 * layer_2_unit_5 + 0.082545 * layer_2_unit_6 + -1.084674 * layer_2_unit_7 + -0.054689 * layer_2_unit_8 + 0.288379 * layer_2_unit_9 + 0.212439 * layer_2_unit_10 + -1.397638);


dihedral_angle_1 = dihedral(p2, p5, p7, p9);
raw_layer_1_unit_1 = cos(dihedral_angle_1);
layer_1_unit_1 = (raw_layer_1_unit_1 - -1.000000) * 12.594699 - 1;
raw_layer_1_unit_5 = sin(dihedral_angle_1);
layer_1_unit_5 = (raw_layer_1_unit_5 - -0.529097) * 1.869480 - 1;
dihedral_angle_2 = dihedral(p5, p7, p9, p15);
raw_layer_1_unit_2 = cos(dihedral_angle_2);
layer_1_unit_2 = (raw_layer_1_unit_2 - -0.989427) * 1.202616 - 1;
raw_layer_1_unit_6 = sin(dihedral_angle_2);
layer_1_unit_6 = (raw_layer_1_unit_6 - -1.000000) * 2.339278 - 1;
dihedral_angle_3 = dihedral(p7, p9, p15, p17);
raw_layer_1_unit_3 = cos(dihedral_angle_3);
layer_1_unit_3 = (raw_layer_1_unit_3 - -0.999976) * 1.000012 - 1;
raw_layer_1_unit_7 = sin(dihedral_angle_3);
layer_1_unit_7 = (raw_layer_1_unit_7 - -0.998335) * 1.001242 - 1;
dihedral_angle_4 = dihedral(p9, p15, p17, p19);
raw_layer_1_unit_4 = cos(dihedral_angle_4);
layer_1_unit_4 = (raw_layer_1_unit_4 - -0.999999) * 15.300775 - 1;
raw_layer_1_unit_8 = sin(dihedral_angle_4);
layer_1_unit_8 = (raw_layer_1_unit_8 - -0.431321) * 2.160693 - 1;

''' %(k1, xi_1_0,k2, xi_2_0)


############################ PARAMETERS END ###############################################################


pdb = PDBFile(input_pdb_file_of_molecule) 
forcefield = ForceField(force_field_file, water_field_file) # with water
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
