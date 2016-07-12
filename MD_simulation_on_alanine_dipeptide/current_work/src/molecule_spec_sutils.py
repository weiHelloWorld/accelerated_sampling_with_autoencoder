"""Sutils: simulation unilities, some of them are molecule-specific (implemented as methods in subclasses)
"""

import copy, pickle, re, os, time, subprocess, datetime, itertools
from config import *
from Bio import PDB

class Sutils(object):
    def __init__(self):
        return

    @staticmethod
    def remove_water_mol_and_Cl_from_pdb_file(folder_for_pdb = CONFIG_12, preserve_original_file=False):
        """
        This is used to remove water molecule from pdb file, purposes:
        - save storage space
        - reduce processing time of pdb file
        """
        filenames = subprocess.check_output(['find', folder_for_pdb, '-name', '*.pdb']).split('\n')[:-1]
        for item in filenames:
            print ('removing water molecules from pdb file: ' + item)
            output_file = item[:-4] + '_rm_tmp.pdb'

            with open(item, 'r') as f_in, open(output_file, 'w') as f_out:
                for line in f_in:
                    if not 'HOH' in line and not 'CL' in line:
                        f_out.write(line)

            if not preserve_original_file:
                subprocess.check_output(['mv', output_file, item])

        print('Done removing water molecules from all pdb files!')
        return

    @staticmethod
    def get_boundary_points(list_of_points,
                            range_of_PCs = CONFIG_26,
                            num_of_bins = CONFIG_10,
                            num_of_boundary_points = CONFIG_11,
                            is_circular_boundary = CONFIG_18,
                            preprocessing = True,
                            ):
        '''
        :param preprocessing: if True, then more weight is not linear, this would be better based on experience
        '''
        dimensionality = len(list_of_points[0])
        list_of_points = zip(*list_of_points)
        hist_matrix, edges = np.histogramdd(list_of_points, bins= num_of_bins * np.ones(dimensionality), range = range_of_PCs)

        # following is the main algorithm to find boundary and holes
        # simply find the points that are lower than average of its 4 neighbors

        if preprocessing:
            hist_matrix = map(lambda x: map(lambda y: - np.exp(- y), x), hist_matrix)   # preprocessing process

        if is_circular_boundary:  # typically works for circular autoencoder
            diff_with_neighbors = hist_matrix - 1.0 / (2 * dimensionality) \
                                            * sum(
                                                map(lambda x: np.roll(hist_matrix, 1, axis=x) + np.roll(hist_matrix, -1, axis=x),
                                                    range(dimensionality)
                                                    )
                                                )
        else:
            # TODO: code not concise and general enough, fix this later
            sum_of_neighbors = np.zeros(num_of_bins * np.ones(dimensionality))
            for item in range(dimensionality):
                temp = np.roll(hist_matrix, 1, axis=item)
                if item == 0:
                    temp[0] = 0
                elif item == 1:
                    temp[:,0] = 0
                elif item == 2:
                    temp[:,:,0] = 0
                elif item == 3:
                    temp[:,:,:,0] = 0
                else:
                    raise Exception("has not been implemented yet!")

                sum_of_neighbors += temp

                temp = np.roll(hist_matrix, -1, axis=item)
                if item == 0:
                    temp[-1] = 0
                elif item == 1:
                    temp[:,-1] = 0
                elif item == 2:
                    temp[:,:,-1] = 0
                elif item == 3:
                    temp[:,:,:,-1] = 0
                else:
                    raise Exception("has not been implemented yet!")

                sum_of_neighbors += temp

            diff_with_neighbors = hist_matrix - 1 / (2 * dimensionality) * sum_of_neighbors

        # get grid centers
        edge_centers = map(lambda x: 0.5 * (np.array(x[1:]) + np.array(x[:-1])), edges)
        grid_centers = np.array(list(itertools.product(*edge_centers)))  # "itertools.product" gives Cartesian/direct product of several lists
        grid_centers = np.reshape(grid_centers, np.append(num_of_bins * np.ones(dimensionality), dimensionality))
        # print grid_centers

        potential_centers = []

        # now sort these grids (that has no points in it)
        # based on total number of points in its neighbors
        
        temp_seperate_index = []

        for item in range(dimensionality):
            temp_seperate_index.append(range(num_of_bins))

        index_of_grids = list(itertools.product(
                        *temp_seperate_index
                        ))

        sorted_index_of_grids = sorted(index_of_grids, key = lambda x: diff_with_neighbors[x]) # sort based on histogram, return index values

        for index in sorted_index_of_grids[:num_of_boundary_points]:  # note index can be of dimension >= 2
            temp_potential_center = map(lambda x: round(x, 2), grid_centers[index])
            potential_centers.append(temp_potential_center)

        return potential_centers


class Alanine_dipeptide(Sutils):
    """docstring for Alanine_dipeptide"""
    def __init__(self):
        return
        
    @staticmethod
    def get_cossin_from_a_coordinate(a_coordinate):
        num_of_coordinates = len(a_coordinate) / 3
        a_coordinate = np.array(a_coordinate).reshape(num_of_coordinates, 3)
        diff_coordinates = a_coordinate[1:num_of_coordinates, :] - a_coordinate[0:num_of_coordinates - 1,:]  # bond vectors
        diff_coordinates_1=diff_coordinates[0:num_of_coordinates-2,:];diff_coordinates_2=diff_coordinates[1:num_of_coordinates-1,:]
        normal_vectors = np.cross(diff_coordinates_1, diff_coordinates_2)
        normal_vectors_normalized = np.array(map(lambda x: x / sqrt(np.dot(x,x)), normal_vectors))
        normal_vectors_normalized_1 = normal_vectors_normalized[0:num_of_coordinates-3, :]; normal_vectors_normalized_2 = normal_vectors_normalized[1:num_of_coordinates-2,:];
        diff_coordinates_mid = diff_coordinates[1:num_of_coordinates-2] # these are bond vectors in the middle (remove the first and last one), they should be perpendicular to adjacent normal vectors

        cos_of_angles = range(len(normal_vectors_normalized_1))
        sin_of_angles_vec = range(len(normal_vectors_normalized_1))
        sin_of_angles = range(len(normal_vectors_normalized_1)) # initialization
        result = []

        for index in range(len(normal_vectors_normalized_1)):
            cos_of_angles[index] = np.dot(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
            sin_of_angles_vec[index] = np.cross(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
            sin_of_angles[index] = sqrt(np.dot(sin_of_angles_vec[index], sin_of_angles_vec[index])) * np.sign(sum(sin_of_angles_vec[index]) * sum(diff_coordinates_mid[index]))
            result += [cos_of_angles[index], sin_of_angles[index]]

        return result

    @staticmethod
    def get_many_cossin_from_coordinates(coordinates):
        return map(Alanine_dipeptide.get_cossin_from_a_coordinate, coordinates)

    @staticmethod
    def get_many_cossin_from_coordiantes_in_list_of_files(list_of_files):
        result = []
        for item in list_of_files:
            coordinates = np.loadtxt(item)
            temp = Alanine_dipeptide.get_many_cossin_from_coordinates(coordinates)
            result += temp

        return result

    @staticmethod
    def get_many_dihedrals_from_coordinates_in_file (list_of_files):
        # why we need to get dihedrals from a list of coordinate files?
        # because we will probably need to plot other files outside self._list_of_coor_data_files
        temp = Alanine_dipeptide.get_many_cossin_from_coordiantes_in_list_of_files(list_of_files)
        return Alanine_dipeptide.get_many_dihedrals_from_cossin(temp)

    @staticmethod
    def get_many_dihedrals_from_cossin(cossin):
        result = []
        for item in cossin:
            assert (len(item) == 8)
            temp_angle = []
            for ii in range(4):
                temp_angle += [np.arccos(item[2 * ii]) * np.sign(item[2 * ii + 1])]
            
            result += [list(temp_angle)]
        return result

    @staticmethod
    def generate_coordinates_from_pdb_files(folder_for_pdb = CONFIG_12):
        filenames = subprocess.check_output(['find', folder_for_pdb, '-name' ,'*.pdb']).split('\n')[:-1]

        index_of_backbone_atoms = ['2', '5', '7', '9', '15', '17', '19']

        for input_file in filenames:
            print ('generating coordinates of ' + input_file)
            output_file = input_file[:-4] + '_coordinates.txt'

            with open(input_file) as f_in:
                with open(output_file, 'w') as f_out:
                    for line in f_in:
                        fields = line.strip().split()
                        if (fields[0] == 'ATOM' and fields[1] in index_of_backbone_atoms):
                            f_out.write(reduce(lambda x,y: x + '\t' + y, fields[6:9]))
                            f_out.write('\t')
                        elif fields[0] == "MODEL" and fields[1] != "1":
                            f_out.write('\n')

                    f_out.write('\n')  # last line
        print("Done generating coordinates files\n")
        return

    @staticmethod
    def get_expression_for_input_of_this_molecule():
        index_of_backbone_atoms = [2, 5, 7, 9, 15, 17, 19]
        expression_for_input_of_this_molecule = ''
        for i in range(len(index_of_backbone_atoms) - 3):
            index_of_coss = 2 * i
            index_of_sins = 2 * i + 1
            expression_for_input_of_this_molecule += 'out_layer_0_unit_%d = raw_layer_0_unit_%d;\n' % (index_of_coss, index_of_coss)
            expression_for_input_of_this_molecule += 'out_layer_0_unit_%d = raw_layer_0_unit_%d;\n' % (index_of_sins, index_of_sins)
            expression_for_input_of_this_molecule += 'raw_layer_0_unit_%d = cos(dihedral_angle_%d);\n' % (index_of_coss, i)
            expression_for_input_of_this_molecule += 'raw_layer_0_unit_%d = sin(dihedral_angle_%d);\n' % (index_of_sins, i)
            expression_for_input_of_this_molecule += 'dihedral_angle_%d = dihedral(p%d, p%d, p%d, p%d);\n' % (i, index_of_backbone_atoms[i], index_of_backbone_atoms[i+1],index_of_backbone_atoms[i+2],index_of_backbone_atoms[i+3])
        return expression_for_input_of_this_molecule


class Trp_cage(Sutils):
    """docstring for Trp_cage"""
    def __init__(self):
        return
        
    @staticmethod
    def get_cossin_of_a_dihedral_from_four_atoms(coord_1, coord_2, coord_3, coord_4):
        """each parameter is a 3D Cartesian coordinates of an atom"""
        coords_of_four = np.array([coord_1, coord_2, coord_3, coord_4])
        num_of_coordinates = 4
        diff_coordinates = coords_of_four[1:num_of_coordinates, :] - coords_of_four[0:num_of_coordinates - 1,:]  # bond vectors
        diff_coordinates_1=diff_coordinates[0:num_of_coordinates-2,:];diff_coordinates_2=diff_coordinates[1:num_of_coordinates-1,:]
        normal_vectors = np.cross(diff_coordinates_1, diff_coordinates_2)
        normal_vectors_normalized = np.array(map(lambda x: x / sqrt(np.dot(x,x)), normal_vectors))
        normal_vectors_normalized_1 = normal_vectors_normalized[0:num_of_coordinates-3, :]; normal_vectors_normalized_2 = normal_vectors_normalized[1:num_of_coordinates-2,:];
        diff_coordinates_mid = diff_coordinates[1:num_of_coordinates-2] # these are bond vectors in the middle (remove the first and last one), they should be perpendicular to adjacent normal vectors

        index = 0
        cos_of_angle = np.dot(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
        sin_of_angle_vec = np.cross(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
        sin_of_angle = sqrt(np.dot(sin_of_angle_vec, sin_of_angle_vec)) * np.sign(sum(sin_of_angle_vec) * sum(diff_coordinates_mid[index]));

        try:
            assert ( cos_of_angle ** 2 + sin_of_angle ** 2 - 1 < 0.0001)  
        except:
            print ("error: cos^2 x+ sin^2 x != 1, it is %f" %(cos_of_angle ** 2 + sin_of_angle ** 2))
            # print ("coordinates of four atoms are:")
            # print (coords_of_four)

        return [cos_of_angle, sin_of_angle]

    @staticmethod
    def get_coordinates_of_atom_with_index(a_coodinate, index):
        """:param a_coodinate is coordinate of all 20 atoms"""
        return [a_coodinate[3 * index], a_coodinate[3 * index + 1], a_coodinate[3 * index + 2]]

    @staticmethod
    def get_cossin_from_a_coordinate(a_coordinate):
        # FIXME: how to write unit test for this function?
        # TODO: to be tested
        total_num_of_residues = 20
        list_of_idx_four_atoms = map(lambda x: [3 * x, 3 * x + 1, 3 * x + 2, 3 * x + 3], range(total_num_of_residues)) \
                               + map(lambda x: [3 * x - 1, 3 * x, 3 * x + 1, 3 * x + 2], range(total_num_of_residues))
        list_of_idx_four_atoms = filter(lambda x: x[0] >= 0 and x[3] < 3 * total_num_of_residues, list_of_idx_four_atoms)

        assert (len(list_of_idx_four_atoms) == 38)

        result = []

        for item in list_of_idx_four_atoms:
            parameter_list = map(
                    lambda x: Trp_cage.get_coordinates_of_atom_with_index(a_coordinate, x),
                    item
                    )
            [cos_value, sin_value] = Trp_cage.get_cossin_of_a_dihedral_from_four_atoms(*parameter_list)
            # print(item)
            # print(cos_value, sin_value)
            result += [cos_value, sin_value]

        return result


    @staticmethod
    def get_many_cossin_from_coordinates(coordinates):
        return map(Trp_cage.get_cossin_from_a_coordinate, coordinates)

    @staticmethod
    def get_many_cossin_from_coordiantes_in_list_of_files(list_of_files):
        result = []
        for item in list_of_files:
            coordinates = np.loadtxt(item)
            temp = Trp_cage.get_many_cossin_from_coordinates(coordinates)
            result += temp

        return result

    @staticmethod
    def get_many_dihedrals_from_coordinates_in_file (list_of_files):
        # why we need to get dihedrals from a list of coordinate files?
        # because we will probably need to plot other files outside self._list_of_coor_data_files
        temp = Trp_cage.get_many_cossin_from_coordiantes_in_list_of_files(list_of_files)
        return Trp_cage.get_many_dihedrals_from_cossin(temp)

    @staticmethod
    def get_many_dihedrals_from_cossin(cossin):
        result = []
        for item in cossin:
            temp_angle = []
            len_of_cos_sin = CONFIG_25
            assert (len(item) == len_of_cos_sin)
            for idx_of_angle in range(len_of_cos_sin / 2):
                temp_angle += [np.arccos(item[2 * idx_of_angle]) * np.sign(item[2 * idx_of_angle + 1])]

            assert (len(temp_angle) == len_of_cos_sin / 2)

            result += [temp_angle]

        assert (len(result) == len(cossin))

        return result

    @staticmethod
    def generate_coordinates_from_pdb_files(folder_for_pdb = CONFIG_12):
        filenames = subprocess.check_output(['find', folder_for_pdb, '-name' ,'*.pdb']).split('\n')[:-1]

        index_of_backbone_atoms = ['1', '2', '3', '17', '18', '19', '36', '37', '38', '57', '58', '59', '76', '77', '78', '93', '94', '95', '117', '118', '119', '136', '137', '138', '158', '159', '160', '170', '171', '172', '177', '178', '179', '184', '185', '186', '198', '199', '200', '209', '210', '211', '220', '221', '222', '227', '228', '229', '251', '252', '253', '265', '266', '267', '279', '280', '281', '293', '294', '295' ]
        assert (len(index_of_backbone_atoms) % 3 == 0)

        for input_file in filenames:
            print ('generating coordinates of ' + input_file)
            output_file = input_file[:-4] + '_coordinates.txt'

            with open(input_file) as f_in:
                with open(output_file, 'w') as f_out:
                    for line in f_in:
                        fields = line.strip().split()
                        if (fields[0] == 'ATOM' and fields[1] in index_of_backbone_atoms):
                            f_out.write(reduce(lambda x,y: x + '\t' + y, fields[6:9]))
                            f_out.write('\t')
                        elif fields[0] == "MODEL" and fields[1] != "1":
                            f_out.write('\n')

                    f_out.write('\n')  # last line
        print("Done generating coordinates files\n")
        return

    @staticmethod
    def get_pairwise_distance_matrices_of_alpha_carbon(list_of_files):
        list_of_files.sort()   # to make the order consistent
        distances_list = []
        for item in list_of_files:
            num_of_residues = 20
            p = PDB.PDBParser()
            structure = p.get_structure('X', item)
            atom_list = [item for item in structure.get_atoms()]
            atom_list = filter(lambda x: x.get_name() == 'CA', atom_list)
            atom_list = zip(*[iter(atom_list)] * num_of_residues)   # reshape the list

            for model in atom_list:
                assert (len(model) == num_of_residues)
                p_distances = np.zeros((num_of_residues, num_of_residues))
                for _1, atom_1 in enumerate(model):
                    for _2, atom_2 in enumerate(model):
                        p_distances[_1][_2] += [atom_1 - atom_2]
                distances_list += [p_distances]

        return np.array(distances_list)

    @staticmethod
    def get_non_repeated_pairwise_distance_as_list_of_alpha_carbon(list_of_files):
        """each element in this result is a list, not a matrix"""
        dis_matrix_list =Trp_cage.get_pairwise_distance_matrices_of_alpha_carbon(list_of_files)
        num_of_residues = 20
        result = []
        for mat in dis_matrix_list:
            p_distances = []
            for item_1 in range(num_of_residues):
                for item_2 in range(item_1 + 1, num_of_residues):
                    p_distances += [mat[item_1][item_2]]
            assert (len(p_distances) == num_of_residues * (num_of_residues - 1 ) / 2)
            result += [p_distances]

        return result

    @staticmethod
    def metric_get_diff_pairwise_distance_matrices_of_alpha_carbon(list_of_files, ref_file ='../resources/1l2y.pdb'):
        ref = Trp_cage.get_pairwise_distance_matrices_of_alpha_carbon([ref_file])
        sample = Trp_cage.get_pairwise_distance_matrices_of_alpha_carbon(list_of_files)
        diff = map(lambda x: np.linalg.norm(ref[0] - x), sample)
        return diff

    @staticmethod
    def metric_get_number_of_native_contacts(list_of_files, ref_file ='../resources/1l2y.pdb', threshold = 8):
        ref = Trp_cage.get_pairwise_distance_matrices_of_alpha_carbon([ref_file])
        sample = Trp_cage.get_pairwise_distance_matrices_of_alpha_carbon(list_of_files)

        result = map(lambda x: sum(sum(((x < threshold) & (ref[0] < threshold)).astype(int))),
                     sample)
        return result

    @staticmethod
    def metric_RMSD_of_atoms(list_of_files, ref_file ='../resources/1l2y.pdb', option = "CA"):
        """
        modified from the code: https://gist.github.com/andersx/6354971
        :param option:  could be either "CA" for alpha-carbon atoms only or "all" for all atoms
        """
        list_of_files.sort()
        pdb_parser = PDB.PDBParser(QUIET=True)
        rmsd_of_all_atoms = []

        ref_structure = pdb_parser.get_structure("reference", ref_file)
        for sample_file in list_of_files:
            sample_structure = pdb_parser.get_structure("sample", sample_file)

            ref_atoms = [item for item in ref_structure[0].get_atoms()]
            if option == "CA":
                ref_atoms = filter(lambda x: x.get_name() == "CA",
                                   ref_atoms)
            elif option == "all":
                pass
            else:
                raise Exception("parameter error: wrong option")

            for sample_model in sample_structure:
                sample_atoms = [item for item in sample_model.get_atoms()]
                if option == "CA":
                    sample_atoms = filter(lambda x: x.get_name() == "CA",
                                          sample_atoms)

                super_imposer = PDB.Superimposer()
                super_imposer.set_atoms(ref_atoms, sample_atoms)
                super_imposer.apply(sample_model.get_atoms())
                rmsd_of_all_atoms.append(super_imposer.rms)

        return rmsd_of_all_atoms

    @staticmethod
    def get_expression_for_input_of_this_molecule():
        index_of_backbone_atoms = ['1', '2', '3', '17', '18', '19', '36', '37', '38', '57', '58', '59', '76', '77', '78', '93', '94', '95', '117', '118', '119', '136', '137', '138', '158', '159', '160', '170', '171', '172', '177', '178', '179', '184', '185', '186', '198', '199', '200', '209', '210', '211', '220', '221', '222', '227', '228', '229', '251', '252', '253', '265', '266', '267', '279', '280', '281', '293', '294', '295' ]
        total_num_of_residues = 20
        list_of_idx_four_atoms = map(lambda x: [3 * x, 3 * x + 1, 3 * x + 2, 3 * x + 3], range(total_num_of_residues)) \
                               + map(lambda x: [3 * x - 1, 3 * x, 3 * x + 1, 3 * x + 2], range(total_num_of_residues))
        list_of_idx_four_atoms = filter(lambda x: x[0] >= 0 and x[3] < 3 * total_num_of_residues, list_of_idx_four_atoms)
        assert (len(list_of_idx_four_atoms) == 38)

        expression_for_input_of_this_molecule = ''
        for index, item in enumerate(list_of_idx_four_atoms):
            index_of_coss = 2 * index
            index_of_sins = 2 * index + 1
            expression_for_input_of_this_molecule += 'out_layer_0_unit_%d = raw_layer_0_unit_%d;\n' % (index_of_coss, index_of_coss)
            expression_for_input_of_this_molecule += 'out_layer_0_unit_%d = raw_layer_0_unit_%d;\n' % (index_of_sins, index_of_sins)
            expression_for_input_of_this_molecule += 'raw_layer_0_unit_%d = cos(dihedral_angle_%d);\n' % (index_of_coss, index)
            expression_for_input_of_this_molecule += 'raw_layer_0_unit_%d = sin(dihedral_angle_%d);\n' % (index_of_sins, index)
            expression_for_input_of_this_molecule += 'dihedral_angle_%d = dihedral(p%s, p%s, p%s, p%s);\n' % (index, 
                                                index_of_backbone_atoms[item[0]], index_of_backbone_atoms[item[1]],
                                                index_of_backbone_atoms[item[2]], index_of_backbone_atoms[item[3]])  # using backbone atoms

        return expression_for_input_of_this_molecule
