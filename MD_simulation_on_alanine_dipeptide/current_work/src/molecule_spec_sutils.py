"""Sutils: simulation unilities, some of them are molecule-specific (implemented as methods in subclasses)
"""

import copy, pickle, re, os, time, subprocess, datetime, itertools
from config import *

class Sutils(object):
    def __init__(self):
        return

    @staticmethod
    def remove_water_mol_from_pdb_file(folder_for_pdb = CONFIG_12, preserve_original_file=False):
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
                    if not 'HOH' in line:
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
                            dimensionality = CONFIG_3[2]
                            ):
        '''
        :param preprocessing: if True, then more weight is not linear, this would be better based on experience
        '''

        list_of_points = zip(*list_of_points)
        hist_matrix, edges = np.histogramdd(list_of_points, bins= num_of_bins * np.ones(dimensionality), range = range_of_PCs)

        # following is the main algorithm to find boundary and holes
        # simply find the points that are lower than average of its 4 neighbors

        if preprocessing:
            hist_matrix = map(lambda x: map(lambda y: - np.exp(- y), x), hist_matrix)   # preprocessing process

        if is_circular_boundary:  # typically works for circular autoencoder
            diff_with_neighbors = hist_matrix - 1 / (2 * dimensionality) \
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
    def get_coordinates_of_alpha_carbon_from_a_file(file_name, file_type="pdb"):
        """return a list of coordinates, each element in list corresponds to a snapshot of the configuration"""
        result = []
        if file_type == 'pdb':
            index_of_alpha_carbons = [2, 18, 37, 58, 77, 94, 118, 137, 159, 171, 178, 185, 199, 210, 221, 228, 252, 266, 280, 294]
            index_of_alpha_carbons = map(lambda x: str(x), index_of_alpha_carbons)

            temp_coordinate = np.zeros((20, 3))
            temp_index = 0
            with open(file_name) as f_in:
                for line in f_in:
                    fields = line.strip().split()
                    if (fields[0] == 'ATOM' and fields[1] in index_of_alpha_carbons):
                        temp_coordinate[temp_index] = np.array([float(fields[6]), float(fields[7]), float(fields[8])])
                        temp_index += 1
                    elif fields[0] == "MODEL" and fields[1] != "1":
                        temp_index = 0
                        result.append(temp_coordinate.copy())
        elif file_type == 'coordinates_txt':  # this is the coordinates file generated by generate_coordinates_from_pdb_files()
            all_coordinates = np.loadtxt(file_name)
            temp_index_of_carbon = filter(lambda x: x % 3 == 1, range(60))
            for coor in all_coordinates:
                temp_coordinate = np.array(map(lambda x: Trp_cage.get_coordinates_of_atom_with_index(coor, x), 
                                               temp_index_of_carbon))
                result.append(temp_coordinate.copy())

        return result

    @staticmethod
    def get_distance_matrix_of_alpha_carbon(coor_of_alpha_carbon):
        """
        :param coor_of_alpha_carbon: 2d-numpy-array
        """
        num = len(coor_of_alpha_carbon)
        distance = np.zeros((num, num))
        for idx_1, x in enumerate(coor_of_alpha_carbon):
            for idx_2, y in enumerate(coor_of_alpha_carbon):
                distance[idx_1][idx_2] = np.linalg.norm(x-y)
        return distance

    @staticmethod
    def get_distance_between_two_coordinates(coor_1, coor_2):
        mat_1 = Trp_cage.get_distance_matrix_of_alpha_carbon(coor_of_alpha_carbon=coor_1)
        mat_2 = Trp_cage.get_distance_matrix_of_alpha_carbon(coor_of_alpha_carbon=coor_2)
        return np.linalg.norm(mat_1 - mat_2)

    @staticmethod
    def get_number_of_native_contacts(coor_1, coor_2, threshold = 8):
        mat_1 = Trp_cage.get_distance_matrix_of_alpha_carbon(coor_of_alpha_carbon=coor_1)
        mat_2 = Trp_cage.get_distance_matrix_of_alpha_carbon(coor_of_alpha_carbon=coor_2)
        print (mat_1 < threshold).astype(int)
        print (mat_2 < threshold).astype(int)
        result = sum(sum(((mat_1 < threshold) & (mat_2 < threshold)).astype(int)))
        return result

    @staticmethod
    def get_list_of_distances_between_coordinates_in_one_file_and_coord_of_folded_state(
                                                            distance_function,
                                                            file_name, file_type,
                                                            pdb_file_of_folded_state = '../resources/1l2y.pdb'):
        """
        :param distance_function: could be distance between matrix, or number of native contacts, etc.
        """
        coor_of_folded = Trp_cage.get_coordinates_of_alpha_carbon_from_a_file(pdb_file_of_folded_state, file_type='pdb')[0]
        list_of_coordinates = Trp_cage.get_coordinates_of_alpha_carbon_from_a_file(file_name=file_name, file_type=file_type)
        result = map(lambda x: distance_function(coor_of_folded, x), list_of_coordinates)
        return result

    @staticmethod
    def get_list_of_distances_between_coordinates_in_many_files_and_coord_of_folded_state(
                                                                distance_function,
                                                                list_of_file_names, file_type,
                                                                pdb_file_of_folded_state='../resources/1l2y.pdb'):
        result = []
        for item in list_of_file_names:
            result += Trp_cage.get_list_of_distances_between_coordinates_in_one_file_and_coord_of_folded_state(
                distance_function,
                item, file_type, pdb_file_of_folded_state
            )
        return result

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
