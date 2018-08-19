"""Sutils: simulation unilities, some of them are molecule-specific (implemented as methods in subclasses)
"""

from config import *
import random
from coordinates_data_files_list import *
from sklearn.cluster import KMeans
from helper_func import *
from functools import reduce

class Sutils(object):
    def __init__(self):
        return

    @staticmethod
    def get_num_of_non_overlapping_hyperspheres_that_filled_explored_phase_space(
            pdb_file_list, atom_selection, radius, step_interval=1, shuffle_list=True,
            distance_metric='RMSD'):
        """
        This functions is used to count how many non-overlapping hyperspheres are needed to fill the explored phase
        space, to estimate volumn of explored region
        :param atom_selection: atom selection statement for MDAnalysis
        :param radius: radius of hyperspheres
        :param distance_metric: distance metric of two frames
        :return: number of hyperspheres
        """
        if shuffle_list: random.shuffle(pdb_file_list)
        index = 0
        positions_list = []
        for sample_file in pdb_file_list:
            sample = Universe(sample_file)
            sample_atom_selection = sample.select_atoms(atom_selection)
            frame_index_list = list(range(sample.trajectory.n_frames))
            if shuffle_list: random.shuffle(frame_index_list)
            for item_index in frame_index_list:
                sample.trajectory[item_index]
                if index % step_interval == 0:
                    current_positions = sample_atom_selection.positions
                    distances_to_previous_frames = np.array(
                        [Sutils.get_RMSD_after_alignment(item, current_positions)
                            for item in positions_list])
                    if len(distances_to_previous_frames) == 0 or np.all(distances_to_previous_frames > radius):
                        # need to include a new hypershere
                        positions_list.append(current_positions)

                index += 1

        return len(positions_list), np.array(positions_list)

    @staticmethod
    def mark_and_modify_pdb_for_calculating_RMSD_for_plumed(pdb_file, out_pdb,
                                                            atom_index_list, item_positions=None):
        """
        :param pdb_file: input pdb
        :param out_pdb: output reference pdb
        :param atom_index_list: index list used to calculate RMSD
        :param item_positions: reference positions of selected atoms, set it None if we do not want to modify positions
        """
        indices = np.array(atom_index_list) - 1  # because atom_index_list starts with 1
        temp_sample = Universe(pdb_file)
        temp_atoms = temp_sample.select_atoms('all')
        if not item_positions is None:
            item_positions = item_positions.reshape((item_positions.shape[0] / 3, 3))
            temp_positions = temp_atoms.positions
            temp_positions[indices] = item_positions
            temp_atoms.positions = temp_positions

        temp_bfactors = np.zeros(len(temp_atoms))
        temp_bfactors[indices] = 1
        temp_atoms.tempfactors = temp_bfactors
        temp_atoms.occupancies = temp_bfactors
        temp_atoms.write(out_pdb)
        return out_pdb

    @staticmethod
    def get_plumed_script_that_generate_a_segment_connecting_two_configs(
            pdb_1, pdb_2, atom_selection_statement, num_steps, force_constant):
        """
        This function uses targeted MD to generate a segment connecting two configurations
        :param pdb_1, pdb_2: two ends of segment
        :param atom_selection_statement: atoms for calculating RMSD in targeted MD
        """
        atom_list = get_index_list_with_selection_statement(pdb_1, atom_selection_statement)
        ref_pdb = pdb_2.replace('.pdb', '_ref.pdb')
        Sutils.mark_and_modify_pdb_for_calculating_RMSD_for_plumed(pdb_2, ref_pdb, atom_list, None)
        rmsd_diff = Sutils.metric_RMSD_of_atoms([pdb_1], ref_file=ref_pdb,
                                                atom_selection_statement=atom_selection_statement, step_interval=100)[0]  # TODO: check units
        plumed_script = """rmsd: RMSD REFERENCE=%s TYPE=OPTIMAL
restraint: MOVINGRESTRAINT ARG=rmsd AT0=%f STEP0=0 KAPPA0=%f AT1=0 STEP1=%d KAPPA1=%f
PRINT STRIDE=500 ARG=* FILE=COLVAR
""" % (ref_pdb, rmsd_diff, force_constant, num_steps, force_constant)
        return plumed_script

    @staticmethod
    def prepare_output_Cartesian_coor_with_multiple_ref_structures(
             folder_list,
             alignment_coor_file_suffix_list,
             scaling_factor
             ):
        my_coor_data_obj = coordinates_data_files_list(list_of_dir_of_coor_data_files=folder_list)
        coor_data_obj_input = my_coor_data_obj.create_sub_coor_data_files_list_using_filter_conditional(
            lambda x: not 'aligned' in x)
        assert (len(alignment_coor_file_suffix_list) == CONFIG_55)
        coor_data_obj_output_list = [my_coor_data_obj.create_sub_coor_data_files_list_using_filter_conditional(
            lambda x: item in x) for item in alignment_coor_file_suffix_list]

        for item in range(len(alignment_coor_file_suffix_list)):
            for _1, _2 in zip(coor_data_obj_input.get_list_of_coor_data_files(),
                              coor_data_obj_output_list[item].get_list_of_coor_data_files()):
                assert (_2 == _1.replace('_coordinates.txt', alignment_coor_file_suffix_list[item])), (_2, _1)

        output_data_set = np.concatenate([Sutils.remove_translation(item.get_coor_data(scaling_factor))
                                          for item in coor_data_obj_output_list] , axis=1)
        return output_data_set

    @staticmethod
    def select_representative_points(data_set, output_data_set):
        # clustering, pick representative points for training, two purposes:
        # 1. avoid that training results are too good for densely-sampled regions, but bad for others.
        # 2. reduce computation cost
        print ("selecting representative points...")
        kmeans = KMeans(init='k-means++', n_clusters=min(CONFIG_59, output_data_set.shape[0]), n_init=10)
        kmeans.fit(output_data_set)
        indices_of_representative_points = np.array([np.where(kmeans.labels_ == ii)[0][0]
                                                     for ii in range(kmeans.n_clusters)])
        return data_set[indices_of_representative_points], output_data_set[indices_of_representative_points]

    @staticmethod
    def create_subclass_instance_using_name(name):
        return {'Alanine_dipeptide': Alanine_dipeptide(), 'Trp_cage': Trp_cage(),
                'Src_kinase': Src_kinase(), 'BetaHairpin': BetaHairpin(), 'C24': C24()}[name]

    @staticmethod
    def load_object_from_pkl_file(file_path):
        return pickle.load(open(file_path, 'rb'))

    @staticmethod
    def write_some_frames_into_a_new_file_based_on_index_list_for_pdb_file_list(list_of_files, index_list, new_pdb_file_name):
        print("note that order may not be preserved!")
        remaining_index_list = index_list
        for _1 in list_of_files:
            remaining_index_list = Sutils.write_some_frames_into_a_new_file_based_on_index_list(_1, remaining_index_list, new_pdb_file_name)

        # check number of frames to be correct
        with open(new_pdb_file_name, 'r') as f_in:
            content = f_in.read().strip().split('MODEL')[1:]
            assert (len(content) == len(index_list)), (len(content), len(index_list))

        return

    @staticmethod
    def write_some_frames_into_a_new_file_based_on_index_list(pdb_file_name, index_list, new_pdb_file_name=None,
                                                              overwrite=False):
        if os.stat(pdb_file_name).st_size > 1000000000: raise Exception('file may be too large, try to use other tools')

        if new_pdb_file_name is None:
            new_pdb_file_name = pdb_file_name.strip().split('.pdb')[0] + '_someframes.pdb'

        with open(pdb_file_name, 'r') as f_in:
            content = [item for item in f_in.readlines() if (not 'REMARK' in item) and (not 'END\n' in item)]
            content = ''.join(content)
            content = content.split('MODEL')[1:]  # remove header
            num_of_frames_in_current_file = len(content)
            index_for_this_file = [_2 for _2 in index_list if _2 < num_of_frames_in_current_file]
            remaining_index_list = [_2 - num_of_frames_in_current_file for _2 in index_list if
                                    _2 >= num_of_frames_in_current_file]
            content_to_write = [content[_2] for _2 in index_for_this_file]

        write_flag = 'w' if overwrite else 'a'
        with open(new_pdb_file_name, write_flag) as f_out:
            for item in content_to_write:
                f_out.write("MODEL")
                f_out.write(item)

        return remaining_index_list

    @staticmethod
    def concat_first_frame_in_all_pdb_files(list_of_pdb_files, new_pdb_file_name):
        for item in list_of_pdb_files:
            Sutils.write_some_frames_into_a_new_file_based_on_index_list(item, [0], new_pdb_file_name)
        return

    @staticmethod
    def write_some_frames_into_a_new_file(pdb_file_name, start_index, end_index, step_interval = 1,  # start_index included, end_index not included
                                          new_pdb_file_name=None, method=1):
        print(('writing frames of %s: [%d:%d:%d]...' % (pdb_file_name, start_index, end_index, step_interval)))
        if new_pdb_file_name is None:
            new_pdb_file_name = pdb_file_name.strip().split('.pdb')[0] + '_frame_%d_%d_%d.pdb' % (start_index, end_index, step_interval)

        if method == 0:
            if os.stat(pdb_file_name).st_size > 1000000000: raise Exception('file may be too large, try to use other tools')
            with open(pdb_file_name, 'r') as f_in:
                content = [item for item in f_in.readlines() if (not 'REMARK' in item) and (not 'END\n' in item)]
                content = ''.join(content)
                content = content.split('MODEL')[1:]  # remove header
                if end_index == 0:
                    content_to_write = content[start_index::step_interval]     # for selecting last few frames
                else:
                    content_to_write = content[start_index:end_index:step_interval]

            with open(new_pdb_file_name, 'w') as f_out:
                for item in content_to_write:
                    f_out.write("MODEL")
                    f_out.write(item)
        elif method == 1:
            index = -1
            with open(pdb_file_name, 'r') as f_in, open(new_pdb_file_name, 'w') as f_out:
                for item in f_in:
                    if 'MODEL' in item: index += 1
                    if (not 'REMARK' in item) and (not 'END\n' in item) and (index % step_interval == 0) \
                            and (
                            (end_index != 0 and (start_index <= index < end_index))
                            or (end_index == 0 and index >= start_index)):
                        f_out.write(item)
        return

    @staticmethod
    def data_augmentation(data_set, output_data_set, num_of_copies, is_output_reconstructed_Cartesian=True):
        """
        assume that center of mass motion of data_set and output_data_set should be removed.
        """
        assert (Sutils.check_center_of_mass_is_at_origin(data_set))
        if is_output_reconstructed_Cartesian:
            assert (Sutils.check_center_of_mass_is_at_origin(output_data_set))

        num_of_data = data_set.shape[0]
        output_data_set = np.array(output_data_set.tolist() * num_of_copies)
        num_atoms = len(data_set[0]) / 3
        data_set = data_set.reshape((num_of_data, num_atoms, 3))
        temp_data_set = []
        for _ in range(num_of_copies):
            temp_data_set.append([Sutils.rotating_randomly_around_center_of_mass(x) for x in data_set])

        data_set = np.concatenate(temp_data_set, axis=0)
        data_set = data_set.reshape((num_of_copies * num_of_data, num_atoms * 3))
        return data_set, output_data_set

    @staticmethod
    def check_center_of_mass_is_at_origin(result):
        return Helper_func.check_center_of_mass_is_at_origin(result=result)

    @staticmethod
    def remove_translation(coords):   # remove the translational degree of freedom
        return Helper_func.remove_translation(coords=coords)

    @staticmethod
    def rotating_randomly_around_center_of_mass(coords):
        axis_vector = np.random.uniform(0, 1, 3)
        angle = np.random.uniform(0, 2 * np.pi)
        return Sutils.rotating_around_center_of_mass(coords, axis_vector, angle)

    @staticmethod
    def rotating_around_center_of_mass(coords, axis_vector, angle):
        center_of_mass = coords.mean(axis=0)
        return Sutils.rotating_coordinates(coords, center_of_mass, axis_vector, angle)

    @staticmethod
    def rotating_coordinates(coords, fixed_coord, axis_vector, angle):
        indices_atoms = list(range(len(coords)))
        return Sutils.rotating_group_of_atoms(coords, indices_atoms, fixed_coord, axis_vector, angle)

    @staticmethod
    def rotating_group_of_atoms(coords, indices_atoms, fixed_coord, axis_vector, angle):
        """
        :param coords: coordinates of all atoms
        :param indices_atoms: indices of atoms to rotate
        :param fixed_coord: coordinates of fixed point
        :param axis_vector: rotation axis
        :param angle: rotation angle
        :return: coordinates of all atoms after rotation
        """
        result = copy.deepcopy(coords)  # avoid modifying original input
        temp_coords = coords[indices_atoms] - fixed_coord  # coordinates for rotation
        temp_coords = np.array(temp_coords)
        cos_value = np.cos(angle); sin_value = np.sin(angle)
        axis_vector_length = np.sqrt(np.sum(np.array(axis_vector) ** 2))
        ux = axis_vector[0] / axis_vector_length; uy = axis_vector[1] / axis_vector_length; uz = axis_vector[2] / axis_vector_length
        rotation_matrix = np.array([[cos_value + ux ** 2 * (1 - cos_value),
                                     ux * uy * (1 - cos_value) - uz * sin_value,
                                     ux * uz * (1 - cos_value) + uy * sin_value],
                                    [ux * uy * (1 - cos_value) + uz * sin_value,
                                     cos_value + uy ** 2 * (1 - cos_value),
                                     uy * uz * (1 - cos_value) - ux * sin_value],
                                    [ux * uz * (1 - cos_value) - uy * sin_value,
                                     uy * uz * (1 - cos_value) + ux * sin_value,
                                     cos_value + uz ** 2 * (1 - cos_value)]])
        result[indices_atoms] = np.dot(temp_coords, rotation_matrix) + fixed_coord
        return result

    @staticmethod
    def _generate_coordinates_from_pdb_files(index_of_backbone_atoms, path_for_pdb=CONFIG_12, step_interval=1):
        index_of_backbone_atoms = [str(item) for item in index_of_backbone_atoms]
        filenames = subprocess.check_output(['find', path_for_pdb, '-name', '*.pdb']).strip().split('\n')
        output_file_list = []

        for input_file in filenames:
            output_file = input_file.replace('.pdb', '_coordinates.txt')
            if step_interval != 1:
                output_file = output_file.replace('_coordinates.txt', '_int_%d_coordinates.txt' % step_interval)

            output_file_list += [output_file]
            if os.path.exists(output_file) and os.path.getmtime(input_file) < os.path.getmtime(output_file):   # check modified time
                print(("coordinate file already exists: %s (remove previous one if needed)" % output_file))
            else:
                print(('generating coordinates of ' + input_file))

                with open(input_file) as f_in:
                    with open(output_file, 'w') as f_out:
                        # fix based on this format: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
                        # reason for the fix is sometimes there is no space between two neighboring fields
                        # for instance, when atom index is greater than 10000, no space between first two fields, leading to wrong parsing
                        for line in f_in:
                            line = line.strip()
                            fields = [line[:6], line[6:11], line[30:38], line[38:46], line[46:54]]
                            fields = [item_field.strip() for item_field in fields]
                            if (fields[0] == 'ATOM' or fields[0] == 'HETATM') and fields[1] in index_of_backbone_atoms:
                                f_out.write(reduce(lambda x, y: x + '\t' + y, fields[2:5]))
                                f_out.write('\t')
                                if fields[1] == index_of_backbone_atoms[-1]:
                                    f_out.write('\n')

                if step_interval > 1:
                    data = np.loadtxt(output_file)[::step_interval]
                    np.savetxt(output_file, data, fmt="%.3f", delimiter='\t')

        print("Done generating coordinates files\n")
        return output_file_list

    @staticmethod
    def _get_expression_script_for_plumed(index_of_backbone_atoms, scaling_factor):
        plumed_script = ""
        plumed_script += "com_1: COM ATOMS=%s\n" % str(index_of_backbone_atoms)[1:-1].replace(' ', '')
        plumed_script += "p_com: POSITION ATOM=com_1\n"

        for item in range(len(index_of_backbone_atoms)):
            plumed_script += "p_%d: POSITION ATOM=%d\n" % (item, index_of_backbone_atoms[item])
        # following remove translation using p_com
        for item in range(len(index_of_backbone_atoms)):
            for _1, _2 in enumerate(['.x', '.y', '.z']):
                plumed_script += "l_0_out_%d: COMBINE PERIODIC=NO COEFFICIENTS=%f,-%f ARG=p_%d%s,p_com%s\n" \
                        % (3 * item + _1, 10.0 / scaling_factor, 10.0 / scaling_factor,
                            # 10.0 exists because default unit is A in OpenMM, and nm in PLUMED
                            item, _2, _2)
        return plumed_script

    @staticmethod
    def _get_plumed_script_with_pairwise_dis_as_input(index_atoms, scaling_factor):
        plumed_string = ''
        index_input = 0
        for item_1 in range(len(index_atoms)):
            for item_2 in range(item_1 + 1, len(index_atoms)):
                plumed_string += "dis_%d:  DISTANCE ATOMS=%d,%d\n" % (
                    index_input, index_atoms[item_1], index_atoms[item_2])
                plumed_string += "l_0_out_%d: COMBINE PERIODIC=NO COEFFICIENTS=%f ARG=dis_%d\n" % (
                    index_input, 10.0 / scaling_factor, index_input)
                index_input += 1
        return plumed_string

    @staticmethod
    def remove_water_mol_and_Cl_from_pdb_file(folder_for_pdb = CONFIG_12, preserve_original_file=True):
        """
        This is used to remove water molecule from pdb file, purposes:
        - save storage space
        - reduce processing time of pdb file
        """
        filenames = subprocess.check_output(['find', folder_for_pdb, '-name', '*.pdb']).split('\n')[:-1]
        for item in filenames:
            print(('removing water molecules from pdb file: ' + item))
            output_file = item[:-4] + '_rm_tmp.pdb'
            is_line_removed_flag = False
            with open(item, 'r') as f_in, open(output_file, 'w') as f_out:
                for line in f_in:
                    if not 'HOH' in line and not 'CL' in line and not "NA" in line and not 'SPC' in line and not 'pseu' in line:
                        f_out.write(line)
                    else: is_line_removed_flag = True

            if not preserve_original_file:
                if is_line_removed_flag:
                    subprocess.check_output(['mv', output_file, item])
                else:
                    subprocess.check_output(['rm', output_file])

        print('Done removing water molecules from all pdb files!')
        return

    @staticmethod
    def get_boundary_points(list_of_points,
                            range_of_PCs = CONFIG_26,
                            num_of_bins = CONFIG_10,
                            num_of_boundary_points = CONFIG_11,
                            is_circular_boundary = CONFIG_18,
                            preprocessing = True,
                            auto_range_for_histogram = CONFIG_39,   # set the range of histogram based on min,max values in each dimension
                            reverse_sorting_mode = CONFIG_41        # whether we reverse the order of sorting of diff_with_neighbors values
                            ):
        """
        :param preprocessing: if True, then more weight is not linear, this would be better based on experience
        """
        dimensionality = len(list_of_points[0])
        list_of_points = list(zip(*list_of_points))
        assert (len(list_of_points) == dimensionality)

        if is_circular_boundary or not auto_range_for_histogram:
            hist_matrix, edges = np.histogramdd(list_of_points, bins= num_of_bins * np.ones(dimensionality), range = range_of_PCs)
        else:
            temp_hist_range = [[min(item) - (max(item) - min(item)) / (num_of_bins - 2), max(item) + (max(item) - min(item)) / (num_of_bins - 2)]\
                                for item in list_of_points]
            hist_matrix, edges = np.histogramdd(list_of_points, bins=num_of_bins * np.ones(dimensionality), range=temp_hist_range)

        # following is the main algorithm to find boundary and holes
        # simply find the points that are lower than average of its 4 neighbors

        if preprocessing:
            hist_matrix = np.array([[- np.exp(- y) for y in x] for x in hist_matrix])   # preprocessing process

        if is_circular_boundary:  # typically works for circular autoencoder
            diff_with_neighbors = hist_matrix - 1.0 / (2 * dimensionality) \
                                            * sum(
                                                [np.roll(hist_matrix, 1, axis=x) + np.roll(hist_matrix, -1, axis=x) for x in list(range(dimensionality))]
                                                )
        else:
            # TODO: code not concise and general enough, fix this later
            diff_with_neighbors = np.zeros(hist_matrix.shape)
            temp_1 = [list(range(item)) for item in hist_matrix.shape]
            for grid_index in itertools.product(*temp_1):
                neighbor_index_list = [(np.array(grid_index) + temp_2).astype(int) for temp_2 in np.eye(dimensionality)]
                neighbor_index_list += [(np.array(grid_index) - temp_2).astype(int) for temp_2 in np.eye(dimensionality)]
                neighbor_index_list = [x for x in neighbor_index_list if np.all(x >= 0) and np.all(x < num_of_bins)]
                # print "grid_index = %s" % str(grid_index)
                # print "neighbor_index_list = %s" % str(neighbor_index_list)
                diff_with_neighbors[tuple(grid_index)] = hist_matrix[tuple(grid_index)] - np.average(
                    [hist_matrix[tuple(temp_2)] for temp_2 in neighbor_index_list]
                )

        # get grid centers
        edge_centers = [0.5 * (np.array(x[1:]) + np.array(x[:-1])) for x in edges]
        grid_centers = np.array(list(itertools.product(*edge_centers)))  # "itertools.product" gives Cartesian/direct product of several lists
        grid_centers = np.reshape(grid_centers, np.append(num_of_bins * np.ones(dimensionality), dimensionality).astype(int))
        # print grid_centers

        potential_centers = []

        # now sort these grids (that has no points in it)
        # based on total number of points in its neighbors

        temp_seperate_index = []

        for _ in range(dimensionality):
            temp_seperate_index.append(list(range(num_of_bins)))

        index_of_grids = list(itertools.product(
                        *temp_seperate_index
                        ))

        index_of_grids =  [x for x in index_of_grids if diff_with_neighbors[x] < 0]     # only apply to grids with diff_with_neighbors value < 0
        sorted_index_of_grids = sorted(index_of_grids, key = lambda x: diff_with_neighbors[x]) # sort based on histogram, return index values
        if reverse_sorting_mode:
            sorted_index_of_grids.reverse()

        for index in sorted_index_of_grids[:num_of_boundary_points]:  # note index can be of dimension >= 2
            temp_potential_center = [round(x, 2) for x in grid_centers[index]]
            potential_centers.append(temp_potential_center)

        return potential_centers

    @staticmethod
    def L_method(evaluation_values, num):
        evaluation_values = np.array(evaluation_values)
        num = np.array(num)
        assert (evaluation_values.shape == num.shape)
        min_weighted_err = float('inf')
        optimal_num = 0
        best_regr = None
        for item in range(1, len(num) - 1):
            y_left = evaluation_values[:item]
            x_left = num[:item].reshape(item, 1)
            y_right = evaluation_values[item - 1:]
            x_right = num[item - 1:].reshape(len(num) - item + 1, 1)
            regr_left = linear_model.LinearRegression()
            regr_left.fit(x_left, y_left)
            y_left_pred = regr_left.predict(x_left)
            regr_right = linear_model.LinearRegression()
            regr_right.fit(x_right, y_right)
            y_right_pred = regr_right.predict(x_right)

            err_left = mean_squared_error(y_left, y_left_pred)
            err_right = mean_squared_error(y_right, y_right_pred)
            weighted_err = (err_left * item + err_right * (len(num) - item + 1)) / (len(num) + 1)
            if weighted_err < min_weighted_err:
                optimal_num = num[item]
                min_weighted_err = weighted_err
                best_regr = [regr_left, regr_right]

        x_data = np.linspace(min(num), max(num), 100).reshape(100, 1)
        y_data_left = best_regr[0].predict(x_data)
        y_data_right = best_regr[1].predict(x_data)

        return optimal_num, x_data, y_data_left, y_data_right

    @staticmethod
    def get_RMSD_after_alignment(position_1, position_2):
        return rmsd(position_1, position_2, center=True, superposition=True)

    @staticmethod
    def metric_RMSD_of_atoms(list_of_files, ref_file='../resources/1l2y.pdb', ref_index=0,
                             atom_selection_statement="name CA", step_interval=1):
        """
        :param atom_selection_statement:  could be either
         - "name CA" for alpha-carbon atoms only
         - "protein" for all atoms
         - "backbone" for backbone atoms
         - others: see more information here: https://pythonhosted.org/MDAnalysis/documentation_pages/selections.html
        """
        ref = Universe(ref_file)
        ref_atom_selection = ref.select_atoms(atom_selection_statement)
        ref.trajectory[ref_index]
        ref_positions = ref_atom_selection.positions
        result_rmsd_of_atoms = []
        index = 0

        for sample_file in list_of_files:
            sample = Universe(sample_file)
            sample_atom_selection = sample.select_atoms(atom_selection_statement)

            for _ in sample.trajectory:
                if index % step_interval == 0:
                    result_rmsd_of_atoms.append(Sutils.get_RMSD_after_alignment(ref_positions,
                                                                    sample_atom_selection.positions))

                index += 1
        return np.array(result_rmsd_of_atoms)

    @staticmethod
    def get_positions_from_list_of_pdb(pdb_file_list, atom_selection_statement='name CA'):
        positions = []
        for sample_file in pdb_file_list:
            sample = Universe(sample_file)
            sample_atom_selection = sample.select_atoms(atom_selection_statement)
            for _ in sample.trajectory:
                positions.append(sample_atom_selection.positions)
        return positions

    @staticmethod
    def get_RMSD_of_a_point_wrt_neighbors_in_PC_space_with_list_of_pdb(PCs, pdb_file_list, radius=0.1):
        """This function calculates RMSD of a configuration with respect to its neighbors in PC space,
        the purpose is to see if similar structures (small RMSD) are projected to points close to each other
        in PC space.
        wrt = with respect to
        """
        from sklearn.metrics.pairwise import euclidean_distances
        positions = Sutils.get_positions_from_list_of_pdb(pdb_file_list)
        pairwise_dis_in_PC = euclidean_distances(PCs)
        neighbor_matrix = pairwise_dis_in_PC < radius
        RMSD_diff_of_neighbors = np.zeros(neighbor_matrix.shape)
        for ii in range(len(PCs)):
            for jj in range(ii + 1, len(PCs)):
                if neighbor_matrix[ii][jj]:
                    RMSD_diff_of_neighbors[ii, jj] = RMSD_diff_of_neighbors[jj, ii] \
                        = Sutils.get_RMSD_after_alignment(positions[ii], positions[jj])
        average_RMSD_wrt_neighbors = [np.average([x for x in RMSD_diff_of_neighbors[ii] if x])
                                      for ii in range(len(PCs))]
        return average_RMSD_wrt_neighbors

    @staticmethod
    def get_pairwise_distance_matrices_of_selected_atoms(list_of_files, step_interval=1, atom_selection='name CA'):
        distances_list = []
        index = 0
        for sample_file in list_of_files:
            sample = Universe(sample_file)
            sample_atom_selection = sample.select_atoms(atom_selection)
            for _ in sample.trajectory:
                if index % step_interval == 0:
                    distances_list.append(
                        distance_array(sample_atom_selection.positions, sample_atom_selection.positions))

                index += 1

        return np.array(distances_list)

    @staticmethod
    def get_non_repeated_pairwise_distance(list_of_files, step_interval=1, atom_selection='name CA'):
        """each element in this result is a list, not a matrix"""
        dis_matrix_list = Sutils.get_pairwise_distance_matrices_of_selected_atoms(list_of_files, step_interval,
                                                                                  atom_selection)
        num_atoms = dis_matrix_list[0].shape[0]
        result = []
        for mat in dis_matrix_list:
            p_distances = []
            for item_1 in range(num_atoms):
                for item_2 in range(item_1 + 1, num_atoms):
                    p_distances += [mat[item_1][item_2]]
            assert (len(p_distances) == num_atoms * (num_atoms - 1) / 2)
            result += [p_distances]

        return np.array(result)

    @staticmethod
    def get_residue_relative_position_list(sample_file):
        sample = Universe(sample_file)
        temp_heavy_atoms = sample.select_atoms('not name H*')
        temp_CA_atoms = sample.select_atoms('name CA')
        residue_relative_position_list = []

        for _ in sample.trajectory:
            temp_residue_relative_position_list = []
            for temp_residue_index in sample.residues.resnums:
                temp_residue_relative_position_list.append(
                    temp_heavy_atoms[temp_heavy_atoms.resnums == temp_residue_index].positions \
                    - temp_CA_atoms[temp_CA_atoms.resnums == temp_residue_index].positions)
            residue_relative_position_list.append(temp_residue_relative_position_list)
        return residue_relative_position_list


class Alanine_dipeptide(Sutils):
    """docstring for Alanine_dipeptide"""
    def __init__(self):
        super(Alanine_dipeptide, self).__init__()
        return

    @staticmethod
    def get_cossin_from_a_coordinate(a_coordinate):
        num_of_coordinates = len(list(a_coordinate)) / 3
        a_coordinate = np.array(a_coordinate).reshape(num_of_coordinates, 3)
        diff_coordinates = a_coordinate[1:num_of_coordinates, :] - a_coordinate[0:num_of_coordinates - 1,:]  # bond vectors
        diff_coordinates_1=diff_coordinates[0:num_of_coordinates-2,:];diff_coordinates_2=diff_coordinates[1:num_of_coordinates-1,:]
        normal_vectors = np.cross(diff_coordinates_1, diff_coordinates_2)
        normal_vectors_normalized = np.array([x / sqrt(np.dot(x,x)) for x in normal_vectors])
        normal_vectors_normalized_1 = normal_vectors_normalized[0:num_of_coordinates-3, :]; normal_vectors_normalized_2 = normal_vectors_normalized[1:num_of_coordinates-2,:]
        diff_coordinates_mid = diff_coordinates[1:num_of_coordinates-2] # these are bond vectors in the middle (remove the first and last one), they should be perpendicular to adjacent normal vectors

        cos_of_angles = list(range(len(normal_vectors_normalized_1)))
        sin_of_angles_vec = list(range(len(normal_vectors_normalized_1)))
        sin_of_angles = list(range(len(normal_vectors_normalized_1))) # initialization
        result = []

        for index in range(len(normal_vectors_normalized_1)):
            cos_of_angles[index] = np.dot(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
            sin_of_angles_vec[index] = np.cross(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
            sin_of_angles[index] = sqrt(np.dot(sin_of_angles_vec[index], sin_of_angles_vec[index])) * np.sign(sum(sin_of_angles_vec[index]) * sum(diff_coordinates_mid[index]))
            result += [cos_of_angles[index], sin_of_angles[index]]

        return result

    @staticmethod
    def get_many_cossin_from_coordinates(coordinates):
        return list(map(Alanine_dipeptide.get_cossin_from_a_coordinate, coordinates))

    @staticmethod
    def get_many_cossin_from_coordinates_in_list_of_files(list_of_files, step_interval=1):
        coordinates = []
        for item in list_of_files:
            temp_coordinates = np.loadtxt(item)  # the result could be 1D or 2D numpy array, need further checking
            if temp_coordinates.shape[0] != 0:  # remove info from empty files
                if len(temp_coordinates.shape) == 1:  # if 1D numpy array, convert it to 2D array for consistency
                    temp_coordinates = temp_coordinates[:, None].T

                coordinates += list(temp_coordinates)

        coordinates = coordinates[::step_interval]
        result = Alanine_dipeptide.get_many_cossin_from_coordinates(coordinates)

        return result

    @staticmethod
    def get_many_dihedrals_from_coordinates_in_file (list_of_files):
        # why we need to get dihedrals from a list of coordinate files?
        # because we will probably need to plot other files outside self._list_of_coor_data_files
        temp = Alanine_dipeptide.get_many_cossin_from_coordinates_in_list_of_files(list_of_files)
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
    def generate_coordinates_from_pdb_files(path_for_pdb=CONFIG_12, step_interval =1):
        index_of_backbone_atoms = [str(item) for item in CONFIG_57[0]]
        output_file_list = Sutils._generate_coordinates_from_pdb_files(index_of_backbone_atoms, path_for_pdb=path_for_pdb, step_interval=step_interval)
        return output_file_list

    @staticmethod
    def get_expression_script_for_plumed(scaling_factor=CONFIG_49):
        index_of_backbone_atoms = CONFIG_57[0]
        return Sutils._get_expression_script_for_plumed(index_of_backbone_atoms, scaling_factor)


class Trp_cage(Sutils):
    """docstring for Trp_cage"""
    def __init__(self):
        super(Trp_cage, self).__init__()
        return

    @staticmethod
    def get_cossin_of_a_dihedral_from_four_atoms(coord_1, coord_2, coord_3, coord_4):
        """each parameter is a 3D Cartesian coordinates of an atom"""
        coords_of_four = np.array([coord_1, coord_2, coord_3, coord_4])
        num_of_coordinates = 4
        diff_coordinates = coords_of_four[1:num_of_coordinates, :] - coords_of_four[0:num_of_coordinates - 1,:]  # bond vectors
        diff_coordinates_1=diff_coordinates[0:num_of_coordinates-2,:];diff_coordinates_2=diff_coordinates[1:num_of_coordinates-1,:]
        normal_vectors = np.cross(diff_coordinates_1, diff_coordinates_2)
        normal_vectors_normalized = np.array([x / sqrt(np.dot(x,x)) for x in normal_vectors])
        normal_vectors_normalized_1 = normal_vectors_normalized[0:num_of_coordinates-3, :]; normal_vectors_normalized_2 = normal_vectors_normalized[1:num_of_coordinates-2,:]
        diff_coordinates_mid = diff_coordinates[1:num_of_coordinates-2] # these are bond vectors in the middle (remove the first and last one), they should be perpendicular to adjacent normal vectors

        index = 0
        cos_of_angle = np.dot(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
        sin_of_angle_vec = np.cross(normal_vectors_normalized_1[index], normal_vectors_normalized_2[index])
        if sin_of_angle_vec[0] != 0 and diff_coordinates_mid[index][0] != 0:
            component_index = 0
        elif sin_of_angle_vec[1] != 0 and diff_coordinates_mid[index][1] != 0:
            component_index = 1
        else:
            component_index = 2

        sin_of_angle = sqrt(np.dot(sin_of_angle_vec, sin_of_angle_vec)) * np.sign(sin_of_angle_vec[component_index] * diff_coordinates_mid[index][component_index])
        try:
            assert ( cos_of_angle ** 2 + sin_of_angle ** 2 - 1 < 0.0001)
        except:
            print(("error: cos^2 x+ sin^2 x != 1, it is %f" %(cos_of_angle ** 2 + sin_of_angle ** 2)))
            # print ("coordinates of four atoms are:")
            # print (coords_of_four)

        return [cos_of_angle, sin_of_angle]

    @staticmethod
    def get_coordinates_of_atom_with_index(a_coodinate, index):
        """:param a_coodinate is coordinate of all 20 atoms"""
        return [a_coodinate[3 * index], a_coodinate[3 * index + 1], a_coodinate[3 * index + 2]]

    @staticmethod
    def get_cossin_from_a_coordinate(a_coordinate):
        total_num_of_residues = 20
        list_of_idx_four_atoms = [[[3 * x - 1, 3 * x, 3 * x + 1, 3 * x + 2],
                                                [3 * x, 3 * x + 1, 3 * x + 2, 3 * x + 3]] for x in list(range(total_num_of_residues))]
        list_of_idx_four_atoms = reduce(lambda x, y: x + y, list_of_idx_four_atoms)
        list_of_idx_four_atoms = [x for x in list_of_idx_four_atoms if x[0] >= 0 and x[3] < 3 * total_num_of_residues]

        assert (len(list_of_idx_four_atoms) == 38)

        result = []

        for item in list_of_idx_four_atoms:
            parameter_list = [Trp_cage.get_coordinates_of_atom_with_index(a_coordinate, x) for x in item]
            [cos_value, sin_value] = Trp_cage.get_cossin_of_a_dihedral_from_four_atoms(*parameter_list)
            # print(item)
            # print(cos_value, sin_value)
            result += [cos_value, sin_value]

        return result

    @staticmethod
    def get_many_cossin_from_coordinates(coordinates):
        return list(map(Trp_cage.get_cossin_from_a_coordinate, coordinates))

    @staticmethod
    def get_many_cossin_from_coordinates_in_list_of_files(list_of_files, step_interval=1):
        coordinates = []
        for item in list_of_files:
            temp_coordinates = np.loadtxt(item)  # the result could be 1D or 2D numpy array, need further checking
            if temp_coordinates.shape[0] != 0:        # remove info from empty files
                if len(temp_coordinates.shape) == 1:  # if 1D numpy array, convert it to 2D array for consistency
                    temp_coordinates = temp_coordinates[:, None].T

                coordinates += list(temp_coordinates)

        coordinates = coordinates[::step_interval]
        result = Trp_cage.get_many_cossin_from_coordinates(coordinates)

        return result

    @staticmethod
    def get_many_dihedrals_from_coordinates_in_file (list_of_files, step_interval=1):
        # why we need to get dihedrals from a list of coordinate files?
        # because we will probably need to plot other files outside self._list_of_coor_data_files
        temp = Trp_cage.get_many_cossin_from_coordinates_in_list_of_files(list_of_files, step_interval)
        return Trp_cage.get_many_dihedrals_from_cossin(temp)

    @staticmethod
    def get_many_dihedrals_from_cossin(cossin):
        result = []
        for item in cossin:
            temp_angle = []
            len_of_cos_sin = 76
            assert (len(item) == len_of_cos_sin), (len(item), len_of_cos_sin)
            for idx_of_angle in range(len_of_cos_sin / 2):
                temp_angle += [np.arccos(item[2 * idx_of_angle]) * np.sign(item[2 * idx_of_angle + 1])]

            assert (len(temp_angle) == len_of_cos_sin / 2)

            result += [temp_angle]

        assert (len(result) == len(cossin))

        return result

    @staticmethod
    def generate_coordinates_from_pdb_files(path_for_pdb = CONFIG_12, step_interval=1):
        index_of_backbone_atoms = [str(item) for item in CONFIG_57[1]]
        assert (len(index_of_backbone_atoms) % 3 == 0)

        output_file_list = Sutils._generate_coordinates_from_pdb_files(index_of_backbone_atoms, path_for_pdb=path_for_pdb,
                                                                       step_interval=step_interval)

        return output_file_list

    @staticmethod
    def metric_get_diff_pairwise_distance_matrices_of_alpha_carbon(list_of_files, ref_file ='../resources/1l2y.pdb', step_interval = 1):
        ref = Trp_cage.get_pairwise_distance_matrices_of_selected_atoms([ref_file])
        sample = Trp_cage.get_pairwise_distance_matrices_of_selected_atoms(list_of_files, step_interval)
        diff = [np.linalg.norm(ref[0] - x) for x in sample]
        return diff

    @staticmethod
    def metric_get_residue_9_16_salt_bridge_distance(list_of_files, step_interval = 1):
        distances_list = []
        index = 0
        for sample_file in list_of_files:
            sample = Universe(sample_file)
            sample_atom_selection_1 = sample.select_atoms("name OD2 and resid 9")
            sample_atom_selection_2 = sample.select_atoms("name NH2 and resid 16")
            for _ in sample.trajectory:
                if index % step_interval == 0:
                    distances_list.append(
                        distance_array(sample_atom_selection_1.positions, sample_atom_selection_2.positions))

                index += 1

        return np.array(distances_list).flatten()

    @staticmethod
    def metric_chirality(list_of_files, step_interval=1):
        result = []
        index = 0
        for temp_file in list_of_files:
            temp_universe = Universe(temp_file)
            for _ in temp_universe.trajectory:
                if index % step_interval == 0:
                    atom_list = [temp_universe.select_atoms('name CA and resid %d' % item).positions[0]
                                 for item in [1, 9, 14, 20]]
                    result.append(Trp_cage.get_cossin_of_a_dihedral_from_four_atoms(
                        atom_list[0], atom_list[1], atom_list[2], atom_list[3])[1])
                index += 1
        return np.array(result)

    @staticmethod
    def metric_vertical_shift(list_of_files, step_interval=1):
        result = []
        index = 0
        for temp_file in list_of_files:
            temp_universe = Universe(temp_file)
            for _ in temp_universe.trajectory:
                if index % step_interval == 0:
                    atom_list = [temp_universe.select_atoms('name CA and resid %d' % item).positions[0]
                                 for item in [1, 11, 20]]
                    result.append(np.linalg.norm(atom_list[0] - atom_list[1]) - np.linalg.norm(atom_list[2] - atom_list[1]))
                index += 1
        return np.array(result)

    @staticmethod
    def metric_get_number_of_native_contacts(list_of_files, ref_file ='../resources/1l2y.pdb', threshold = 8, step_interval = 1):
        ref = Trp_cage.get_pairwise_distance_matrices_of_selected_atoms([ref_file])
        sample = Trp_cage.get_pairwise_distance_matrices_of_selected_atoms(list_of_files, step_interval)

        result = [sum(sum(((x < threshold) & (ref[0] < threshold)).astype(int))) for x in sample]
        return result

    @staticmethod
    def metric_radius_of_gyration(list_of_files, step_interval = 1, atom_selection_statement = "name CA"):
        result = []
        index = 0
        for item_file in list_of_files:
            temp_sample = Universe(item_file)
            temp_atoms = temp_sample.select_atoms(atom_selection_statement)
            for _ in temp_sample.trajectory:
                if index % step_interval == 0:
                    result.append(temp_atoms.radius_of_gyration())
                index += 1

        return result

    @staticmethod
    def get_pairwise_RMSD_after_alignment_for_a_file(sample_file, atom_selection_statement = 'name CA'):
        sample_1 = Universe(sample_file); sample_2 = Universe(sample_file)    # should use two variables here, otherwise it will be 0, might be related to iterator issue?
        sel_1 = sample_1.select_atoms(atom_selection_statement); sel_2 = sample_2.select_atoms(atom_selection_statement)

        return [[rmsd(sel_1.positions, sel_2.positions, center=True, superposition=True) for _2 in sample_2.trajectory] for _1 in sample_1.trajectory]

    @staticmethod
    def structure_clustering_in_a_file(sample_file, atom_selection_statement = 'name CA',
                                       write_most_common_class_into_file = False,
                                       output_file_name = None,
                                       eps=0.5,
                                       min_num_of_neighboring_samples = 2
                                       ):
        pairwise_RMSD = Trp_cage.get_pairwise_RMSD_after_alignment_for_a_file(sample_file, atom_selection_statement=atom_selection_statement)
        from sklearn.cluster import DBSCAN

        dbscan_obj = DBSCAN(metric='precomputed', eps=eps, min_samples=min_num_of_neighboring_samples).fit(pairwise_RMSD)
        class_labels = dbscan_obj.labels_
        max_class_label = max(class_labels)
        num_in_each_class = {label: np.where(class_labels == label)[0].shape[0] for label in range(-1, max_class_label + 1)}
        most_common_class_labels = sorted(list(num_in_each_class.keys()), key=lambda x: num_in_each_class[x], reverse=True)
        with open(sample_file, 'r') as in_file:
            content = [item for item in in_file.readlines() if not 'REMARK' in item]
            content = ''.join(content)
            content = content.split('MODEL')[1:]  # remove header
            assert (len(content) == len(class_labels))

        if most_common_class_labels[0] == -1:
            raise Exception("too many outliers, check if there is actually a cluster, or adjust parameters")
        else:
            index_of_most_common_class = np.where(class_labels == most_common_class_labels[0])[0]
            if write_most_common_class_into_file:
                if output_file_name is None:
                    output_file_name = sample_file.replace('.pdb', '_most_common.pdb')

                frames_to_use = [content[ii] for ii in index_of_most_common_class]
                with open(output_file_name, 'w') as out_file:
                    for frame in frames_to_use:
                        out_file.write("MODEL" + frame)

        return num_in_each_class, index_of_most_common_class, most_common_class_labels[0]

    @staticmethod
    def rotating_dihedral_angles_and_save_to_pdb(input_pdb, target_dihedrals, output_pdb):
        pdb_parser = PDB.PDBParser(QUIET=True)
        temp_structure = pdb_parser.get_structure('temp', input_pdb)
        coor_file = Trp_cage.generate_coordinates_from_pdb_files(input_pdb)[0]
        current_dihedrals = Trp_cage.get_many_dihedrals_from_coordinates_in_file([coor_file])
        rotation_angles = np.array(target_dihedrals) - np.array(current_dihedrals)

        atom_indices_in_each_residue = [[]] * 20
        temp_model = list(temp_structure.get_models())[0]
        for _1, item in list(enumerate(temp_model.get_residues())):
            atom_indices_in_each_residue[_1] = [int(_2.get_serial_number()) - 1 for _2 in item.get_atoms()]

        for temp_model in temp_structure.get_models():
            atoms_in_this_frame = list(temp_model.get_atoms())
            temp_coords = np.array([_1.get_coord() for _1 in atoms_in_this_frame])

            for item in range(19):  # 19 * 2 = 38 dihedrals in total
                C_atom_in_this_residue = filter(lambda x: x.get_name() == "C", atoms_in_this_frame)[item]
                CA_atom_in_this_residue = filter(lambda x: x.get_name() == "CA", atoms_in_this_frame)[item]
                CA_atom_in_next_residue = filter(lambda x: x.get_name() == "CA", atoms_in_this_frame)[item + 1]
                N_atom_in_next_residue = filter(lambda x: x.get_name() == "N", atoms_in_this_frame)[item + 1]

                axis_vector_0 = C_atom_in_this_residue.get_coord() - CA_atom_in_this_residue.get_coord()
                axis_vector_1 = CA_atom_in_next_residue.get_coord() - N_atom_in_next_residue.get_coord()

                fixed_coord_0 = temp_coords[int(C_atom_in_this_residue.get_serial_number()) - 1]
                fixed_coord_1 = temp_coords[int(N_atom_in_next_residue.get_serial_number()) - 1]

                indices_atom_to_rotate = reduce(lambda x, y: x + y, atom_indices_in_each_residue[:item + 1])

                temp_coords = Sutils.rotating_group_of_atoms(temp_coords, indices_atom_to_rotate, fixed_coord_0,
                                                             axis_vector_0, rotation_angles[temp_model.get_id()][2 * item])
                temp_coords = Sutils.rotating_group_of_atoms(temp_coords, indices_atom_to_rotate, fixed_coord_1,
                                                             axis_vector_1, rotation_angles[temp_model.get_id()][2 * item + 1])

            # save coordinates into structure
            for _1, item in enumerate(temp_model.get_atoms()):
                item.set_coord(temp_coords[_1])

        io = PDB.PDBIO()
        io.set_structure(temp_structure)
        io.save(output_pdb)
        return

    @staticmethod
    def get_expression_script_for_plumed(scaling_factor=CONFIG_49):
        index_of_backbone_atoms = CONFIG_57[1]
        return Sutils._get_expression_script_for_plumed(index_of_backbone_atoms, scaling_factor)


class Src_kinase(Sutils):
    def __init__(self):
        super(Src_kinase, self).__init__()
        return

    @staticmethod
    def generate_coordinates_from_pdb_files(path_for_pdb=CONFIG_12, step_interval=1):
        index_of_backbone_atoms = [str(item) for item in CONFIG_57[2]]
        output_file_list = Sutils._generate_coordinates_from_pdb_files(
            index_of_backbone_atoms, path_for_pdb=path_for_pdb, step_interval=step_interval)
        return output_file_list

    @staticmethod
    def get_expression_script_for_plumed(scaling_factor=CONFIG_49):
        index_of_backbone_atoms = CONFIG_57[2]
        return Sutils._get_expression_script_for_plumed(index_of_backbone_atoms, scaling_factor)

    @staticmethod
    def metric_salt_bridge_switching(list_of_files, step_interval=1):
        """this function measures (distance between E310-R409) - (distance between E310-K295),
        this is a good metric showing switching between two salt bridges
        """

        dis_310_409 = []
        dis_310_295 = []
        index = 0
        for sample_file in list_of_files:
            sample = Universe(sample_file)
            # note that we only simulate residue 260-521
            temp_atom_310 = sample.select_atoms('resid 51 and name CD')
            temp_atom_409 = sample.select_atoms('resid 150 and name CZ')
            temp_atom_295 = sample.select_atoms('resid 36 and name NZ')
            for _ in sample.trajectory:
                if index % step_interval == 0:
                    dis_310_409.append(
                        distance_array(temp_atom_310.positions, temp_atom_409.positions))
                    dis_310_295.append(
                        distance_array(temp_atom_310.positions, temp_atom_295.positions))

                index += 1

        return np.array(dis_310_409).flatten() - np.array(dis_310_295).flatten()

class BetaHairpin(Sutils):
    def __init__(self):
        super(BetaHairpin, self).__init__()
        return

    @staticmethod
    def generate_coordinates_from_pdb_files(path_for_pdb=CONFIG_12, step_interval=1):
        index_of_backbone_atoms = [str(item) for item in CONFIG_57[3]]
        output_file_list = Sutils._generate_coordinates_from_pdb_files(index_of_backbone_atoms, path_for_pdb=path_for_pdb,
                                                                       step_interval=step_interval)

        return output_file_list

class C24(Sutils):
    def __init__(self):
        super(C24, self).__init__()
        return

    @staticmethod
    def generate_coordinates_from_pdb_files(path_for_pdb=CONFIG_12, step_interval=1):
        index_of_backbone_atoms = [str(item) for item in CONFIG_57[4]]
        output_file_list = Sutils._generate_coordinates_from_pdb_files(index_of_backbone_atoms,
                                                                       path_for_pdb=path_for_pdb,
                                                                       step_interval=step_interval,
                                                                       start_index_of_xyz_field=5)
        return output_file_list
