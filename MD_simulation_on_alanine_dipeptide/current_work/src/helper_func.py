from config import *
from scipy.special import erf

class Helper_func(object):
    def __init__(self):
        return

    @staticmethod
    def get_mutual_info_of_two_continuous_vars(temp_var_0, temp_var_1, bins=10, normalization=True):
        temp_hist_0, _ = np.histogramdd(temp_var_0, bins=bins)
        temp_hist_1, _ = np.histogramdd(temp_var_1, bins=bins)
        temp_hist_2, _ = np.histogramdd(np.array([temp_var_0, temp_var_1]).T, bins=bins)
        temp_hist_0 /= temp_hist_0.sum()
        temp_hist_1 /= temp_hist_1.sum()
        temp_hist_2 /= temp_hist_2.sum()
        result = np.sum([temp_hist_2[item_x, item_y] * np.log(
            temp_hist_2[item_x, item_y] / temp_hist_0[item_x] / temp_hist_1[item_y])
                     for item_x in range(bins) for item_y in range(bins) if temp_hist_2[item_x, item_y] != 0])
        if normalization:
            entropy_0 = - np.sum(temp_hist_0 * np.log(temp_hist_0))
            entropy_1 = - np.sum(temp_hist_1 * np.log(temp_hist_1))
            result /= (0.5 * (entropy_0 + entropy_1))
        return result

    @staticmethod
    def generate_alkane_residue_code_in_openmm_xml(num, name):
        print('''<Residue name="%s">
<Atom charge="0.09" name="H11" type="HGA3"/>
<Atom charge="0.09" name="H12" type="HGA3"/>
<Atom charge="0.09" name="H13" type="HGA3"/>
<Atom charge="-0.27" name="C1" type="CG331"/>''' % name)
        for item in range(num - 2):
            print('''<Atom charge="0.09" name="H%d1" type="HGA2"/>
<Atom charge="0.09" name="H%d2" type="HGA2"/>
<Atom charge="-0.18" name="C%d" type="CG321"/>''' % (item + 2, item + 2, item + 2))
        print("""<Atom charge="0.09" name="H%d1" type="HGA3"/>
<Atom charge="0.09" name="H%d2" type="HGA3"/>
<Atom charge="0.09" name="H%d3" type="HGA3"/>
<Atom charge="-0.27" name="C%d" type="CG331"/>
<Bond atomName1="H11" atomName2="C1"/>
<Bond atomName1="H12" atomName2="C1"/>
<Bond atomName1="H13" atomName2="C1"/>""" % (num, num, num, num))
        for item in range(num - 1):
            print("""<Bond atomName1="C%d" atomName2="C%d"/>
<Bond atomName1="H%d1" atomName2="C%d"/>
<Bond atomName1="H%d2" atomName2="C%d"/>""" % (item + 1, item + 2, item + 2, item + 2, item + 2, item + 2))
        print("""<Bond atomName1="H%d3" atomName2="C%d"/>
<AllowPatch name="MET1"/>
<AllowPatch name="MET2"/>
</Residue>""" % (num, num))
        return

    @staticmethod
    def check_center_of_mass_is_at_origin(result):
        coords_of_center_of_mass_after = [[np.average(result[item, ::3]), np.average(result[item, 1::3]),
                                           np.average(result[item, 2::3])]
                                          for item in range(result.shape[0])]
        return np.all(np.abs(np.array(coords_of_center_of_mass_after).flatten()) < 1e-5)

    @staticmethod
    def remove_translation(coords):  # remove the translational degree of freedom
        if len(coords.shape) == 1:  # convert 1D array (when there is only one coord) to 2D array
            coords = coords.reshape((1, coords.shape[0]))
        number_of_atoms = coords.shape[1] // 3
        coords_of_center_of_mass = [[np.average(coords[item, ::3]), np.average(coords[item, 1::3]),
                                     np.average(coords[item, 2::3])] * number_of_atoms
                                    for item in range(coords.shape[0])]
        result = coords - np.array(coords_of_center_of_mass)
        assert Helper_func.check_center_of_mass_is_at_origin(result)
        return result

    @staticmethod
    def get_gyration_tensor_and_principal_moments(coords):
        coords = Helper_func.remove_translation(coords)
        temp_coords = coords.reshape(coords.shape[0], coords.shape[1] // 3, 3)
        gyration = np.zeros((coords.shape[0], 3, 3))
        for xx in range(3):
            for yy in range(3):
                gyration[:, xx, yy] = (temp_coords[:, :, xx] * temp_coords[:, :, yy]).mean(axis=-1)
        moments_gyration = np.linalg.eig(gyration)[0]
        moments_gyration.sort(axis=-1)
        return gyration, moments_gyration[:, ::-1]

    @staticmethod
    def get_norm_factor(rcut, sig):
        rcut2 = rcut*rcut
        sig2 = 2.0*sig*sig
        normconst = np.sqrt( np.pi * sig2 ) * erf( rcut / (sqrt(2.0)*sig) ) - 2*rcut* np.exp( - rcut2 / sig2 )
        preerf = np.sqrt( 0.5 * np.pi * sig * sig ) / normconst
        prelinear = np.exp( - rcut2 / sig2 ) / normconst
        return normconst, preerf, prelinear

    @staticmethod
    def get_cg_count_in_sphere(dis, r_hi, rcut, sig):  # get coarse grained counts
        # TODO: test if this function is correct
        normconst, preerf, prelinear = Helper_func.get_norm_factor(rcut, sig)
        hiMinus = r_hi - rcut
        hiPlus = r_hi + rcut
        count = np.float64((dis <= hiPlus).sum(axis=-1))
        temp_in_boundary_region = ((dis > hiMinus) & (dis <= hiPlus))
        temp_correction = ( 0.5 + preerf * erf( np.sqrt(0.5) * (dis - r_hi)/sig ) \
                                             - prelinear * (dis - r_hi))
        # print count.shape, temp_in_boundary_region.shape, temp_correction.shape
        count -= (temp_in_boundary_region * temp_correction).sum(axis=-1)
        actual_count = (dis < r_hi).sum(axis=-1)
        return count, actual_count

    @staticmethod
    def get_cg_count_in_shell(dis, r_low, r_hi, rcut, sig):
        cg_1, actual_1 = Helper_func.get_cg_count_in_sphere(dis, r_hi, rcut, sig)
        cg_2, actual_2 = Helper_func.get_cg_count_in_sphere(dis, r_low, rcut, sig)
        return cg_1 - cg_2, actual_1 - actual_2

    @staticmethod
    def get_cg_count_slice_representation(dis, r_shell_low, r_shell_high, num, rcut, sig):
        temp_r = np.linspace(r_shell_low, r_shell_high, num)
        r_low_list = temp_r[:-1]
        r_high_list = temp_r[1:]
        result = [Helper_func.get_cg_count_in_shell(dis, r_low, r_high, rcut, sig)[0]
                  for (r_low, r_high) in zip(r_low_list, r_high_list)]
        return np.concatenate(result, axis=1), temp_r

    @staticmethod
    def get_box_length_list_fom_reporter_file(reporter_file, unit):  # require unit explicitly
        reporter_file_content = np.loadtxt(reporter_file, delimiter=',', usecols=(6,))  # column 6 is volume of box
        if unit == 'nm': scaling_factor = 1
        elif unit == 'A': scaling_factor = 10
        return scaling_factor * np.cbrt(reporter_file_content)

    @staticmethod
    def compute_distances_min_image_convention(atoms_pos_1, atoms_pos_2, box_length_list):
        # note: box_length may be different for different frames when using NPT, typically is read from reporter file
        # shape of atoms_pos_{1,2}: (num of frames, num of atoms * 3)
        # output: distance matrix
        # why don't we use mdtraj?  Because it requires large memory for loading large pdb files
        # why don't we use MDAnalysis?  Because it is not fast enough (looping over trajectory would take long time)
        # this function is especially useful when both atoms_pos_1, atoms_pos_2 are not super long, while the number of frames is large, 
        # since it vectorizes computation over frames
        temp_dis_2 = np.zeros((atoms_pos_1.shape[0], atoms_pos_1.shape[1] // 3, atoms_pos_2.shape[1] // 3))
        for index_1 in range(atoms_pos_1.shape[1] // 3):
            # print index_1
            for index_2 in range(atoms_pos_2.shape[1] // 3):
                temp_diff = atoms_pos_1[:, 3 * index_1: 3 * index_1 + 3] - atoms_pos_2[:, 3 * index_2: 3 * index_2 + 3]
                temp_vec = np.array([(item + box_length_list / 2.0) % box_length_list - box_length_list / 2.0 for item in temp_diff.T])
                temp_dis_2[:, index_1, index_2] = np.linalg.norm(temp_vec, axis=0)
        return temp_dis_2

    @staticmethod
    def get_index_list_of_O_atom_in_water(pdb_file, ignore_TER_line):
        """this is used for solvent analysis, e.g. biased simulation with PLUMED"""
        temp_u = Universe(pdb_file)
        atom_sel = temp_u.select_atoms('resname HOH and name O')
        if ignore_TER_line: return atom_sel.indices + 1
        else: raise Exception('double check your pdb')

    @staticmethod
    def get_distances_with_water_for_atom_list(pdb_file, atom_selection, box_length_list):
        # box_length information is stored in reporter_file
        temp_u = Universe(pdb_file)
        water_pos, atoms_pos = [], []
        water_sel = temp_u.select_atoms('resname HOH and name O')
        atoms_sel = temp_u.select_atoms(atom_selection)
        for _ in temp_u.trajectory:
            water_pos.append(water_sel.positions.flatten())
            atoms_pos.append(atoms_sel.positions.flatten())
        atoms_pos = np.array(atoms_pos)
        water_pos = np.array(water_pos)
        distances = Helper_func.compute_distances_min_image_convention(atoms_pos_1=atoms_pos, atoms_pos_2=water_pos,
                                                                       box_length_list=box_length_list)
        return distances

    @staticmethod
    def get_list_of_cg_count_for_atom_list(pdb_file, atom_selection, box_length_list, r_low, r_hi, rcut, sig):
        """ cg = coarse grained, atom list is specified by atom_selection """
        distances = Helper_func.get_distances_with_water_for_atom_list(pdb_file, atom_selection, box_length_list)
        return Helper_func.get_cg_count_in_shell(distances, r_low, r_hi, rcut, sig)

    @staticmethod
    def get_radial_distribution(distances, num, nbins, dr, length):
        hist = np.zeros(nbins, )
        for item in distances:
            temp_target_index = int(item / dr)
            if temp_target_index < nbins:
                hist[temp_target_index] += 1.0 / (4 / 3.0 * np.pi) / (
                            ((temp_target_index + 1) * dr) ** 3 - ((temp_target_index + 0) * dr) ** 3)
        return hist / (num / length ** 3)

    @staticmethod
    def backup_rename_file_if_exists(filename):
        extension = '.' + filename.split('.')[-1]
        if os.path.isfile(filename):  # backup file if previous one exists
            new_filename = filename + ".bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + extension
            os.rename(filename, new_filename)
        else: new_filename = None
        return new_filename

    @staticmethod
    def attempt_to_save_npy(npy_file, npy_array):
        """when trying to save a npy array to a file, if it exists and contains a different value,
        then save to another file"""
        if npy_file.strip()[-4:] != '.npy': npy_file += '.npy'
        original_npy_file = npy_file
        index = 0
        while True:
            if os.path.isfile(npy_file):
                content = np.load(npy_file)
                if np.all(npy_array == content):
                    break
                else:
                    npy_file = original_npy_file.replace('.npy', '_%d.npy' % index)
                    index += 1
            else:
                np.save(npy_file, npy_array)
                break
        return npy_file

    @staticmethod
    def run_multiple_jobs_on_local_machine(commands, num_of_jobs_in_parallel=CONFIG_56):
        total_num_failed_jobs = 0
        for item in range(int(len(commands) / num_of_jobs_in_parallel) + 1):
            temp_commands_parallel = commands[item * num_of_jobs_in_parallel: (item + 1) * num_of_jobs_in_parallel]
            print("running: \t" + '\n'.join(temp_commands_parallel))
            procs_to_run_commands = [subprocess.Popen(_1.strip(), shell=True) for _1 in temp_commands_parallel]
            exit_codes = [p.wait() for p in procs_to_run_commands]
            total_num_failed_jobs += sum(exit_codes)
        return total_num_failed_jobs

    @staticmethod
    def shuffle_multiple_arrays(list_of_arrays):
        """can be used for shuffle training and validation set to improve sampling"""
        indices = np.arange(list_of_arrays[0].shape[0])
        np.random.shuffle(indices)
        return [item[indices] for item in list_of_arrays]

    @staticmethod
    def find_indices_of_points_in_array_near_each_point_in_ref_list(point_list, ref_list, threshold_r):
        """used to find points near a specific point (in the reference list), useful for sampling structures
        in a pdb file that are near a specific point in CV space  (result is the indices of pdb snapshots)
        """
        return [np.where(np.linalg.norm(point_list - item, axis=1) < threshold_r)[0]
                for item in ref_list]

    @staticmethod
    def tica_inverse_transform(tica, data_list):
        from msmbuilder.decomposition import tICA
        assert (isinstance(tica, tICA))
        result_list = []
        for data in data_list:
            result = np.dot(tica.covariance_.T, np.dot(tica.components_.T, data.T)).T + tica.means_
            assert_almost_equal(tica.transform([result])[0], data)
            result_list.append(result)
        return result_list

    @staticmethod
    def get_autocorr(x_list, lag_time):
        return np.corrcoef(np.array([x_list[0:len(x_list) - lag_time], x_list[lag_time:len(x_list)]]))[0, 1]

    @staticmethod
    def generate_sequence_with_constant_autocorrelation(constant_autocorrelation, length):
        traj_list = [np.random.normal()]
        for _ in range(length - 1):
            temp_value = np.random.normal(constant_autocorrelation * traj_list[-1], scale=1)
            traj_list.append(temp_value)
        return traj_list

    @staticmethod
    def load_object_from_pkl_file(file_path):
        return pickle.load(open(file_path, 'rb'))
