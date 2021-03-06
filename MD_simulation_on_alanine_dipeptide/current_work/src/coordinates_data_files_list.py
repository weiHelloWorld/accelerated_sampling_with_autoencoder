from config import *
from helper_func import *

class coordinates_data_files_list(object):
    def __init__(self,
                list_of_dir_of_coor_data_files = CONFIG_1, # this is the directory that holds corrdinates data files
                ):
        assert (isinstance(list_of_dir_of_coor_data_files, list))    # to avoid passing the string in the constructor
        self._list_of_dir_of_coor_data_files = list_of_dir_of_coor_data_files
        self._list_of_coor_data_files = []

        for item in self._list_of_dir_of_coor_data_files:
            self._list_of_coor_data_files += subprocess.check_output('''find %s -name "*coordinates.npy"''' % item, shell=True).decode("utf-8").strip().split('\n')

        self._list_of_coor_data_files = list(set(self._list_of_coor_data_files))  # remove duplicates
        self._list_of_coor_data_files = [x for x in self._list_of_coor_data_files if os.stat(x).st_size > 0]   # remove empty files
        self._list_of_coor_data_files.sort()                # to be consistent
        self._list_num_frames = [np.load(_1).shape[0] for _1 in self._list_of_coor_data_files]

        return

    def create_sub_coor_data_files_list_using_filter_conditional(self, filter_conditional):
        """
        :param filter_conditional: a lambda conditional expression on file names
        :return: a coordinates_data_files_list object
        """
        temp_coor_files = list(filter(filter_conditional, self._list_of_coor_data_files))
        return coordinates_data_files_list(temp_coor_files)

    def get_list_of_coor_data_files(self):
        return self._list_of_coor_data_files

    def get_coor_data(self, scaling_factor, format='npy'):
        result = np.concatenate([
            Helper_func.load_npy(item, format=format) for item in self._list_of_coor_data_files], axis=0) / scaling_factor
        assert (sum(self._list_num_frames) == result.shape[0])
        return result

    def get_list_of_corresponding_pdb_dcd(self):
        list_of_corresponding_pdb_files = [x.strip().replace('_coordinates.npy', '.pdb') for x in self.get_list_of_coor_data_files()]
        for item in range(len(list_of_corresponding_pdb_files)):
            if not os.path.exists(list_of_corresponding_pdb_files[item]):
                list_of_corresponding_pdb_files[item] = list_of_corresponding_pdb_files[item].replace('.pdb', '.dcd')
                try:
                    assert os.path.exists(list_of_corresponding_pdb_files[item])
                except:
                    raise Exception('%s does not exist!' % list_of_corresponding_pdb_files[item])

        return list_of_corresponding_pdb_files

    def write_pdb_frames_into_file_with_list_of_coor_index(self, list_of_coor_index, out_file_name, verbose=True):
        """
        This function picks several frames from pdb files, and write a new pdb file as output,
        we could use this together with the mouse-clicking callback implemented in the scatter plot:
        first we select a few points interactively in the scatter plot, and get corresponding index in the data point
        list, the we find the corresponding pdb frames with the index
        """
        Helper_func.backup_rename_file_if_exists(out_file_name)
        list_of_coor_index.sort()
        pdb_files = self.get_list_of_corresponding_pdb_dcd()
        accum_sum = np.cumsum(np.array(self._list_num_frames))  # use accumulative sum to find corresponding pdb files
        for item in range(len(accum_sum)):
            if item == 0:
                temp_index_related_to_this_pdb_file = [x for x in list_of_coor_index if x < accum_sum[item]]
            else:
                temp_index_related_to_this_pdb_file = [x for x in list_of_coor_index if accum_sum[item - 1] <= x < accum_sum[item]]
                temp_index_related_to_this_pdb_file = [x - accum_sum[item - 1] for x in temp_index_related_to_this_pdb_file]
            temp_index_related_to_this_pdb_file.sort()

            if len(temp_index_related_to_this_pdb_file) != 0:
                if verbose: print(pdb_files[item])
                with open(pdb_files[item], 'r') as in_file:
                    content = in_file.read().split('MODEL')[1:]  # remove header
                    frames_to_use = [content[ii] for ii in temp_index_related_to_this_pdb_file]
                    with open(out_file_name, 'a') as out_file:
                        for frame in frames_to_use:
                            out_file.write("MODEL" + frame)

        return

    def get_pdb_name_and_corresponding_frame_index_with_global_coor_index(self, coor_index):
        for item, temp_pdb in zip(self._list_num_frames, self.get_list_of_corresponding_pdb_dcd()):
            if coor_index < item: break
            else: coor_index -= item
        return temp_pdb, coor_index

    def concat_all_pdb_files(self, out_pdb_file):
        """
        Why don't I use 'cat' in terminal? since I want to make order consistent with Python sort() function 
        """
        with open(out_pdb_file, 'w') as outfile:
            for fname in self.get_list_of_corresponding_pdb_dcd():
                with open(fname) as infile:
                    outfile.write(infile.read())
        return

