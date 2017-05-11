from config import *

class coordinates_data_files_list(object):
    def __init__(self,
                list_of_dir_of_coor_data_files = CONFIG_1, # this is the directory that holds corrdinates data files
                ):
        assert (isinstance(list_of_dir_of_coor_data_files, list))    # to avoid passing the string in the constructor
        self._list_of_dir_of_coor_data_files = list_of_dir_of_coor_data_files
        self._list_of_coor_data_files = []

        for item in self._list_of_dir_of_coor_data_files:
            self._list_of_coor_data_files += subprocess.check_output('''find %s -name "*coordinates.txt"''' % item, shell=True).strip().split('\n')

        self._list_of_coor_data_files = list(set(self._list_of_coor_data_files))  # remove duplicates
        self._list_of_coor_data_files = filter(lambda x: os.stat(x).st_size > 0, self._list_of_coor_data_files)   # remove empty files
        self._list_of_coor_data_files.sort()                # to be consistent
        self._list_of_line_num_of_coor_data_file = map(lambda x: int(subprocess.check_output(['wc', '-l', x]).strip().split()[0]),
                                                       self._list_of_coor_data_files)

        return

    def create_sub_coor_data_files_list_using_filter_conditional(self, filter_conditional):
        """
        :param filter_conditional: a lambda conditional expression on file names
        :return: a coordinates_data_files_list object
        """
        temp_coor_files = filter(filter_conditional, self._list_of_coor_data_files)
        return coordinates_data_files_list(temp_coor_files)

    def get_list_of_coor_data_files(self):
        return self._list_of_coor_data_files

    def get_coor_data(self, scaling_factor):
        result = np.concatenate([np.loadtxt(item) for item in self._list_of_coor_data_files], axis=0) / scaling_factor
        assert (sum(self._list_of_line_num_of_coor_data_file) == result.shape[0])
        return result

    def get_list_of_corresponding_pdb_files(self):
        list_of_corresponding_pdb_files = map(lambda x: x.strip().split('_coordinates.txt')[0] + '.pdb',
                                              self.get_list_of_coor_data_files()
                                              )
        for item in list_of_corresponding_pdb_files:
            try:
                assert os.path.exists(item)
            except:
                raise Exception('%s does not exist!' % item)

        return list_of_corresponding_pdb_files

    def get_list_of_line_num_of_coor_data_file(self):
        return self._list_of_line_num_of_coor_data_file

    def write_pdb_frames_into_file_with_list_of_coor_index(self, list_of_coor_index, out_file_name):
        """
        This function picks several frames from pdb files, and write a new pdb file as output,
        we could use this together with the mouse-clicking callback implemented in the scatter plot:
        first we select a few points interactively in the scatter plot, and get corresponding index in the data point
        list, the we find the corresponding pdb frames with the index
        """
        if os.path.isfile(out_file_name):  # backup files
            os.rename(out_file_name,
                      out_file_name.split('.pdb')[0] + "_bak_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pdb")

        list_of_coor_index.sort()
        pdb_files = self.get_list_of_corresponding_pdb_files()
        accum_sum = np.cumsum(np.array(self._list_of_line_num_of_coor_data_file))  # use accumulative sum to find corresponding pdb files
        for item in range(len(accum_sum)):
            if item == 0:
                temp_index_related_to_this_pdb_file = filter(lambda x: x < accum_sum[item], list_of_coor_index)
            else:
                temp_index_related_to_this_pdb_file = filter(lambda x: accum_sum[item - 1] <= x < accum_sum[item], list_of_coor_index)
                temp_index_related_to_this_pdb_file = map(lambda x: x - accum_sum[item - 1], temp_index_related_to_this_pdb_file)
            temp_index_related_to_this_pdb_file.sort()

            if len(temp_index_related_to_this_pdb_file) != 0:
                print(pdb_files[item])
                with open(pdb_files[item], 'r') as in_file:
                    content = in_file.read().split('MODEL')[1:]  # remove header
                    frames_to_use = [content[ii] for ii in temp_index_related_to_this_pdb_file]
                    with open(out_file_name, 'a') as out_file:
                        for frame in frames_to_use:
                            out_file.write("MODEL" + frame)

        return

    def concat_all_pdb_files(self, out_pdb_file):
        """
        Why don't I use 'cat' in terminal? since I want to make order consistent with Python sort() function 
        """
        with open(out_pdb_file, 'w') as outfile:
            for fname in self.get_list_of_corresponding_pdb_files():
                with open(fname) as infile:
                    outfile.write(infile.read())
        return

