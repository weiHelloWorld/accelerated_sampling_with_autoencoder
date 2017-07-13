from ANN_simulation import *

class String_method(object):
    def __init__(self, num_iterations, num_images=None):
        self._num_iterations = num_iterations
        self._num_images = num_images
        return

    def get_images_and_write_into_a_pdb_file(self, folder_containing_data_in_previous_iteration,
                                             autoencoder_filename, num_images, new_pdb_file_name='temp_new.pdb',
                                             scaling_factor=CONFIG_49):
        _1 = coordinates_data_files_list([folder_containing_data_in_previous_iteration])
        _1 = _1.create_sub_coor_data_files_list_using_filter_conditional(lambda x: not 'aligned' in x)
        coor_data = Sutils.remove_translation(_1.get_coor_data(scaling_factor=scaling_factor))
        temp_autoencoder = Sutils.load_object_from_pkl_file(autoencoder_filename)
        assert (isinstance(temp_autoencoder, autoencoder))
        PCs = temp_autoencoder.get_PCs(coor_data)
        image_params = np.linspace(PCs.min(), PCs.max(), num_images + 2)
        index_list_of_images = [(np.abs(PCs - temp_image_param)).argmin() for temp_image_param in image_params]
        print index_list_of_images, PCs[index_list_of_images], image_params
        if os.path.exists(new_pdb_file_name):
            subprocess.check_output(['rm', new_pdb_file_name])

        Sutils.write_some_frames_into_a_new_file_based_on_index_list_for_pdb_file_list(
            list_of_files=_1.get_list_of_corresponding_pdb_files(),index_list=index_list_of_images,
            new_pdb_file_name=new_pdb_file_name
        )
        # following is assertion part
        temp_coor_file = molecule_type.generate_coordinates_from_pdb_files(new_pdb_file_name)[0]
        _2 = coordinates_data_files_list([temp_coor_file])
        coor_data = Sutils.remove_translation(_2.get_coor_data(scaling_factor=scaling_factor))
        expected = PCs[index_list_of_images]; actual =temp_autoencoder.get_PCs(coor_data)
        print actual
        assert_almost_equal(expected, actual, decimal=4)
        return new_pdb_file_name

    def reparametrize(self, folder_containing_data_in_previous_iteration, index, num_images):
        iteration.preprocessing(target_folder=folder_containing_data_in_previous_iteration)  # structural alignment and generate coordinate files
        temp_output = subprocess.check_output(['python', '../src/train_network_and_save_for_iter.py', str(index),
                                 '--data_folder', folder_containing_data_in_previous_iteration,
                                 '--num_PCs', '1'
                                 ])
        print temp_output
        autoencoder_filename = temp_output.strip().split('\n')[-1]
        new_pdb_file_name = self.get_images_and_write_into_a_pdb_file(folder_containing_data_in_previous_iteration,
                                             autoencoder_filename, num_images)
        return new_pdb_file_name

    def relax_using_multiple_images_contained_in_a_pdb(self, output_folder, pdb_file):
        command_list = []
        for item in range(Universe(pdb_file).trajectory.n_frames):
            command = ['python', '../src/biased_simulation_general.py', '2src',
                         '50', '500', '0', output_folder, 'none', 'pc_0,%d' % item,
                         'explicit', 'NPT', '--platform', 'CUDA',
                         '--starting_pdb_file', pdb_file, '--starting_frame', str(item),
                         '--temperature', '0', '--equilibration_steps', '0']
            command = ['python', '../src/biased_simulation.py',
                       '50', '500', '0', output_folder, 'none', 'pc_0,%d' % item,
                       '--platform', 'CPU',
                       '--starting_pdb_file', pdb_file, '--starting_frame', str(item),
                       '--temperature', '0', '--equilibration_steps', '0']
            print ' '.join(command)
            command_list.append(command)
            subprocess.check_output(command)
        return

    def run_iteration(self, data_folder, index, num_images):
        new_pdb_file_name = self.reparametrize(data_folder, index=index, num_images=num_images)
        self.relax_using_multiple_images_contained_in_a_pdb('temp_output', new_pdb_file_name)
        return

if __name__ == '__main__':
    a = String_method(10)
    a.run_iteration('../target/Alanine_dipeptide', 1, 10)
