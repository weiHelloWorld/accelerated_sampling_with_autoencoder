from ANN_simulation import *

class classification_sampler(object):
    def __init__(self, index, all_states, end_state_index):    # end_states defined by pdb files
        self._index = index
        self._end_state_index = end_state_index
        self._all_states = all_states
        self._classifier = None
        self._scaling_factor = 5.0
        self._atom_selection = 'not name H*'
        return

    def get_input_from_pdbs(self, pdb_list):
        """can be modified to other input features later"""
        return Sutils.get_non_repeated_pairwise_distance(
            pdb_list, atom_selection=self._atom_selection) / self._scaling_factor

    def get_training_data(self):
        # TODO: need to make sure all classes have roughly equal number of data
        train_in = []
        class_labels = []
        for _1, item in enumerate(self._all_states):
            temp_train_in = self.get_input_from_pdbs([item])
            temp_class_labels = np.zeros((temp_train_in.shape[0], len(self._all_states)))
            temp_class_labels[:, _1] = 1
            train_in.append(temp_train_in)
            class_labels.append(temp_class_labels)

        assert (len(train_in) == len(self._all_states))
        train_in = np.concatenate(train_in, axis=0)
        class_labels = np.concatenate(class_labels, axis=0)
        return train_in, class_labels

    def train_classifier(self, lr=.01, momentum=0.9, total_num_training = 5):
        train_in, train_out = self.get_training_data()
        node_num = [train_in.shape[1], 100, len(self._all_states)]
        best_val_loss = None
        for _ in range(total_num_training):   # train multiple models and pick the best one
            [train_in, train_out] = Helper_func.shuffle_multiple_arrays([train_in, train_out])
            inputs_net = Input(shape=(node_num[0],))
            x = Dense(node_num[1], activation='tanh')(inputs_net)
            x = Dense(node_num[2], activation='softmax')(x)
            molecule_net = Model(inputs=inputs_net, outputs=x)
            molecule_net.compile(loss='categorical_crossentropy', metrics=['categorical_crossentropy'],
                                 optimizer=SGD(lr=lr, momentum=momentum, nesterov=True))
            call_back_list = []
            earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
            call_back_list += [earlyStopping]

            train_history = molecule_net.fit(train_in, train_out, epochs=200, batch_size=50,
                                             verbose=0, validation_split=0.2, callbacks=call_back_list)
            if best_val_loss is None or train_history.history['val_loss'][-1] < best_val_loss:
                best_val_loss = train_history.history['val_loss'][-1]
                print "best_val_loss at iter %02d = %f" % (self._index, best_val_loss)
                fig, axes = plt.subplots(1, 2)
                axes[0].plot(train_history.history['loss'])
                axes[1].plot(train_history.history['val_loss'])
                png_file = 'history_%02d.png' % self._index
                Helper_func.backup_rename_file_if_exists(png_file)
                fig.savefig(png_file)
                model_file = 'classifier_%02d.hdf5' % self._index
                Helper_func.backup_rename_file_if_exists(model_file)
                molecule_net.save(model_file)
                self._classifier = model_file
        return

    def write_classifier_coeff_info(self, out_file, mode="ANN_Force"):
        molecule_net = load_model(self._classifier)
        node_num = [molecule_net.layers[0].output_shape[1], molecule_net.layers[1].output_shape[1],
                    molecule_net.layers[2].output_shape[1]]
        script = Sutils._get_plumed_script_with_pairwise_dis_as_input(
            get_index_list_with_selection_statement('../resources/alanine_dipeptide.pdb', self._atom_selection),
            self._scaling_factor)
        connection_between_layers_coeffs = [item.get_weights()[0].T.flatten() for item in
                                            molecule_net.layers if isinstance(item, Dense)]
        connection_with_bias_layers_coeffs = [item.get_weights()[1] for item in molecule_net.layers if
                                              isinstance(item, Dense)]
        index_CV_layer = 2
        if mode == 'plumed':
            temp_AE = autoencoder_Keras(1, None, None)
            script += temp_AE.get_expression_script_for_plumed(
                node_num=node_num,
                connection_between_layers_coeffs=connection_between_layers_coeffs,
                connection_with_bias_layers_coeffs=connection_with_bias_layers_coeffs,
                index_CV_layer=index_CV_layer, activation_function_list=['tanh', 'softmax'])
            script += 'mypotential: RESTRAINT ARG='
            for item in range(node_num[2]): script += 'l_2_out_%d,' % item
            result = script[:-1] + ' '
            with open(out_file, 'w') as f_out:
                f_out.write(result)
        elif mode == 'ANN_Force':
            with open(out_file, 'w') as f_out:
                for item in range(index_CV_layer):
                    f_out.write(str(list(connection_between_layers_coeffs[item])))
                    f_out.write(',\n')
                for item in range(index_CV_layer):
                    f_out.write(str(list(connection_with_bias_layers_coeffs[item])))
                    f_out.write(',\n')
        return

    def choose_two_states_list_between_which_we_sample_intermediates(self, metric="RMSD", option=0):
        dis_to_end_states = self.get_dis_to_end_states(metric)
        print dis_to_end_states
        if option == 0:
            sorted_index = [np.argsort(item) for item in dis_to_end_states]
            for temp_index in [0, 1]:
                assert (dis_to_end_states[temp_index][sorted_index[temp_index][0]] < 1e-5)       # because distance to itself should be 0
            two_states_closest_to_two_ends = [sorted_index[0][1], sorted_index[1][1]]          # why choose state with sorted_index = 1? since the distance to itself = 0
            if dis_to_end_states[0][two_states_closest_to_two_ends[0]] < dis_to_end_states[1][two_states_closest_to_two_ends[1]]:
                chosen_end_state_index = 1
            else:
                chosen_end_state_index = 0
            result_list = [two_states_closest_to_two_ends[chosen_end_state_index], self._end_state_index[chosen_end_state_index]]
        return result_list

    def get_dis_to_end_states(self, metric):
        if metric == 'input':  # what would be a good distance metric?
            ave_input_list = [np.average(self.get_input_from_pdbs([item]), axis=0)
                              for item in self._all_states]
            dis_to_end_states = [
                [np.linalg.norm(ave_input_list[self._end_state_index[temp_index]] - item)
                 for item in ave_input_list] for temp_index in [0, 1]]
        elif metric == 'RMSD':
            atom_pos = [Sutils.get_positions_from_list_of_pdb([item], atom_selection_statement=self._atom_selection)
                        for item in self._all_states]
            dis_to_end_states = []
            for temp_end_index in [self._end_state_index[0], self._end_state_index[1]]:
                temp_dis_to_end_states = []
                for temp_state_index, _ in enumerate(self._all_states):
                    if temp_state_index == temp_end_index:
                        temp_average_RMSD = 0
                    else:
                        temp_average_RMSD = np.average([Sutils.get_RMSD_after_alignment(item_1, item_2)
                                                        for item_1 in atom_pos[temp_end_index] for item_2 in
                                                        atom_pos[temp_state_index]])
                    temp_dis_to_end_states.append(temp_average_RMSD)
                dis_to_end_states.append(temp_dis_to_end_states)
        else:
            raise Exception('metric error')
        return dis_to_end_states

    def sample_intermediate_between_two_states(self, state_index_1, state_index_2, folder, coeff_info_file,
                                               force_constant=500, mode='ANN_Force'):
        pc_string = ['0'] * len(self._all_states)
        pc_string[state_index_1] = pc_string[state_index_2] = '0.5'
        pc_string = ','.join(pc_string)
        out_pdb = folder + '/out_%02d_between_%02d_%02d.pdb' % (len(self._all_states), state_index_1, state_index_2)
        if mode == 'plumed':
            kappa_string = ','.join([str(force_constant)] * len(self._all_states))
            command = 'python ../src/biased_simulation.py 50 50000 0 %s none pc_%s --platform CPU ' % (folder, pc_string)
            command += '--output_pdb  %s ' % out_pdb
            command += '--bias_method plumed_other --plumed_file %s ' % coeff_info_file
            command += ' --plumed_add_string " AT=%s KAPPA=%s"' % (pc_string, kappa_string)
        elif mode == 'ANN_Force':
            command = 'python ../src/biased_simulation.py 50 50000 %s %s %s pc_%s --platform CPU ' % (
                str(force_constant), folder, coeff_info_file, pc_string)
            command += '--output_pdb  %s --layer_types "Tanh,Softmax" --num_of_nodes 45,100,%d --data_type_in_input_layer 2' % (
                out_pdb, len(self._all_states))
            command += ' --scaling_factor 5.0'
        else: raise Exception('mode error')
        print command
        subprocess.check_output(command, shell=True)
        self._all_states.append(out_pdb)
        return
