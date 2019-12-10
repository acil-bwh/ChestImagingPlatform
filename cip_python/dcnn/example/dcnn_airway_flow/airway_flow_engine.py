import os
import datetime
import numpy as np
import pandas as pd
import json
import pickle
import time
from sklearn.model_selection import train_test_split

import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
import tensorflow.keras.optimizers as optimizers

if K.backend() == "tensorflow":
    import tensorflow as tf

from cip_python.dcnn.example.airway_flow_neural_network import AirwayFlowNeuralNetwork, NetworkParameters
from cip_python.dcnn.example.airway_flow_generator import AirwayFlowDataGenerator
import cip_python.dcnn.example.utils.custom_metrics as cm

from sklearn.model_selection import GridSearchCV


class Engine(object):
    """
    Class that controls the global processes of the CNN (training, validating, testing)
    """
    def __init__(self, output_folder=None, train_method='synth', network_desc=None):
        self.network = None
        self.output_folder = output_folder
        self.train_method = train_method
        self.network_desc = network_desc

    @property
    def folder_id(self):
        execution_date = datetime.datetime.now().strftime('%Y-%m-%d')
        return execution_date if self.network_desc is None else self.network_desc

    @property
    def model_folder(self):
        """
        Folder where the outputs of the algorithm will be stored.
        Create the folder if it does not exist.
        """
        if self.output_folder:
            model_folder = os.path.join(self.output_folder, 'AirwayFlowModels')
            if not os.path.isdir(model_folder):
                os.mkdir(model_folder)

            model_folder = os.path.join(model_folder, self.folder_id)
            if not os.path.isdir(model_folder):
                os.mkdir(model_folder)
                print ('    {} model folder created: '.format(model_folder))
        else:
            model_folder = None
        return model_folder

    @property
    def model_name(self):
        model_name = 'Model_' + self.network_desc + '.hdf5'
        return model_name

    @property
    def best_model_name(self):
        best_model_name = 'Model_' + self.network_desc + '_best.hdf5'
        return best_model_name

    @property
    def results_folder(self):
        """
        Folder where the results of the algorithm will be stored.
        Create the folder if it does not exist.
        """
        if self.output_folder:
            results_folder = os.path.join(self.output_folder, 'AirwayFlowResults')
            if not os.path.isdir(results_folder):
                os.mkdir(results_folder)

            results_folder = os.path.join(results_folder, self.folder_id)
            if not os.path.isdir(results_folder):
                os.mkdir(results_folder)
                print ('    {} folder created'.format(results_folder))
        else:
            results_folder = None
        return results_folder

    @property
    def results_folder_two_branches(self):
        """
        Folder where the results of the algorithm will be stored.
        Create the folder if it does not exist.
        """
        if self.output_folder:
            results_folder_two_branches = os.path.join(self.results_folder, 'TwoBranches')
            if not os.path.isdir(results_folder_two_branches):
                os.mkdir(results_folder_two_branches)
                print ('    {} folder created'.format(results_folder_two_branches))
        else:
            results_folder_two_branches = None
        return results_folder_two_branches

    @property
    def out_plot_file(self):
        return self.network_desc + '_{}'

    @property
    def network_parameters_file_path(self):
        """
        Json file that represents the current network parameters
        """
        file_name = 'Parameters_' + self.network_desc + '.json'
        return os.path.join(self.model_folder, file_name)

    @property
    def mean_std_file_path(self):
        """
        pkl file that represents the current network parameters
        """
        file_name = 'MeanStd' + self.network_desc + '.pkl'
        return os.path.join(self.model_folder, file_name)

    @staticmethod
    def is_tensorflow():
        return K.backend() == 'tensorflow'

    @staticmethod
    def load_model_from_keras(model_file):
        """
        Load a pretrained Keras model from a file that will contain both configuration and weights
        :param model_file:
        :return:
        """
        if not os.path.exists(model_file):
            raise Exception("{} not found".format(model_file))

        network_model = load_model(model_file, custom_objects={'customized_loss': cm.customized_loss,
                                                               'loss_function': cm.precision_accuracy_loss(d=0.0025, c_prec=2.5)})

        return network_model

    def build_and_compile_network_model(self, network_params, save_arch=False):
        self.network = AirwayFlowNeuralNetwork()
        self.network.build_and_compile_network(network_params)

        if save_arch:
            self.plot_network()

    def build_and_compile_network_model_CV(self, d=0.0025, c_prec=2.5):
        model = Sequential()
        model.add(Dense(12, input_shape=(12,), kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(7, kernel_initializer='normal'))
        optimizer_ = optimizers.Adam(lr=5e-4)
        model.compile(loss=cm.precision_accuracy_loss(d=d, c_prec=c_prec), optimizer=optimizer_)

        return model

    def plot_network(self):
        """
        Save a png with the current Keras model.
        The file will be saved in the model folder as 'architecture.png')
        """
        from tensorflow.keras.utils import plot_model

        arch_name = self.network_desc + '_architecture.png'
        if self.model_folder:
            plot_model(self.network.model, to_file=os.path.join(self.model_folder, arch_name),
                       show_shapes=True)
            print ('    Model architecture saved!')
        else:
            raise Exception('No output folder specified to plot network')

    def save_network_model(self):
        model_path = os.path.join(self.model_folder, self.model_name)
        self.network.model.save(model_path)

    def save_network_parameters_and_configuration(self, network_parameters):
        """
        Save network parameters and model configuration
        :param network_parameters:
        :return:
        """
        parameters_to_save = dict()
        with open(self.network_parameters_file_path, 'wb') as f:
            parameters_to_save['Parameters'] = network_parameters.__dict__

            model_json = self.network.model.to_json()
            # Reformat for a more friendly visualization
            parsed = json.loads(model_json)
            parameters_to_save['Configuration'] = parsed

            f.write(json.dumps(parameters_to_save, indent=4, sort_keys=True))

    def save_mean_std_values(self, in_mean, in_std, out_mean, out_std):
        with open(self.mean_std_file_path, 'wb') as f:
            pickle.dump([in_mean, in_std, out_mean, out_std], f)

    @staticmethod
    def normalize_data(array, mean, std):
        return (array - mean) / std

    @staticmethod
    def load_dataset(dataset):
        dataframe = pd.read_csv(dataset, delim_whitespace=False, header=0)
        dataset = dataframe.values

        # split into input (X) and output (Y) variables
        input_array = dataset[:, 0:12]
        output_array = dataset[:, 12:19]

        return input_array, output_array

    @staticmethod
    def split_train_test(in_dataset, out_dataset, percentage=0.1):
        return train_test_split(in_dataset, out_dataset, test_size=percentage)

    def create_data_for_CV(self, x_dataset, y_dataset, x_mean, x_std, y_mean, y_std):
        x_array = np.zeros((x_dataset.shape[0] * 50, x_dataset.shape[1]), np.float32)
        y_array = np.zeros((y_dataset.shape[0] * 50, y_dataset.shape[1]), np.float32)

        for idx in range(x_dataset.shape[0]):
            pos = 0
            x_array[pos:pos+1] = x_dataset[idx].astype(np.float32)
            y_array[pos:pos+1] = y_dataset[idx].astype(np.float32)
            pos += 1
            for ii in range(1, 50):
                mass_flow_noise = np.random.uniform(-0.000001, 0.000001)
                augmented_data = x_dataset[idx].astype(np.float32)
                augmented_data[-1] += mass_flow_noise

                x_array[pos:pos + ii] = augmented_data
                y_array[pos:pos + ii] = y_dataset[idx].astype(np.float32)
                pos += ii
        x_data_norm = self.normalize_data(x_array, x_mean, x_std)
        y_data_norm = self.normalize_data(y_array, y_mean, y_std)

        return x_data_norm, y_data_norm

    def train_CV_network(self, train_dataset):
        """
        :param train_dataset:
        :param network_params:
        :param save_arch:
        :return:
        """
        print ('Cross validation...')

        all_input, all_outputs = self.load_dataset(train_dataset)
        x_train, x_test, y_train, y_test = self.split_train_test(all_input, all_outputs, percentage=0.25)
        # x_train, _, y_train, _ = self.split_train_test(x_tr_val, y_tr_val, percentage=0.1)

        input_mean = np.mean(x_train, axis=0)
        input_std = np.std(x_train, axis=0)
        output_mean = np.mean(y_train, axis=0)
        output_std = np.std(y_train, axis=0)

        X, Y = self.create_data_for_CV(x_train, y_train, input_mean, input_std, output_mean, output_std)
        model = KerasRegressor(build_fn=self.build_and_compile_network_model_CV,
                               batch_size=350,
                               verbose=1)

        d = np.linspace(0.001, 0.01, 30)
        c_prec = np.linspace(1.5, 3.5, 15)

        param_grid = dict(d=d, c_prec=c_prec)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_result = grid.fit(X, Y)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    def train_network(self, train_dataset, network_params=None, save_arch=False):
        """
        :param train_dataset:
        :param network_params:
        :param save_arch:
        :return:
        """
        start = time.time()

        # If no NetworkParameters was defined, default values are used!
        if network_params is None:
            network_params = NetworkParameters()

        self.build_and_compile_network_model(network_params, save_arch=save_arch)

        callbacks_ = list()
        if network_params.use_early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=2, mode='min')
            callbacks_.append(early_stopping)

        if Engine.is_tensorflow():
            tf.summary.scalar('learning_rate', self.network.model.optimizer.lr)
            tensorBoard_callback = TensorBoard(log_dir=self.model_folder, histogram_freq=1, write_graph=True,
                                               write_images=False)
            callbacks_.append(tensorBoard_callback)

        best_model_path = os.path.join(self.model_folder, self.best_model_name)
        model_chek_point = ModelCheckpoint(best_model_path, monitor='val_loss',
                                           verbose=0, save_best_only=True,
                                           mode='min')
        callbacks_.append(model_chek_point)

        all_input, all_outputs = self.load_dataset(train_dataset)
        x_tr_val, x_test, y_tr_val, y_test = self.split_train_test(all_input, all_outputs, percentage=0.25)
        x_train, x_val, y_train, y_val = self.split_train_test(x_tr_val, y_tr_val, percentage=0.1)

        input_mean = np.mean(x_train, axis=0)
        input_std = np.std(x_train, axis=0)
        output_mean = np.mean(y_train, axis=0)
        output_std = np.std(y_train, axis=0)

        self.save_mean_std_values(input_mean, input_std, output_mean, output_std)

        tr_data_generator = AirwayFlowDataGenerator(x_train, y_train, input_mean, input_std, output_mean, output_std,
                                                    batch_size=network_params.batch_size,
                                                    num_augmented_data=network_params.nb_augmented_data,
                                                    shuffle=True)
        train_steps = int(np.ceil(float(tr_data_generator.data_after_augmentation) / network_params.batch_size))

        val_data_generator = AirwayFlowDataGenerator(x_val, y_val, input_mean, input_std, output_mean, output_std,
                                                     batch_size=network_params.batch_size,
                                                     num_augmented_data=network_params.nb_augmented_data,
                                                     shuffle=True)
        val_steps = int(np.ceil(float(val_data_generator.data_after_augmentation) / network_params.batch_size))

        training_generator = tr_data_generator.data_generator()
        validation_generator = val_data_generator.data_generator()

        _ = self.network.model.fit_generator(training_generator, train_steps,
                                             epochs=network_params.nb_epochs,
                                             verbose=network_params.verbosity,
                                             validation_data=validation_generator,
                                             validation_steps=val_steps,
                                             callbacks=callbacks_,
                                             max_queue_size=100)

        total_time = time.time() - start
        print ('    Total training time: {:0>8}. ({} s/epoch)'.format(datetime.timedelta(seconds=total_time),
                                                                     float(total_time) / network_params.nb_epochs))

    def test_single_bifurcation(self, test_dataset):
        model_path = os.path.join(self.model_folder, self.best_model_name)
        network_model = self.load_model_from_keras(model_path)

        with open(self.mean_std_file_path) as f:
            in_mean, in_std, out_mean, out_std = pickle.load(f)

        all_input, all_outputs = self.load_dataset(test_dataset)
        _, x_test, _, y_test = self.split_train_test(all_input, all_outputs, percentage=0.25)

        x_test_norm = self.normalize_data(x_test, in_mean, in_std)
        y_test_norm = self.normalize_data(y_test, out_mean, out_std)

        predictions = network_model.predict(x_test_norm)

        self.generate_results_plots_one_branch(y_test_norm, predictions)

    def generate_results_plots_one_branch(self, y_true, y_pred):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        outputs = ['Mean Velocity 1', 'Mean Velocity 2', 'Mass Flow 1', 'Mass Flow 2', 'Dynamic Pressure 1',
                   'Dynamic Pressure 2', 'Wall Shear Stress']

        for ii, nn in enumerate(outputs):
            plt.plot(y_true[:, ii], y_pred[:, ii], 'rx')
            plt.xlabel('Simulation')
            plt.ylabel('Regression')
            plt.title(nn)
            axes = plt.gca()
            axes.set_xlim([np.min(y_true[:, ii]), np.max(y_true[:, ii])])
            axes.set_ylim([np.min(y_pred[:, ii]), np.max(y_pred[:, ii])])
            lims = [
                np.min([axes.get_xlim(), axes.get_ylim()]),  # min of both axes
                np.max([axes.get_xlim(), axes.get_ylim()]),  # max of both axes
            ]

            # now plot both limits against each other
            axes.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            axes.set_aspect('equal')
            axes.set_xlim(lims)
            axes.set_ylim(lims)

            output_plot = os.path.join(self.results_folder, self.out_plot_file.format(str(ii)))
            plt.savefig(output_plot)

            print ('Pearson correlation coefficient, 2-tailed p-value, {}: '.format(nn),\
                self.corr_coefficient(y_pred[:, ii], y_true[:, ii]))


    def test_double_bifurcation(self, test_dataset, test_dataset_dauA, test_dataset_dauB):
        model_path = os.path.join(self.model_folder, self.best_model_name)
        network_model = self.load_model_from_keras(model_path)

        with open(self.mean_std_file_path) as f:
            in_mean, in_std, out_mean, out_std = pickle.load(f)

        x_test, _ = self.load_dataset(test_dataset)

        # DAUGHTER A
        x_test_dauA, y_test_dauA = self.load_dataset(test_dataset_dauA)

        # DAUGHTER B
        x_test_dauB, y_test_dauB = self.load_dataset(test_dataset_dauB)

        x_test_norm = self.normalize_data(x_test, in_mean, in_std)

        # Predict on mother branch
        predictions_mother_norm = network_model.predict(x_test_norm)
        predictions_mother = predictions_mother_norm * out_std + out_mean

        # The predicted mass flows is set to the daughter branches
        for ii in range(x_test_dauA.shape[0]):
            x_test_dauA[ii][-1] = -predictions_mother[ii][2]
            x_test_dauB[ii][-1] = -predictions_mother[ii][3]

        # Normalization
        x_test_dauA_norm = self.normalize_data(x_test_dauA, in_mean, in_std)
        y_test_dauA_norm = self.normalize_data(y_test_dauA, out_mean, out_std)

        x_test_dauB_norm = self.normalize_data(x_test_dauB, in_mean, in_std)
        y_test_dauB_norm = self.normalize_data(y_test_dauB, out_mean, out_std)

        # Second prediction
        predictions_dauA_norm = network_model.predict(x_test_dauA_norm)
        predictions_dauB_norm = network_model.predict(x_test_dauB_norm)

        predictions_dauA = predictions_dauA_norm * out_std + out_mean
        predictions_dauB = predictions_dauB_norm * out_std + out_mean

        self.generate_results_plots_two_branches(y_test_dauA, predictions_dauA, y_test_dauB, predictions_dauB)

        print ('CCC daughter A mass flow 1: ', self.CCC_score(predictions_dauA[:, 2], y_test_dauA[:, 2]))
        print ('CCC daughter A mass flow 2: ', self.CCC_score(predictions_dauA[:, 3], y_test_dauA[:, 3]))
        print ('CCC daughter B mass flow 1: ', self.CCC_score(predictions_dauB[:, 2], y_test_dauB[:, 2]))
        print ('CCC daughter B mass flow 2: ', self.CCC_score(predictions_dauB[:, 3], y_test_dauB[:, 3]))

        print ('Pearson correlation coefficient, 2-tailed p-value, daughter A mass flow 1: ', \
            self.corr_coefficient(predictions_dauA[:, 2], y_test_dauA[:, 2]))
        print ('Pearson correlation coefficient, 2-tailed p-value, daughter A mass flow 2: ', \
            self.corr_coefficient(predictions_dauA[:, 3], y_test_dauA[:, 3]))
        print ('Pearson correlation coefficient, 2-tailed p-value, daughter B mass flow 1: ', \
            self.corr_coefficient(predictions_dauB[:, 2], y_test_dauB[:, 2]))
        print ('Pearson correlation coefficient, 2-tailed p-value, daughter B mass flow 2: ', \
            self.corr_coefficient(predictions_dauB[:, 3], y_test_dauB[:, 3]))

    def generate_results_plots_two_branches(self, y_true_dauA, y_pred_dauA, y_true_dauB, y_pred_dauB):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        outputs = ['Mean Velocity 1', 'Mean Velocity 2', 'Mass Flow 1', 'Mass Flow 2', 'Dynamic Pressure 1',
                   'Dynamic Pressure 2', 'Wall Shear Stress']

        # RESULTS FOR DAUGHTER BRANCH A
        for ii, nn in enumerate(outputs):
            plt.plot(y_true_dauA[:, ii], y_pred_dauA[:, ii], 'rx')
            plt.xlabel('Simulation')
            plt.ylabel('Regression')
            plt.title(nn)
            axes = plt.gca()
            axes.set_xlim([np.min(y_pred_dauA[:, ii]), np.max(y_pred_dauA[:, ii])])
            axes.set_ylim([np.min(y_pred_dauA[:, ii]), np.max(y_pred_dauA[:, ii])])
            lims = [
                np.min([axes.get_xlim(), axes.get_ylim()]),  # min of both axes
                np.max([axes.get_xlim(), axes.get_ylim()]),  # max of both axes
            ]

            # now plot both limits against each other
            axes.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            axes.set_aspect('equal')
            axes.set_xlim(lims)
            axes.set_ylim(lims)

            output_plot = os.path.join(self.results_folder_two_branches, self.out_plot_file.format('dauA_' + str(ii)))
            plt.savefig(output_plot)

        # RESULTS FOR DAUGHTER BRANCH B
        for ii, nn in enumerate(outputs):
            plt.plot(y_true_dauB[:, ii], y_pred_dauB[:, ii], 'rx')
            plt.xlabel('Simulation')
            plt.ylabel('Regression')
            plt.title(nn)
            axes = plt.gca()
            axes.set_xlim([np.min(y_pred_dauB[:, ii]), np.max(y_pred_dauB[:, ii])])
            axes.set_ylim([np.min(y_pred_dauB[:, ii]), np.max(y_pred_dauB[:, ii])])
            lims = [
                np.min([axes.get_xlim(), axes.get_ylim()]),  # min of both axes
                np.max([axes.get_xlim(), axes.get_ylim()]),  # max of both axes
            ]

            # now plot both limits against each other
            axes.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            axes.set_aspect('equal')
            axes.set_xlim(lims)
            axes.set_ylim(lims)

            output_plot = os.path.join(self.results_folder_two_branches, self.out_plot_file.format('dauB_' + str(ii)))
            plt.savefig(output_plot)

    def CCC_score(self, data1, data2):
        nb = data1.shape[0]
        x_mean = np.mean(data1)
        y_mean = np.mean(data2)
        ss_x = np.sum((data1 - x_mean) ** 2) / float(nb)
        ss_y = np.sum((data2 - y_mean) ** 2) / float(nb)
        ss_xy = np.sum((data1 - x_mean) * (data2 - y_mean)) / float(nb)

        return (2.0 * ss_xy) / (ss_x + ss_y + (x_mean - y_mean) ** 2)

    def corr_coefficient(self, data1, data2):
        from scipy.stats.stats import pearsonr
        return pearsonr(data1, data2)
