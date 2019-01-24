from __future__ import division
from future.utils import iteritems, listvalues
import os
import logging
import argparse
import datetime
import json
import pickle
import tempfile
import time
import warnings
import math
import numpy as np

import keras
import tensorflow as tf
import keras.callbacks as callbacks
import keras.backend as K
import keras.optimizers as optimizers

from cip_python.dcnn.logic.utils import Utils
from cip_python.dcnn.data import H5Manager

class Engine(object):
    '''
    Class that controls the global processes of the CNN (training, validating, testing).
    '''

    def __init__(self, output_folder=None):
        '''
        Constructor
        Args:
            output_folder: all the output folder will be created here (log, models, training files, etc.)
        '''
        if not output_folder:
            # Generate a temp folder
            self.output_folder = os.path.join(tempfile.gettempdir(), Utils.now())
        else:
            self.output_folder = output_folder
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        logging.info("Results will be stored in " + self.output_folder)

        self.model = None
        self.parameters_dict = None
        self.train_dataset_manager = None


    @property
    def parameters_file_path(self):
        '''
        Path to save the parameters json file
        Returns:
            Path to the file
        '''
        return os.path.join(self.output_folder, 'parameters_dict.json')

    @property
    def keras_model_file_path(self):
        '''
        Path to save the architecture of the network (Keras)
        Returns:
            Path to the file
        '''
        return os.path.join(self.output_folder, 'kerasModel.json')

    @property
    def model_summary_file_path(self):
        '''
        Path to save the Keras model summary
        Returns:
            Path to the file
        '''
        return os.path.join(self.output_folder, "model_summary.txt")

    def save_summary(self, model):
        """
        Save the model Keras summary in a txt file
        :param model:
        """
        f = open(self.model_summary_file_path, 'w')

        def print_fn(s):
            f.write(s + "\n")

        model.summary(print_fn=print_fn)
        f.close()

    def get_loss_and_additional_metrics(self, metrics_manager, parameters_dict):
        """
        Obtain the loss function and the additional metric functions that are being used.
        Each element can be either a string representing a regular Keras function or a "pointer" to
        a customized metric.

        When the metrics are obtained from a dictionary of parameters, the 'loss' and (optionally) 'metrics' have a value

        In the case of metrics that contain parameters, two different syntaxes are allowed:
            - metric;param1;param2
            - metric,param1=value1,param2=value2
        :return: Tuple with:
            - Loss function (string or pointer to a custom metric function)
            - List of strings (for regular Keras functions) or pointers to custom additional metrics
        """
        metric_names = []
        metric_functions = []
        loss_function = Utils.get_param('loss', parameters_dict)
        metric_names.append(loss_function)
        # Check for additional metrics
        if Utils.get_param('metrics', parameters_dict):
            metric_names.extend(Utils.get_param('metrics', parameters_dict))
        for metric_str in metric_names:
            components = metric_str.split(',')
            metric_name = components[0]
            if metrics_manager.contains_metric(metric_name):
                # Custom metric. Parse parameters
                args = []
                kwargs = {}
                for i in range(1, len(components)):
                    kv = tuple(map(str.strip, components[i].split('=')))
                    if len(kv) == 1:
                        # Positional arg
                        args.append(kv[0])
                    else:
                        # Named arg
                        kwargs[kv[0]] = kv[1]
                # Get the "pointer" to the corresponding function
                metric = metrics_manager.get_metric(metric_name, *args, **kwargs)
                metric_functions.append(metric)
            else:
                # Assume this is a regular Keras metric. Save the string
                metric_functions.append(metric_name)

        # The first element in the list is the loss function
        loss_function = metric_functions.pop(0)
        additional_metrics = metric_functions

        return loss_function, additional_metrics

    def get_optimizer(self, parameters_dict):
        """
        Build a Keras optimizer based on a dictionary of parameters
        :param parameters_dict: dictionary of string-string
        """
        optimizer = Utils.get_param('optimizer', parameters_dict)
        assert optimizer is not None, "'optimizer' param is needed"
        initial_learning_rate = Utils.get_param('initial_learning_rate', parameters_dict)
        assert initial_learning_rate is not None, "'initial_learning_rate' param is needed"

        with K.name_scope('optimization') as scope:
            p = {}
            p['lr'] = initial_learning_rate
            if optimizer == 'Adam':
                if Utils.get_param('optim_adam_beta_1', parameters_dict):
                    p['beta_1'] = Utils.get_param('optim_adam_beta_1', parameters_dict)
                if Utils.get_param('optim_adam_beta_2', parameters_dict):
                    p['beta_2'] = Utils.get_param('optim_adam_beta_2', parameters_dict)
                return optimizers.Adam(**p)
            elif optimizer == 'SGD':
                if Utils.get_param('optim_sgd_momentum', parameters_dict):
                    p['momentum'] = Utils.get_param('optim_sgd_momentum', parameters_dict)
                if Utils.get_param('optim_sgd_nesterov', parameters_dict):
                    p['nesterov'] = Utils.get_param('optim_sgd_nesterov', parameters_dict)
                return optimizers.SGD(**p)
            else:
                # Other Keras optimizer
                return eval("optimizers.{}(lr=initial_learning_rate)".format(optimizer))

    def keras_generator(self, ds_manager, network, batch_size, batch_type):
        """
        Iterator to be used in Keras training/testing when there is no data augmentation on the fly
        :param ds_manager: H5DatasetManager object (or any of its child classes)
        :param network: Network object (child class)
        :param batch_size: int. Batch size
        :param batch_type: int. One of the values defined in H5DatasetManager (TRAIN, VALIDATION, TEST)
        :return: data batch for each iteration ready to be used by Keras
        """
        while True:
            # Get the data in the original H5 Dataset format
            xs, ys = ds_manager.get_next_batch(batch_size, batch_type)
            # Adapt to the network structure
            self.format_data_to_network(xs, ys, network)
            yield xs, ys

    def keras_generator_dynamic_augmentation(self, ds_manager, network,
                                             data_augmentor, num_dynamic_augmented_data_per_data_point,
                                             batch_size, batch_type=H5Manager.TRAIN):
        """
        Iterator to be used in Keras training/testing when there is data augmentation on the fly
        :param ds_manager: H5DatasetManager object (or any of its child classes)
        :param network: Network object (child class)
        :param data_augmentor: cip_python.dcnn.data.DataAugmentor. Object that will be used to generate augmented data
        :param num_dynamic_augmented_data_per_data_point: int. Number of data points that will be generated on the fly
                                                         for each data point in the dataset
        :param batch_size: int. Total batch size
        :param batch_type: int. One of the values defined in H5DatasetManager (TRAIN, VALIDATION, TEST)
        :return: data batch for each iteration ready to be used by Keras
        """
        # Get the number of original data that will be used for the batch
        num_original_data = math.ceil(batch_size / (num_dynamic_augmented_data_per_data_point  + 1))
        while True:
            # Initialize batch memory
            xs_sizes, ys_sizes = network.get_xs_ys_size()
            num_inputs = len(xs_sizes)
            num_outputs = len(ys_sizes)
            batch_xs = [np.zeros(((batch_size,) + s), dtype=np.float32) for s in xs_sizes]
            batch_ys = [np.zeros(((batch_size,) + s), dtype=np.float32) for s in ys_sizes]
            n = 0
            while n < batch_size:
                # Read all the original data points
                original_xs, original_ys = ds_manager.get_next_batch(num_original_data, batch_type)

                for j in range(num_original_data):
                    xs = [a[j] for a in original_xs]
                    ys = [a[j] for a in original_ys]
                    augmented_xs, augmented_ys = data_augmentor.generate_augmented_data_points(xs, ys,
                                                                           num_dynamic_augmented_data_per_data_point)
                    self.format_data_to_network(xs, ys, network)

                    # Insert original data
                    for i in range(num_inputs):
                        batch_xs[i][n] = xs[i]
                    for i in range(num_outputs):
                        batch_ys[i][n] = ys[i]

                    n += 1
                    # Insert augmented data for the current original data point
                    k = 0
                    while k < num_dynamic_augmented_data_per_data_point and n < batch_size:
                        xs = [a[k] for a in augmented_xs]
                        ys = [a[k] for a in augmented_ys]
                        self.format_data_to_network(xs, ys, network)
                        for i in range(num_inputs):
                            batch_xs[i][n] = xs[i]
                        for i in range(num_outputs):
                            batch_ys[i][n] = ys[i]
                        n += 1
                        k += 1
            yield batch_xs, batch_ys

    def format_data_to_network(self, xs, ys, network, intensity_checking=False):
        """
        Get a list of inputs/outputs and scale the sizes (if needed) and intensities to the format
        expected by the network.
        This method overrides the original data to save memory!
        :param xs: list of numpy arrays. Network inputs
        :param ys: list of numpy arrays. Network outputs
        :param intensity_checking: bool. Perform a validation of the intensity levels received (it can be used on predictions)
        :return: None (the method updates the passed data)
        """
        raise NotImplementedError("This method should be implemented in a child class")

    def train(self, parameters_dict):
        """
        Train the network.
        This method has to be implemented by any child classes, but some source code is provided just as a template
        :param parameters_dict: dictionary of parameters that controls the whole process
        :param use_tensorboard: use Tensorboard to log training
        """
        raise NotImplementedError("This method should be implemented in a child class!")
        start = time.time()

        # MODEL
        self.network = NetworkFactory.build_network(parameters_dict, compile_model=True)
        self.model = self.network.model
        # Save model
        try:
            with open(self.keras_model_file_path, 'wb') as f:
                model_json = self.model.to_json()
                # Reformat for a more friendly visualization
                parsed = json.loads(model_json)
                f.write(json.dumps(parsed, indent=4).encode())
            logging.info('Keras model saved to {}'.format(self.keras_model_file_path))
        except Exception as ex:
            warnings.warn("The model could not be saved to json: {}".format(ex))

        # DATA
        self.train_dataset_manager = H5Dataset(parameters_dict['train_dataset_path'],
                                               network=self.network,
                                               batch_size=parameters_dict['batch_size'],
                                               # Rest of parameters here......
                                               )

        if load_all_data_in_memory:
            # Load all the data in memory!
            train_data = self.train_dataset_manager.get_all_train_data()
            validation_data = self.train_dataset_manager.get_all_validation_data()
        else:
            # Load with Keras generators
            # Generate a random partition
            first_test_index = int(parameters_dict['first_test_index'])
            # Partition using cases
            self.train_dataset_manager.generate_train_validation_ixs_random_using_key('cids',
                                                                                      num_data_points_used=first_test_index)
            # Partition using any data
            self.train_dataset_manager.generate_train_validation_ixs_random(num_data_points_used=first_test_index)

            if Utils.get_param("force_steps_per_epoch", parameters_dict):
                # Force the number of steps per epoch
                training_steps_per_epoch = parameters_dict['force_steps_per_epoch']
            else:
                training_steps_per_epoch = self.train_dataset_manager.get_steps_per_epoch_train()

            if self.train_dataset_manager.num_validation_points > 0:
                # The validation dataset is a subset of the train dataset
                validation_data_generator = self.train_dataset_manager.keras_generator_validation()
                m = Utils.get_param('max_validation_steps', parameters_dict)
                if m:
                    validation_steps = m  # In big validation datasets, maybe we want only a subset of the whole dataset
                else:
                    # Analyze the whole validation dataset
                    validation_steps = self.train_dataset_manager.get_steps_validation()
            else:
                # No validation
                validation_data_generator = None
                validation_steps = None

        # CALLBACKS
        my_callbacks = []
        # Add the callback for the rest of the parameters_dict
        if use_tensorboard:
            self.tensorBoard_callback = callbacks.TensorBoard(log_dir=self.output_folder, write_graph=True)
            my_callbacks.append(self.tensorBoard_callback)
        else:
            self.tensorBoard_callback = None

        lrmode = Utils.get_param('learning_rate_reduce_mode', parameters_dict)
        loss = "loss" if Utils.get_param("overfit", parameters_dict) else "val_loss"
        if lrmode == 'CONSTANT':
            lr = parameters_dict['initial_learning_rate']
            decay = parameters_dict['learning_rate_reduce_factor']
            warnings.warn("Learning rate scheduler may have changed in recent versions of Keras")
            # We use as epoch the real epoch in the dataset instead of the Keras "artificial" one
            reduce_lr_on_epoch_end_callback = callbacks.LearningRateScheduler(lambda epoch: lr * pow(decay, self.train_dataset_manager.current_epoch))
            my_callbacks.append(reduce_lr_on_epoch_end_callback)
        elif parameters_dict['learning_rate_reduce_mode'] != 'OFF':
            # Reduce learning rate when the validation loss is not reduced
            reduce_lr_on_plateau_callback = callbacks.ReduceLROnPlateau(
                                                monitor=loss,
                                                factor=parameters_dict['learning_rate_reduce_factor'],
                                                patience=parameters_dict['learning_rate_reduce_patience'],
                                                mode=parameters_dict['learning_rate_reduce_mode'],
                                                cooldown=parameters_dict['learning_rate_reduce_cooldown'],
                                                verbose=1)
            my_callbacks.append(reduce_lr_on_plateau_callback)

        if "save_single_model" in parameters_dict and parameters_dict['save_single_model'] == False:
            # Save models every epoch
            p = os.path.join(self.output_folder, '{epoch:02d}.hdf5')
            checkpointer_callback = callbacks.ModelCheckpoint(filepath=p, verbose=0, save_best_only=False,
                                                              save_weights_only=True)
        else:
            # Save only one model
            p = os.path.join(self.output_folder, 'model.hdf5')
            checkpointer_callback = callbacks.ModelCheckpoint(filepath=p, verbose=0, save_best_only=True,
                                                              monitor=loss, save_weights_only=True)

        my_callbacks.append(checkpointer_callback)

        early_stop_patience = Utils.get_param('early_stop_patience', parameters_dict)
        if early_stop_patience:
            early_stopper_callback = callbacks.EarlyStopping(patience=early_stop_patience, monitor=loss)
            my_callbacks.append(early_stopper_callback)

        # Personalized callback
        # callback = GenericCallback(self)
        # my_callbacks.append(callback)

        # Fill dynamic parameters_dict
        giturl = Utils.get_git_url()
        parameters_dict['git_tag'] = giturl[0]
        parameters_dict['git_tag_is_dirty'] = giturl[1]
        parameters_dict['tensorflow_version'] = tf.__version__
        parameters_dict['keras_version'] = keras.__version__

        # Save current parameters_dict to a file
        with open(self.parameters_file_path, 'wb') as f:
            f.write(json.dumps(parameters_dict, indent=4, sort_keys=False).encode())
            logging.info('Parameters saved to {}'.format(self.parameters_file_path))

        # Save the current model summary to a file
        Utils.save_keras_model_summary(self.network.model, self.model_summary_file_path)

        if load_all_data_in_memory:
            # Save the parameters_dict to Tensorboard
            summaries = []
            for k, v in iteritems(parameters_dict):
                summaries.append(tf.summary.text(k, tf.convert_to_tensor(str(v))))
            with tf.Session() as sess:
                summary_writer = tf.summary.FileWriter(self.output_folder)
                for summary_op in summaries:
                    text = sess.run(summary_op)
                    summary_writer.add_summary(text, 0)
            summary_writer.close()

        self.parameters_dict = parameters_dict

        # TRAIN
        logging.info("Total number of model parameters: {}".format(self.model.count_params()))

        train_data_generator = self.train_dataset_manager.keras_generator_train()

        if load_all_data_in_memory:
            # Train will all the data loaded in memory
            training_history = self.model.fit(x=train_data[0],
                                              y=train_data[1],
                                              batch_size=self.train_dataset_manager.batch_size,
                                              epochs=parameters_dict['num_epochs'],
                                              validation_data=validation_data,
                                              shuffle=True,
                                              callbacks=my_callbacks,
                                              )
        else:
            training_history = self.model.fit_generator(train_data_generator,
                                                        training_steps_per_epoch,
                                                        epochs=parameters_dict['num_epochs'],
                                                        validation_data=validation_data_generator,
                                                        validation_steps=validation_steps,
                                                        callbacks=my_callbacks,
                                                        use_multiprocessing=False,
                                                        max_queue_size=5,
                                                        # workers=20,
                                                        )

        total_time = time.time() - start
        # logging.info(training_history)
        logging.info('Total training time: {}. ({} s/epoch)'.format(datetime.timedelta(seconds=total_time),
                                                                    float(total_time) / parameters_dict['num_epochs']))

        # Write the content of the history object to a file
        logging.debug("Saving history")
        with open(self.output_folder + '/trainHistoryDict.txt', 'wb') as f:
            pickle.dump(training_history.history, f)


    def predict(self, input_data, model_folder):
        # TODO
        raise NotImplementedError()


## THIS CODE IS MEANT BE USED ONLY AS AN EXAMPLE!!
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lung registration')
    parser.add_argument('--operation', help="TRAIN / TEST, etc.", type=str, required=True, choices=['TRAIN', 'TEST'])
    parser.add_argument('--output_folder', type=str, help="Program output logging folder (additional)")
    parser.add_argument('--params_file', type=str, help="Parameters file")

    args = parser.parse_args()

    current_folder = os.path.dirname(os.path.realpath(__file__))
    default_output_folder = os.path.realpath(os.path.join(current_folder, "..", "output"))
    output_folder = args.output_folder if args.output_folder else default_output_folder

    try:
        if not os.path.isdir(output_folder):
            print ("Creating output folder " + output_folder)
            os.makedirs(output_folder)
    except:
        # Default output folder
        output_folder = default_output_folder
        if not os.path.isdir(output_folder):
            print ("Creating output folder " + output_folder)
            os.makedirs(output_folder)

    if args.params_file:
        with open(args.params_file, 'r') as f:
            parameters_dict = json.loads(f.read())
        output_folder = os.path.join(output_folder, "{}_{}".format(parameters_dict['network_description'], Utils.now()))
        os.makedirs(output_folder)

    Utils.configure_loggers(default_level=logging.INFO, log_file=os.path.join(output_folder, "output.log"),
                            file_logging_level=logging.DEBUG)

    e = Engine(output_folder)

    if args.operation == 'TRAIN':
        parameters_dict = Utils.read_parameters_dict(args.params_file)
        e.train(parameters_dict)
    elif args.operation == 'TEST':
        pass
