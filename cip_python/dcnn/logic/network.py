import os
import inspect
from future.utils import listvalues

from keras import backend as K
import keras.optimizers as optimizers
import keras.models as kmodels

from . import Metrics, Utils

class Network(object):
    def __init__(self, parameters_dict=None):
        """
        Constructor
        :param parameters_dict: dictionary of parameters to build the network
        """
        self._xs_sizes_ = None
        self._ys_sizes_ = None
        self._model_ = None
        self._loss_function_ = None
        self._additional_metrics_ = None

        self.parameters_dict = parameters_dict
        self.metrics_manager = Metrics()

    def build_model(self, compile_model, optimizer=None):
        """
        Create a new model from scratch
        If use_keras_api==False (default), the model will be built from scratch.
        Otherwise, load the model using keras API

        Args:
            compile_model: bool. Compile the model (needed for training)
            optimizer: Keras Optimizer. Needed to train the model. If None, it will be initialized from parameters_dict

        Returns:
            Keras Model
        """
        self._model_ = self._build_model_()
        # Check if there is a weights file
        model_path = Utils.get_param('previously_existing_model_path', self.parameters_dict)
        if model_path:
            # Load previously saved weights
            self._model_.load_weights(model_path, by_name=True)

        if compile_model:
            self.compile_model(optimizer=optimizer)

        return self._model_

    @classmethod
    def network_from_keras_file(cls, keras_model_file):
        """
        Create a Network object using a Keras file.
        This network cannot be used as it is for training purposes (metrics needed for optimization)
        :param keras_model_file: str. Path to a json model file
        :return:  Network object with model, input sizes and output sizes initialized (ready for testing but not training!)
        """
        assert os.path.isfile(keras_model_file), "Model file not found ({})".format(keras_model_file)
        try:
            # Preferred method
            model = kmodels.load_model(keras_model_file, compile=False)
        except:
            # Alternative method. Use hdf5 + json keras files
            json_file = keras_model_file.replace(os.path.basename(keras_model_file), "kerasModel.json")
            print ("Model could note be loaded. Reading config from json file {}".format(json_file))
            if not os.path.exists(json_file):
                model = None
            else:
                with open(json_file, 'rb') as f:
                    js = f.read()
                model = kmodels.model_from_json(js)
                model.load_weights(keras_model_file)

        net = cls()
        net._model_ = model
        # Get the inputs/outputs sizes
        net._xs_sizes_ = list(map(lambda l: l.shape.as_list()[1:], model.inputs))
        net._ys_sizes_ = list(map(lambda l: l.shape.as_list()[1:], model.outputs))

        return net

    @property
    def model(self):
        """
        Keras model object
        """
        return self._model_

    def get_xs_ys_size(self):
        """
        Get a tuple of tuples with the input and the output sizes
        :return: 2-tuple of int-tuples
        """
        return self._xs_sizes_, self._ys_sizes_

    def set_metrics(self, loss, additional_metrics=None):
        """
        Set manually a list of metrics to be used during the training (strings or pointers to functions)
        :param loss: string (keras function) or pointer to a custom function that will be used as a loss
        :param additional_metrics: list of [str / "pointer" to functions]
        """
        self._loss_function_ = loss
        self._additional_metrics_ = additional_metrics

    def get_learning_rate(self):
        """
        Get the current learning rate
        Returns:

        """
        return K.get_value(self.model.optimizer.lr)

    def set_learning_rate(self, lr):
        """
        Update the learning rate of the model
        Args:
            lr: float. New learning rate
        """
        K.set_value(self.model.optimizer.lr, lr)

    def compile_model(self, optimizer=None):
        """
        Compile the current model.
        :param optimizer: Keras optimizer. When None, the optimizer will be obtained from parameters_dict
        """
        assert self.model is not None, "Model not initialized!"
        if optimizer is None:
            # Search in parameters_dict
            with K.name_scope('optimization') as scope:
                lr = self.parameters_dict['learning_rate_initial']
                if self.parameters_dict['optimizer'] == 'Adam':
                    if 'optim_adam_beta_1' in self.parameters_dict:
                        optimizer = optimizers.Adam(lr=lr, beta_1=self.parameters_dict['optim_adam_beta_1'],
                                                    beta_2=self.parameters_dict['optim_adam_beta_2'])
                    else:
                        optimizer = optimizers.Adam(lr=lr)

                elif self.parameters_dict['optimizer'] == 'SGD':
                    if 'optim_sgd_momentum' in self.parameters_dict:
                        optimizer = optimizers.SGD(lr=lr, momentum=self.parameters_dict['optim_sgd_momentum'],
                                                   nesterov=self.parameters_dict['optim_sgd_nesterov'])
                    else:
                        optimizer = optimizers.SGD(lr=lr)
                else:
                    # Unknown optimizer. Try just to initialize with the Keras string and learning rate
                    optimizer = eval("optimizers.{}(lr=lr)".format(self.parameters_dict['optimizer']))

        loss, metrics = self._get_loss_and_additional_metrics_()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def predict(self, input_data, adjust_to_network_format=True):
        """
        Predict input data (with the possibility to preprocess the data to adapt them to the current format)
        :param input_data: list of numpy arrays
        :return: list of numpy arrays (model prediction)
        """
        if self.model is None:
            raise Exception("The model has not been created. Please call 'build_model' function first")
        if adjust_to_network_format:
            input_data, _ = self.format_data_to_network(input_data, None)
        return self.model.predict(input_data)

    def gradients(self, input_data):
        """
        Compute the gradients for some input data
        :param input_data: list of numpy array in the network shape (Batch X [Dims] X Channels)
        :return: list of arrays with the gradients for all the trainable layers
        """
        if self.model is None:
            raise Exception("The model has not been created. Please call 'build_model' function first")
        grad = K.gradients(self.model.output, self.model.trainable_weights)
        sess = K.get_session()
        ev_grad = sess.run(grad, feed_dict={self.model.input: input_data})
        return ev_grad

    def _get_loss_and_additional_metrics_(self):
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
        if self._loss_function_ is not None:
            # Metrics have been already initialized
            return self._loss_function_, self._additional_metrics_

        if self.parameters_dict is None:
            # No metrics available
            return None

        parameters_dict = self.parameters_dict
        metric_names = []
        metric_functions = []
        # First metric will be the loss function
        loss_function = Utils.get_param('loss', parameters_dict)
        assert loss_function is not None, "Loss function not found! Make sure that parameter 'loss' is initialized"
        metric_names.append(loss_function)
        # Check for additional metrics
        additional_metrics_list = Utils.get_param('metrics', parameters_dict)
        if additional_metrics_list:
            metric_names.extend(additional_metrics_list)
        for metric_str in metric_names:
            components = metric_str.split(',')
            metric_name = components[0]
            if self.metrics_manager.contains_metric(metric_name):
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
                metric = self.metrics_manager.get_metric(metric_name, *args, **kwargs)
                metric_functions.append(metric)
            else:
                # Assume this is a regular Keras metric. Save the string
                metric_functions.append(metric_name)

        # The first element in the list is the loss function
        self._loss_function_ = metric_functions.pop(0)
        self._additional_metrics_ = metric_functions

        return self._loss_function_, self._additional_metrics_


    #######################################################################################################
    # METHODS THAT MUST BE IMPLEMENTED IN CHILD CLASSES
    #######################################################################################################
    def _build_model_(self):
        """
        Build the network structure

        Returns:
            Keras model
        """
        raise NotImplementedError("This method should be implemented in a child class")

    def format_data_to_network(self, xs, ys, inplace=True):
        """
        Adjust the input/output data to a format that is going to be understood by the network.
        By default, modify the original data for efficiency (inplace=True)
        :param xs: numpy array (or list of numpy array) that contains the input data
        :param ys: numpy array (or list of numpy array) that contains the labels for an image in the original format.
                       If None, the labels are ignored and only an image is returned
        :param inplace: bool. When True, the transformation will be made in place over the inputs/outputs for efficiency
        :return: if inplace==True, return None (the original parameters will be modified).
                 Otherwise, return a tuple of lists with the transformed inputs/outputs
        """
        raise NotImplementedError("This method must be implemented in a child class")

    def generate_augmented_data_point(self, input_data, output_data):
        """
        Generate n input/output augmented data points from an input/output data point
        :param input_data: list of numpy arrays (inputs)
        :param output_data: list of numpy arrays (outputs)
        :return: tuple with augmented inputs / outputs
        """
        raise NotImplementedError("This method must be implemented in a child class")