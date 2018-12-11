import os
import inspect

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
        self.parameters_dict = parameters_dict
        self.metrics_manager = Metrics()
        self._model_ = None

    @property
    def model(self):
        """
        Keras model object
        """
        return self._model_

    @property
    def xs_sizes(self):
        """
        Tuple (single input) or tuple of tuples (multiinput) with the size/s of the network input
        :return:
        """
        return self._xs_sizes_

    @property
    def ys_sizes(self):
        """
        Tuple (single output) or tuple of tuples (multioutput) with the size/s of the network output
        """
        return self._ys_sizes_

    def build_model(self, compile_model, use_keras_api=False):
        """
        Create a new model.
        If use_keras_api==False (default), the model will be built from scratch.
        Otherwise, load the model using keras API

        Args:
            parameters_dict: dictionary of parameters
            compile_model: bool. Compile the model (needed for training)
            use_keras_api: bool. Use Keras API to load the model directly instead of building it from scratch

        Returns:
            Keras Model
        """
        # self.__class__.parameters_dict = parameters_dict
        model_path = Utils.get_param('previously_existing_model_path', self.parameters_dict)
        if use_keras_api:
            assert model_path is not None, "The model path cannot be None if you are using the Keras API!"
            model = self._load_keras_existing_model_(model_path)
            if model is None:
                raise Exception("The model could not be built from the file {}".format(model_path))
        else:
            # Build the model from scratch
            model = self._build_model_()
            if model_path:
                # Load previously saved weights
                model.load_weights(model_path, by_name=True)

        if compile_model:
            self.compile_model(model)

        self._model_ = model
        return model

    def _load_keras_existing_model_(self, keras_model_file):
        """
        Load a model from an existing keras hdf5 file
        Args:
            keras_model_file:

        Returns:
            Keras model
        """
        if not os.path.exists(keras_model_file):
            raise Exception("{} not found".format(keras_model_file))

        try:
            # Preferred method
            model = kmodels.load_model(keras_model_file, self.get_metrics())
        except:
            # Alternative method. Use hdf5 + json keras files
            json_file = keras_model_file.replace("model.hdf5", "kerasModel.json")
            if not os.path.exists(json_file):
                model = None
            else:
                with open(json_file, 'rb') as f:
                    js = f.read()
                model = kmodels.model_from_json(js, custom_objects=self.get_metrics())
                model.load_weights(keras_model_file)

        return model

    def compile_model(self, model):
        """
        Compile the current model with the current parameters_dict for optimizer and metrics
        Args:
            model: keras model
            parameters_dict: dictionary of parameters

        Returns:

        """
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


        metrics = self.get_metrics()
        # Extract the loss function (first element)
        loss = metrics.pop(0)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_xs_ys_size(self):
        """
        Get a tuple of tuples with the input and the output sizes
        :return: 2-tuple of int-tuples
        """
        return self.xs_sizes, self.ys_sizes

    def get_metrics(self):
        """
        Obtain the loss function and the additional metric functions that are being used.
        Each element in the list can be either a string representing a regular Keras function or a "pointer" to
        a customized metric (as the ones defined in "keras.py".
        In the case of metrics that contain parameters, two different syntaxes are allowed:
            - metric;param1;param2
            - metric;param1=value1;param2=value2
        The first element in the result list will always be the loss function
        :return: List of functions or keras functions names.
        """
        # Check if the loss is defined in our custom metrics. Otherwise we will use the text that should be recognized as
        # a Keras builtin function
        parameters_dict = self.parameters_dict
        metric_names = []
        metric_functions = []
        # First metric will be the loss function
        metric_names.append(Utils.get_param('loss', parameters_dict))
        # Check for additional metrics
        if Utils.get_param('metrics', parameters_dict):
            metric_names.extend(Utils.get_param('metrics', parameters_dict))
        for metric_str in metric_names:
            components = metric_str.split(';')
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
                metric = self.metrics_manager.get_metric(metric_name, *args, **kwargs)
                metric_functions.append(metric)
            else:
                # Assume this is a regular Keras metric
                metric_functions.append(metric_name)
        return metric_functions


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

    def _build_model_(self):
        """
        Build the network structure

        Returns:
            Keras model
        """
        raise NotImplementedError("This method should be implemented in a child class")

    def get_last_layer(self):
        """
        Get the last layer of the model (for transfer learning purposes)
        Returns:

        """
        raise NotImplementedError("This method should be implemented in a child class")

    def predict(self, input_data, adjust_to_network_format=True):
        """
        Predict input data (with the possibility to preprocess the data to adapt them to the current format)
        :param input_data:
        :return:
        """
        if self.model is None:
            raise Exception("The model has not been created. Please call 'build_model' function first")
        if adjust_to_network_format:
            input_data, _ = self.format_data_to_network(input_data, None)
        return self.model.predict(input_data)

    def gradients(self, input_images):
        """
        Compute the gradients for some input data
        :param input_images: numpy array in the network shape (Batch X [Dims] X Channels)
        :return: list of arrays with the gradients for all the trainable layers
        """
        if self.model is None:
            raise Exception("The model has not been created. Please call 'build_model' function first")
        grad = K.gradients(self.model.output, self.model.trainable_weights)
        sess = K.get_session()
        ev_grad = sess.run(grad, feed_dict={self.model.input: input_images})
        return ev_grad

    #######################################################################################################
    # DATA CONVERSION UTILS
    #######################################################################################################
    def format_data_to_network(self, xs, ys):
        """
        Adjust the input/output data to a format that is going to be understood by the network
        :param xs: numpy array (or list of numpy array) that contains the input data
        :param ys: numpy array (or list of numpy array) that contains the labels for an image in the original format.
                       If None, the labels are ignored and only an image is returned
        :return: tuple with the new (input/s, output/s) if output != None.
                 Otherwise, just input/s with the new format
        """
        raise NotImplementedError("This method must be implemented in a child class")

