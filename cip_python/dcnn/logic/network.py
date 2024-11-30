import os

from tensorflow.python.keras import backend as K

class Network(object):
    def __init__(self, xs_sizes, ys_sizes):
        """
        Constructor
        :param xs_sizes: tuple of int-tuples. Each position i of the tuple contains the shape for the input i of the network
        :param ys_sizes: tuple of int-tuples. Each position j of the tuple contains the shape for the output j of the network
        """
        self._xs_sizes_ = xs_sizes
        self._ys_sizes_ = ys_sizes
        self._model_ = None     # Keras model
        self._expected_input_values_range_ = None
        # Tuple of min-max expected range for input values (ex: (0, 1))

    @property
    def expected_input_values_range(self):
        return self._expected_input_values_range_

    def build_model(self, compile_model, optimizer=None, loss_function=None, loss_weights=None, additional_metrics=None,
                    pretrained_weights_file_path=None):
        """
        Create a new model from scratch
        If use_keras_api==False (default), the model will be built from scratch.
        Otherwise, load the model using keras API

        Args:
            compile_model: bool. Compile the model (needed for training)
            optimizer: keras Optimizer. Used for training
            loss_function: "pointer" to a function (custom metric) or string (standard keras metric).
                  It can be a single object or a list (in case of having more than one loss)
            loss_weights: list of weights for the losses (default: [1.0])
            additional_metrics: list of "pointers" to functions or strings.
            pretrained_weights_file_path: str. Path to an existing weights file path
        Returns:
            Keras Model
            :param loss_weights:
        """
        self._model_ = self._build_model_()
        # Check if there is a weights file path
        if pretrained_weights_file_path:
            # Load previously saved weights
            print ("Loading weights from {}...".format(pretrained_weights_file_path))
            self._model_.load_weights(pretrained_weights_file_path, by_name=True)

        if compile_model:
            assert optimizer is not None, "An optimizer is needed to compile the model"
            assert loss_function is not None, "At least a loss function is needed to compile the model"
            if loss_weights is None and isinstance(loss_function, list):
                loss_weights = [1.0] * len(loss_function)
            self.model.compile(optimizer=optimizer, loss=loss_function, loss_weights=loss_weights,
                               metrics=additional_metrics)
        return self._model_

    # @classmethod
    # def network_from_keras_file(cls, keras_model_file):
    #     """
    #     Create a Network object using a Keras file.
    #     This network CANNOT be used as it is for training purposes (metrics needed for optimization)
    #     :param keras_model_file: str. Path to a json model file
    #     :return:  Network object with model
    #     """
    #     assert os.path.isfile(keras_model_file), "Model file not found ({})".format(keras_model_file)
    #
    #     try:
    #         # Preferred method
    #         model = kmodels.load_model(keras_model_file, compile=False)
    #     except:
    #         # Alternative method. Use hdf5 + json keras files
    #         json_file = keras_model_file.replace(os.path.basename(keras_model_file), "kerasModel.json")
    #         print ("Model could note be loaded. Reading config from json file {}".format(json_file))
    #         if not os.path.exists(json_file):
    #             model = None
    #         else:
    #             with open(json_file, 'rb') as f:
    #                 js = f.read()
    #             model = kmodels.model_from_json(js)
    #             model.load_weights(keras_model_file)
    #
    #     net = cls()
    #     net._model_ = model
    #     # Get the inputs/outputs sizes
    #     net._xs_sizes_ = list(map(lambda l: l.shape.as_list()[1:], model.inputs))
    #     net._ys_sizes_ = list(map(lambda l: l.shape.as_list()[1:], model.outputs))
    #
    #     return net

    @property
    def model(self):
        """
        Keras model object
        """
        return self._model_

    @property
    def xs_sizes(self):
        return self._xs_sizes_

    @property
    def ys_sizes(self):
        return self._ys_sizes_

    def get_xs_ys_size(self):
        """
        Get a tuple of tuples with the input and the output sizes
        :return: 2-tuple of int-tuples
        """
        return self._xs_sizes_, self._ys_sizes_

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

    def predict(self, input_data):
        """
        Predict input data (with the possibility to preprocess the data to adapt them to the current format)
        :param input_data: list of numpy arrays
        :return: list of numpy arrays (model prediction)
        """
        if self.model is None:
            raise Exception("The model has not been created. Please call 'build_model' function first")

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

    # def format_data_to_network(self, xs, ys, inplace=True):
    #     """
    #     Adjust the input/output data to a format that is going to be understood by the network.
    #     By default, modify the original data for efficiency (inplace=True)
    #     :param xs: numpy array (or list of numpy array) that contains the input data
    #     :param ys: numpy array (or list of numpy array) that contains the labels for an image in the original format.
    #                    If None, the labels are ignored and only an image is returned
    #     :param inplace: bool. When True, the transformation will be made in place over the inputs/outputs for efficiency
    #     :return: if inplace==True, return None (the original parameters will be modified).
    #              Otherwise, return a tuple of lists with the transformed inputs/outputs
    #     """
    #     raise NotImplementedError("This method must be implemented in a child class")
