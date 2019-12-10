import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense

import cip_python.dcnn.example.utils.custom_metrics as custom_metrics


class AirwayFlowNeuralNetwork:
    def __init__(self):
        self.model = None

    def build_and_compile_network(self, network_parameters):
        par = network_parameters

        input_layer = Input(shape=(12,))
        dense1 = Dense(12, kernel_initializer=par.kernel_initializer, activation=par.dense_activation)(input_layer)
        dense2 = Dense(10, kernel_initializer=par.kernel_initializer, activation='relu')(dense1)
        output_layer = Dense(7, kernel_initializer=par.kernel_initializer, activation=par.output_activation)(dense2)

        self.model = Model(inputs=[input_layer], outputs=[output_layer])

        optimizer_ = self._init_optimizer_(par)
        self.model.compile(loss=custom_metrics.precision_accuracy_loss(d=0.0025, c_prec=2.5), optimizer=optimizer_)

    @staticmethod
    def _init_optimizer_(network_parameters):
        if network_parameters.optimizer_type == 'SGD':
            opt_momentum = 0.002
            return optimizers.SGD(lr=network_parameters.learning_rate, decay=network_parameters.decay,
                                  momentum=opt_momentum, nesterov=False)
        elif network_parameters.optimizer_type == 'NESTEROV':
            opt_momentum = 0.002
            return optimizers.SGD(lr=network_parameters.learning_rate, decay=network_parameters.decay,
                                  momentum=opt_momentum, nesterov=True)
        elif network_parameters.optimizer_type == 'ADAM':
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 1e-08
            return optimizers.Adam(lr=network_parameters.learning_rate, decay=network_parameters.decay, beta_1=beta_1,
                                   beta_2=beta_2, epsilon=epsilon)


class NetworkParameters(object):
    """
        Parameters for building/training the network
    """
    def __init__(self, kernel_initializer='normal', optimizer_type='ADAM', learning_rate=5e-4,
                 dense_activation='relu', output_activation='linear', decay=0.,
                 use_early_stopping=True, batch_size=32, nb_epochs=100, nb_augmented_data=2, verbosity=2):

        self.kernel_initializer = kernel_initializer
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.decay = decay
        self.use_early_stopping = use_early_stopping
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.nb_augmented_data = nb_augmented_data
        self.verbosity = verbosity
