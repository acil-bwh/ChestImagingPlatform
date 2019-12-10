from cip_python.dcnn.example.airway_flow_neural_network import NetworkParameters
from cip_python.dcnn.example.airway_flow_engine import Engine


def train(tr_dataset, operation='TRAIN', out_folder=None, network_description=None, save_arch=False):
    dense_act = 'relu'
    kernel_init = 'normal'
    output_act = 'linear'

    # Optimizer and learning rate parameters
    decay = 0.0
    optimizer = 'ADAM'
    lr = 5e-4

    # # EarlyStopping
    early_stopping = False

    # Training parameters
    batch_sz = 300
    nb_augmented_data = 49
    nb_epochs = 150

    # Verbosity
    verb = 2

    # Object for network parameters
    net_parameters = NetworkParameters(dense_activation=dense_act,
                                       kernel_initializer=kernel_init,
                                       output_activation=output_act,
                                       optimizer_type=optimizer,
                                       learning_rate=lr,
                                       decay=decay,
                                       use_early_stopping=early_stopping,
                                       batch_size=batch_sz,
                                       nb_epochs=nb_epochs,
                                       nb_augmented_data=nb_augmented_data,
                                       verbosity=verb)

    ee = Engine(output_folder=out_folder, network_desc=network_description)
    if operation == 'TRAIN':
        train_history = ee.train_network(tr_dataset, network_params=net_parameters, save_arch=save_arch)
        print ('    Network trained...')
        ee.save_network_model()
        print ('    Model saved')
        return train_history
    else:
        ee.train_CV_network(tr_dataset)
        return 0


def test(test_dataset, out_folder, network_description, test_data_dauA=None, test_data_dauB=None):
    ee = Engine(output_folder=out_folder, network_desc=network_description)
    if test_data_dauA is None or test_data_dauB is None:
        ee.test_single_bifurcation(test_dataset)
    else:
        ee.test_double_bifurcation(test_dataset, test_data_dauA, test_data_dauB)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CNN method for airway flow (CFD)')
    parser.add_argument('--operation', dest='operation', help="Operations allowed: [TRAIN|TEST|CV]", type=str,
                        required=True)
    parser.add_argument("--train_dataset", help='Input train dataset (.csv)', dest="train_data", metavar='<string>',
                        required=False)
    parser.add_argument('--output_folder', dest='out_folder', help='Folder to save algorithm outputs. Required for both'
                                                                   ' TRAIN and TEST operations',
                        required=False, type=str, default=None)
    parser.add_argument("--network_description", help='Network description. Required for both TRAIN and TEST '
                                                      'operations. Format should be: v<version_number>_#<GitHubTag>',
                        dest="net_desc", metavar='<string>', default=None, required=False)
    parser.add_argument("--test_dataset", help='.csv file with the dataset for testing. Required for both TRAIN '
                                               'and TEST operations', dest="test_data",  metavar='<string>',
                        default=None, required=False)
    parser.add_argument("--test_dataset_dauA", help='.csv file with the dataset for testing double bifurcation. '
                                                    'If not specified, only the test_dataset will be used for testing',
                        dest="test_data_dauA", metavar='<string>',
                        default=None, required=False)
    parser.add_argument("--test_dataset_dauB", help='.csv file with the dataset for testing double bifurcation. '
                                                    'If not specified, only the test_dataset will be used for testing',
                        dest="test_data_dauB", metavar='<string>',
                        default=None, required=False)
    parser.add_argument('-save_arch', dest='save_arch', help="Flag to save the network architecture",
                        action='store_true')
    op = parser.parse_args()

    if op.operation == 'TRAIN' or op.operation == 'CV':
        print ('    Training the net...')
        training_history = train(op.train_data, operation=op.operation, out_folder=op.out_folder,
                                 network_description=op.net_desc, save_arch=op.save_arch)

    test(op.test_data, op.out_folder, op.net_desc, op.test_data_dauA, op.test_data_dauB)
