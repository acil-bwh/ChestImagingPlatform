from __future__ import division
import os
import logging
import json
import time
import datetime
import warnings
import numpy as np
import SimpleITK as sitk
import nrrd
import h5py

import tensorflow as tf
import tensorflow.keras.backend as K

from cip_python.dcnn.data import H5Manager, DataProcessing, DataAugmentor
from cip_python.dcnn.logic import Utils, Engine
from cip_python.input_output import ImageReaderWriter

from cip_python.dcnn.projects.low_to_high_dose_filter.networks import *


class LowDoseToHighDoseEngine(Engine):
    """
    Class that controls the global processes of the CNN (training, validating, testing)
    """

    def __init__(self, parameters_dict, output_folder=None):
        """
        Constructor
        :param output_folder: all the output folders will be created here (log, models, training files, etc.)
        """
        Engine.__init__(self, parameters_dict, output_folder)

        self.model = None
        self.parameters_dict = parameters_dict
        if parameters_dict is not None:
            self.network_description = parameters_dict['network_description']
        min_delta = 1e-4
        self.monitor_op = lambda a, b: np.less(a, b - min_delta)

    @property
    def model_folder(self):
        """
        Folder where the outputs of the algorithm will be stored.
        Create the folder if it does not exist.
        """
        mf = os.path.join(self.output_folder, 'LowToHighDoseModels')
        if not os.path.isdir(mf):
            os.mkdir(mf)

        mf = os.path.join(mf, self.network_description)
        if not os.path.isdir(mf):
            os.mkdir(mf)

        return mf

    @property
    def l2h_results_folder(self):
        """
        Folder where the fake to real results of the algorithm will be stored.
        Create the folder if it does not exist.
        """
        rf = os.path.join(self.output_folder, 'LowToHighDoseResults')
        if not os.path.isdir(rf):
            os.mkdir(rf)

        rf = os.path.join(rf, self.network_description)
        if not os.path.isdir(rf):
            os.mkdir(rf)
            print('    {} folder created'.format(rf))

        rf = os.path.join(rf, 'Low2High')
        if not os.path.isdir(rf):
            os.mkdir(rf)
            print('    {} folder created'.format(rf))

        return rf

    @property
    def h2l_results_folder(self):
        """
        Folder where the fake to real results of the algorithm will be stored.
        Create the folder if it does not exist.
        """
        rf = os.path.join(self.output_folder, 'LowToHighDoseResults')
        if not os.path.isdir(rf):
            os.mkdir(rf)

        rf = os.path.join(rf, self.network_description)
        if not os.path.isdir(rf):
            os.mkdir(rf)
            print('    {} folder created'.format(rf))

        rf = os.path.join(rf, 'High2Low')
        if not os.path.isdir(rf):
            os.mkdir(rf)
            print ('    {} folder created'.format(rf))

        return rf

    @property
    def model_name_DLH(self):
        model_name = 'Discriminator_L2H_' + self.network_description
        return model_name

    @property
    def model_name_DHL(self):
        model_name = 'Discriminator_H2L_' + self.network_description
        return model_name

    @property
    def model_name_GLH(self):
        model_name = 'Generator_L2H_' + self.network_description
        return model_name

    @property
    def model_name_GHL(self):
        model_name = 'Generator_H2L_' + self.network_description
        return model_name

    @property
    def model_DLH_summary_file_path(self):
        """
        Path to save the Keras model summary
        Returns:
            Path to the file
        """
        model_summary_name = self.model_name_DLH + '_summary.txt'
        return os.path.join(self.model_folder, model_summary_name)

    @property
    def model_DHL_summary_file_path(self):
        """
        Path to save the Keras model summary
        Returns:
            Path to the file
        """
        model_summary_name = self.model_name_DHL + '_summary.txt'
        return os.path.join(self.model_folder, model_summary_name)

    @property
    def model_GLH_summary_file_path(self):
        """
        Path to save the Keras model summary
        Returns:
            Path to the file
        """
        model_summary_name = self.model_name_GLH + '_summary.txt'
        return os.path.join(self.model_folder, model_summary_name)

    @property
    def model_GHL_summary_file_path(self):
        """
        Path to save the Keras model summary
        Returns:
            Path to the file
        """
        model_summary_name = self.model_name_GHL + '_summary.txt'
        return os.path.join(self.model_folder, model_summary_name)

    @property
    def model_DLH_architecture_file_path(self):
        """
        Json file that represents the architecture of the network
        Returns:
            Path to the file
        """
        arch_name = self.model_name_DLH + '_Architecture.json'
        return os.path.join(self.model_folder, arch_name)

    @property
    def model_DHL_architecture_file_path(self):
        """
        Json file that represents the architecture of the network
        Returns:
            Path to the file
        """
        arch_name = self.model_name_DHL + '_Architecture.json'
        return os.path.join(self.model_folder, arch_name)

    @property
    def model_GLH_architecture_file_path(self):
        """
        Json file that represents the architecture of the network
        Returns:
            Path to the file
        """
        arch_name = self.model_name_GLH + '_Architecture.json'
        return os.path.join(self.model_folder, arch_name)

    @property
    def model_GHL_architecture_file_path(self):
        """
        Json file that represents the architecture of the network
        Returns:
            Path to the file
        """
        arch_name = self.model_name_GHL + '_Architecture.json'
        return os.path.join(self.model_folder, arch_name)

    @property
    def parameters_file_path(self):
        return os.path.join(self.model_folder, 'Parameters_' + self.network_description + '.json')

    def save_DLH_summary(self, model):
        """
        Save the model Keras summary in a txt file
        :param model:
        """
        f = open(self.model_DLH_summary_file_path, 'w')

        def print_fn(s):
            f.write(s + "\n")

        model.summary(print_fn=print_fn)
        f.close()

    def save_DHL_summary(self, model):
        """
        Save the model Keras summary in a txt file
        :param model:
        """
        f = open(self.model_DHL_summary_file_path, 'w')

        def print_fn(s):
            f.write(s + "\n")

        model.summary(print_fn=print_fn)
        f.close()

    def save_GLH_summary(self, model):
        """
        Save the model Keras summary in a txt file
        :param model:
        """
        f = open(self.model_GLH_summary_file_path, 'w')

        def print_fn(s):
            f.write(s + "\n")

        model.summary(print_fn=print_fn)
        f.close()

    def save_GHL_summary(self, model):
        """
        Save the model Keras summary in a txt file
        :param model:
        """
        f = open(self.model_GHL_summary_file_path, 'w')

        def print_fn(s):
            f.write(s + "\n")

        model.summary(print_fn=print_fn)
        f.close()

    def save_best_network_models(self, g_LH, g_HL, d_LH, d_HL, save_weights_only=True):
        generator_LH_path = os.path.join(self.model_folder, self.model_name_GLH + '_best.h5')
        generator_HL_path = os.path.join(self.model_folder, self.model_name_GHL + '_best.h5')
        discriminator_LH_path = os.path.join(self.model_folder, self.model_name_DLH + '_best.h5')
        discriminator_HL_path = os.path.join(self.model_folder, self.model_name_DHL + '_best.h5')

        if save_weights_only:
            g_LH.save_weights(generator_LH_path, overwrite=True)
            g_HL.save_weights(generator_HL_path, overwrite=True)
            d_LH.save_weights(discriminator_LH_path, overwrite=True)
            d_HL.save_weights(discriminator_HL_path, overwrite=True)
        else:
            g_LH.save(generator_LH_path, overwrite=True)
            g_HL.save(generator_HL_path, overwrite=True)
            d_LH.save(discriminator_LH_path, overwrite=True)
            d_HL.save(discriminator_HL_path, overwrite=True)

        self.save_parameters_to_model_h5(self.parameters_dict, generator_LH_path)
        self.save_parameters_to_model_h5(self.parameters_dict, generator_HL_path)
        self.save_parameters_to_model_h5(self.parameters_dict, discriminator_LH_path)
        self.save_parameters_to_model_h5(self.parameters_dict, discriminator_HL_path)

    def save_network_models(self, g_LH, g_HL, d_LH, d_HL, save_weights_only=True):
        generator_LH_path = os.path.join(self.model_folder, self.model_name_GLH + '.h5')
        generator_HL_path = os.path.join(self.model_folder, self.model_name_GHL + '.h5')
        discriminator_LH_path = os.path.join(self.model_folder, self.model_name_DLH + '.h5')
        discriminator_HL_path = os.path.join(self.model_folder, self.model_name_DHL + '.h5')

        if save_weights_only:
            g_LH.save_weights(generator_LH_path, overwrite=True)
            g_HL.save_weights(generator_HL_path, overwrite=True)
            d_LH.save_weights(discriminator_LH_path, overwrite=True)
            d_HL.save_weights(discriminator_HL_path, overwrite=True)
        else:
            g_LH.save(generator_LH_path, overwrite=True)
            g_HL.save(generator_HL_path, overwrite=True)
            d_LH.save(discriminator_LH_path, overwrite=True)
            d_HL.save(discriminator_HL_path, overwrite=True)

        self.save_parameters_to_model_h5(self.parameters_dict, generator_LH_path)
        self.save_parameters_to_model_h5(self.parameters_dict, generator_HL_path)
        self.save_parameters_to_model_h5(self.parameters_dict, discriminator_LH_path)
        self.save_parameters_to_model_h5(self.parameters_dict, discriminator_HL_path)

    @staticmethod
    def cycle_variables_2d(netG1, netG2):
        real_input = netG1.inputs[0]
        fake_output = netG1.outputs[0]
        rec_input = netG2([fake_output])
        fn_generate = K.function([real_input], [fake_output, rec_input])
        return real_input, fake_output, rec_input, fn_generate

    @staticmethod
    def cycle_variables_l2h(L2H_net, H2L_net):
        real_input = L2H_net.inputs[0]
        fake_output = L2H_net.outputs[0]
        rec_input = H2L_net([fake_output])
        fn_generate = K.function([real_input], [fake_output, rec_input])
        return real_input, fake_output, rec_input, fn_generate

    @staticmethod
    def cycle_variables_h2l(H2L_net, L2H_net):
        def tile_layer(n):  # Tile image Layer
            def func(x):
                return tf.tile(x, n)

            return tf.keras.layers.Lambda(func)

        real_input = H2L_net.inputs[0]
        fake_output = H2L_net.outputs[0]
        rec_input = L2H_net([tile_layer(tf.constant([1, 1, 1, 5, 1]))(fake_output)])
        fn_generate = K.function([real_input], [fake_output, rec_input])
        return real_input, fake_output, rec_input, fn_generate

    def D_loss_2d(self, netD, real, fake, rec):
        if self.parameters_dict['use_lsgan']:
            loss_fn = lambda output, target: K.mean(K.abs(K.square(output - target)))
        else:
            loss_fn = lambda output, target: -K.mean(
                K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))  # Eq 1 in paper

        output_real = netD([real])
        output_fake = netD([fake])
        loss_D_real = loss_fn(output_real, K.ones_like(output_real))
        loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
        loss_G = loss_fn(output_fake, K.ones_like(output_fake))
        loss_D = loss_D_real + loss_D_fake
        loss_cyc = K.mean(K.abs(rec - real))
        return loss_D, loss_G, loss_cyc

    def D_loss_L2H(self, L2H_net, real, fake, rec):
        def tile_layer(n):  # Tile image Layer
            def func(x):
                return tf.tile(x, n)

            return tf.keras.layers.Lambda(func)

        if self.parameters_dict['use_lsgan']:
            loss_fn = lambda output, target: K.mean(K.abs(K.square(output - target)))
        else:
            loss_fn = lambda output, target: -K.mean(
                K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))  # Eq 1 in paper

        output_real = L2H_net([real])
        output_fake = L2H_net([tile_layer(tf.constant([1, 1, 1, 9, 1]))(fake)])
        loss_D_real = loss_fn(output_real, K.ones_like(output_real))
        loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
        loss_G = loss_fn(output_fake, K.ones_like(output_fake))
        loss_D = loss_D_real + loss_D_fake
        loss_cyc = K.mean(K.abs(rec - real))
        return loss_D, loss_G, loss_cyc

    def D_loss_H2L(self, H2L_net, real, fake, rec):
        def tile_layer(n):  # Tile image Layer
            def func(x):
                return tf.tile(x, n)

            return tf.keras.layers.Lambda(func)

        if self.parameters_dict['use_lsgan']:
            loss_fn = lambda output, target: K.mean(K.abs(K.square(output - target)))
        else:
            loss_fn = lambda output, target: -K.mean(
                K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))  # Eq 1 in paper

        output_real = H2L_net([real])
        output_fake = H2L_net([fake])
        loss_D_real = loss_fn(output_real, K.ones_like(output_real))
        loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
        loss_G = loss_fn(output_fake, K.ones_like(output_fake))
        loss_D = loss_D_real + loss_D_fake
        loss_cyc = K.mean(K.abs(rec - real))
        return loss_D, loss_G, loss_cyc

    def train(self):
        """
        Train the network
        """
        start_time = time.time()

        Utils.configure_loggers(default_level=logging.INFO,
                                log_file=os.path.join(self.model_folder,
                                                      "{}_output.log".format(self.network_description)),
                                file_logging_level=logging.DEBUG)

        # Define loss and additional metrics
        # Build network models
        p = {}
        if 'image_shape' in self.parameters_dict:
            p['img_shape'] = self.parameters_dict['image_shape']
        if 'generator_filters' in self.parameters_dict:
            p['gf'] = self.parameters_dict['generator_filters']
        if 'discriminator_filters' in self.parameters_dict:
            p['df'] = self.parameters_dict['discriminator_filters']
        if 'lamba_cycle' in self.parameters_dict:
            p['lambda_cycle'] = self.parameters_dict['lambda_cycle']

        network = LowToHighDoseNetwork(**p)
        optimizer = self.get_optimizer(self.parameters_dict)

        img_shape = self.parameters_dict['image_shape']

        if len(img_shape) == 2:
            generator_LH = network.build_generator_unet(pretrained_weights_file_path=None)
            generator_HL = network.build_generator_unet(pretrained_weights_file_path=None)
            discriminator_LH = network.build_discriminator_2d()
            discriminator_HL = network.build_discriminator_2d()
            real_low, fake_high, rec_low, cycle_low_generate = self.cycle_variables_2d(generator_LH, generator_HL)
            real_high, fake_low, rec_high, cycle_high_generate = self.cycle_variables_2d(generator_HL, generator_LH)
            loss_D_low, loss_G_low, loss_cyc_low = self.D_loss_2d(discriminator_LH, real_low, fake_low, rec_low)
            loss_D_high, loss_G_high, loss_cyc_high = self.D_loss_2d(discriminator_HL, real_high, fake_high, rec_high)
        else:
            generator_LH = network.build_generator_3d(pretrained_weights_file_path=None, generator_type='l2h')
            generator_HL = network.build_generator_3d(pretrained_weights_file_path=None, generator_type='h2l')
            discriminator_LH = network.build_discriminator_3d(discr_type='l2h')
            discriminator_HL = network.build_discriminator_3d(discr_type='h2l')
            real_low, fake_high, rec_low, cycle_low_generate = self.cycle_variables_l2h(generator_LH, generator_HL)
            real_high, fake_low, rec_high, cycle_high_generate = self.cycle_variables_h2l(generator_HL, generator_LH)
            loss_D_low, loss_G_low, loss_cyc_low = self.D_loss_L2H(discriminator_LH, real_low, fake_low, rec_low)
            loss_D_high, loss_G_high, loss_cyc_high = self.D_loss_H2L(discriminator_HL, real_high, fake_high, rec_high)

        loss_cyc = loss_cyc_low + loss_cyc_high

        loss_G = loss_G_low + loss_G_high + self.parameters_dict['lambda_cycle'] * loss_cyc  # Eq. 3 in paper
        loss_D = loss_D_low + loss_D_high

        weightsD = discriminator_LH.trainable_weights + discriminator_HL.trainable_weights
        weightsG = generator_LH.trainable_weights + generator_HL.trainable_weights

        training_updates_D = optimizer.get_updates(loss_D, weightsD)
        D_train = K.function([real_low, real_high], [loss_D_low / 2, loss_D_high / 2], training_updates_D)
        training_updates_G = optimizer.get_updates(loss_G, weightsG)
        G_train = K.function([real_low, real_high], [loss_G_low, loss_G_high, loss_cyc], training_updates_G)

        # Save models in Keras format
        try:
            with open(self.model_DLH_architecture_file_path, 'wb') as f:
                d_A_json = discriminator_LH.to_json()

                # Reformat for a more friendly visualization
                parsed = json.loads(d_A_json)
                f.write(json.dumps(parsed, indent=4).encode())
            logging.info('Keras model saved to {}'.format(self.model_DLH_architecture_file_path))

            with open(self.model_DHL_architecture_file_path, 'wb') as f:
                d_B_json = discriminator_HL.to_json()

                # Reformat for a more friendly visualization
                parsed = json.loads(d_B_json)
                f.write(json.dumps(parsed, indent=4).encode())
            logging.info('Keras model saved to {}'.format(self.model_DHL_architecture_file_path))

            with open(self.model_GLH_architecture_file_path, 'wb') as f:
                g_LH = generator_LH.to_json()

                # Reformat for a more friendly visualization
                parsed = json.loads(g_LH)
                f.write(json.dumps(parsed, indent=4).encode())
            logging.info('Keras model saved to {}'.format(self.model_GLH_architecture_file_path))

            with open(self.model_GHL_architecture_file_path, 'wb') as f:
                g_HL = generator_HL.to_json()

                # Reformat for a more friendly visualization
                parsed = json.loads(g_HL)
                f.write(json.dumps(parsed, indent=4).encode())
            logging.info('Keras model saved to {}'.format(self.model_GHL_architecture_file_path))

        except Exception as ex:
            warnings.warn("The model could not be saved to json: {}".format(ex))

        # Save the current models summary to a txt file
        self.save_DLH_summary(discriminator_LH)
        self.save_DHL_summary(discriminator_HL)
        self.save_GLH_summary(generator_LH)
        self.save_GHL_summary(generator_HL)

        # Fill dynamic parameters_dict
        self.parameters_dict['tensorflow_version'] = tf.__version__
        self.parameters_dict['keras_version'] = tf.keras.__version__

        # Save current parameters_dict to a file
        self.save_parameters_file(parameters_dict=self.parameters_dict, file_path=self.parameters_file_path)


        # DATA
        low_dataset_manager = H5Manager(h5_file_path=self.parameters_dict['low_dataset_path'],
                                        batch_size=int(self.parameters_dict['batch_size']),
                                        train_ixs=None, validation_ixs=None, test_ixs=None,
                                        xs_dataset_names=('low_dose_CT',), ys_dataset_names=(),
                                        use_pregenerated_augmented_train_data=False)
        low_dataset_manager.generate_ixs_from_dataset('train_status')

        high_dataset_manager = H5Manager(h5_file_path=self.parameters_dict['high_dataset_path'],
                                         batch_size=int(self.parameters_dict['batch_size']),
                                         train_ixs=None, validation_ixs=None, test_ixs=None,
                                         xs_dataset_names=('high_dose_CT',), ys_dataset_names=(),
                                         use_pregenerated_augmented_train_data=False)
        high_dataset_manager.generate_ixs_from_dataset('train_status')

        training_steps_per_epoch = high_dataset_manager.get_steps_per_epoch_train()

        low_data_generator = self.keras_generator(low_dataset_manager, network,
                                                  low_dataset_manager.batch_size,
                                                  low_dataset_manager.TRAIN)

        if len(img_shape) == 2:
            high_data_generator = self.keras_generator(high_dataset_manager, network,
                                                       high_dataset_manager.batch_size,
                                                       high_dataset_manager.TRAIN)
        else:
            high_data_generator = self.keras_generator_h2l(high_dataset_manager, network,
                                                           high_dataset_manager.batch_size,
                                                           high_dataset_manager.TRAIN)

        # TRAIN
        # Calculate output shape of D (PatchGAN)
        errCyc_sum = errG_low_sum = errG_high_sum = errD_low_sum = errD_high_sum = 0

        monitor_op = np.less
        best_cycle_loss = np.Inf

        for epoch in range(self.parameters_dict['nb_epochs']):
            t0 = time.time()
            for batch_i in range(training_steps_per_epoch):
                # Training
                imgs_low, _ = next(low_data_generator)
                imgs_high, _ = next(high_data_generator)

                errD_low, errD_high = D_train([imgs_low, imgs_high])
                errD_low_sum += errD_low
                errD_high_sum += errD_high

                errG_low, errG_high, errCyc = G_train([imgs_low, imgs_high])
                errG_low_sum += errG_low
                errG_high_sum += errG_high
                errCyc_sum += errCyc

            elapsed_time = time.time() - t0
            print('[%d/%d] Loss_D (low, high): %f, %f Loss_G (low, high): %f, %f loss_cyc %f'
                  % (epoch, self.parameters_dict['nb_epochs'],
                     errD_low_sum / training_steps_per_epoch, errD_high_sum / training_steps_per_epoch,
                     errG_low_sum / training_steps_per_epoch, errG_high_sum / training_steps_per_epoch,
                     errCyc_sum / training_steps_per_epoch), 'elapsed time: {} second'.format(elapsed_time))

            # current_D_low = errD_low_sum / training_steps_per_epoch
            current_cycle_loss = errCyc_sum / training_steps_per_epoch
            if monitor_op(current_cycle_loss, best_cycle_loss):
                print('\nEpoch %05d: Cycle loss improved from %0.5f to %0.5f saving models' % (
                epoch + 1, best_cycle_loss, current_cycle_loss))
                best_cycle_loss = current_cycle_loss

                self.save_best_network_models(generator_LH, generator_HL, discriminator_LH, discriminator_HL,
                                              save_weights_only=True)

            errCyc_sum = errG_low_sum = errG_high_sum = errD_low_sum = errD_high_sum = 0

        total_time = time.time() - start_time
        logging.info('Total training time: {}.'.format(datetime.timedelta(seconds=total_time)))

        total_time = time.time() - start_time
        logging.info('Total training time: {}.'.format(datetime.timedelta(seconds=total_time)))

        self.save_network_models(generator_LH, generator_HL, discriminator_LH, discriminator_HL, save_weights_only=True)

    def format_data_to_network(self, xs, ys, network, intensity_checking=False):
        """
        Adjust the xs/ys data to a format that is going to be understood by the network
        By default, modify the original data for efficiency (inplace=True)
        :param xs: numpy array (or list of numpy array) that contains the xs data
        :param ys: numpy array (or list of numpy array) that contains the labels for an image in the original format.
                               If None, the labels are ignored and only an image is returned
        :param network:
        :param intensity_checking:
        :return: array. Array adjuested to the format required by the network
        """
        assert network.expected_input_values_range is not None, \
            "Make sure you set a value for 'expected_input_values_range' property in your network"

        min_output, max_output = network.expected_input_values_range
        final_shape = self.parameters_dict['image_shape']

        if len(final_shape) == 2:
            cc = xs[0].shape[2] // 2
            img = xs[0][:, :, :, cc].astype(np.float32)
            reformatted_img = np.zeros((img.shape[0], final_shape[0], final_shape[1]), dtype=np.float32)
        else:
            cc = xs[0].shape[3] // 2
            img = xs[0][:, :, :, cc-final_shape[2]//2:cc+final_shape[2]//2+1].astype(np.float32)
            reformatted_img = np.zeros((img.shape[0], final_shape[0], final_shape[1], final_shape[2]), dtype=np.float32)

        if img.shape[1] != final_shape[0] or img.shape[2] != final_shape[1]:
            for ii in range(img.shape[0]):
                reformatted_img[ii] = self.resize_image(img[ii], final_shape)
                reformatted_img[ii] = DataProcessing.normalize_CT_image_intensity(reformatted_img[ii],
                                                                                  min_value=-1024,
                                                                                  max_value=1000,
                                                                                  min_output=min_output,
                                                                                  max_output=max_output)
        else:
            for ii in range(img.shape[0]):
                reformatted_img[ii] = DataProcessing.normalize_CT_image_intensity(img[ii],
                                                                                  min_value=-1024,
                                                                                  max_value=1000,
                                                                                  min_output=min_output,
                                                                                  max_output=max_output)

        reformatted_img = np.expand_dims(reformatted_img, -1)

        if intensity_checking:
            assert "expected_inputs_range" in self.parameters_dict, "I do not know what range to expect!"
            # Validate inputs
            TOLERANCE = 0.2
            for ip in reformatted_img:
                min_ = ip.min()
                max_ = ip.max()
                if abs(min_ - self.parameters_dict["expected_inputs_range"][0]) / min_ > TOLERANCE \
                    or abs(max_ - self.parameters_dict["expected_inputs_range"][1]) / max_ > TOLERANCE:
                    raise AssertionError("The expected value range for the inputs is {}-{}. Found: {}-{}".format(
                        self.parameters_dict["expected_inputs_range"][0],
                        self.parameters_dict["expected_inputs_range"][1],
                        min_, max_
                    ))

        return reformatted_img, ys

    def format_data_to_network_h2l(self, xs, ys, network, intensity_checking=False):
        """
        Adjust the xs/ys data to a format that is going to be understood by the network
        By default, modify the original data for efficiency (inplace=True)
        :param xs: numpy array (or list of numpy array) that contains the xs data
        :param ys: numpy array (or list of numpy array) that contains the labels for an image in the original format.
                               If None, the labels are ignored and only an image is returned
        :param network:
        :param intensity_checking:
        :return: array. Array adjuested to the format required by the network
        """
        assert network.expected_input_values_range is not None, \
            "Make sure you set a value for 'expected_input_values_range' property in your network"

        min_output, max_output = network.expected_input_values_range
        image_shape = self.parameters_dict['image_shape']
        final_shape = [image_shape[0], image_shape[1], 1]

        img = xs[0][:, :, :, image_shape[2] // 2].astype(np.float32)
        img = np.expand_dims(img, axis=-1)
        reformatted_img = np.zeros((img.shape[0], final_shape[0], final_shape[1], 1), dtype=np.float32)

        if img.shape[1] != final_shape[0] or img.shape[2] != final_shape[1]:
            for ii in range(img.shape[0]):
                reformatted_img[ii] = self.resize_image(img[ii], final_shape)
                reformatted_img[ii] = DataProcessing.normalize_CT_image_intensity(reformatted_img[ii],
                                                                                  min_value=-1024,
                                                                                  max_value=1000,
                                                                                  min_output=min_output,
                                                                                  max_output=max_output)
        else:
            for ii in range(img.shape[0]):
                reformatted_img[ii] = DataProcessing.normalize_CT_image_intensity(img[ii],
                                                                                  min_value=-1024,
                                                                                  max_value=1000,
                                                                                  min_output=min_output,
                                                                                  max_output=max_output)

        reformatted_img = np.expand_dims(reformatted_img, -1)

        if intensity_checking:
            assert "expected_inputs_range" in self.parameters_dict, "I do not know what range to expect!"
            # Validate inputs
            TOLERANCE = 0.2
            for ip in reformatted_img:
                min_ = ip.min()
                max_ = ip.max()
                if abs(min_ - self.parameters_dict["expected_inputs_range"][0]) / min_ > TOLERANCE \
                    or abs(max_ - self.parameters_dict["expected_inputs_range"][1]) / max_ > TOLERANCE:
                    raise AssertionError("The expected value range for the inputs is {}-{}. Found: {}-{}".format(
                        self.parameters_dict["expected_inputs_range"][0],
                        self.parameters_dict["expected_inputs_range"][1],
                        min_, max_
                    ))

        return reformatted_img, ys

    def format_ct_image_to_network(self, input_image, network, model_type='3Dto2D'):
        """
        Adjust the input image to a format that is going to be understood by the network
        By default, modify the original data for efficiency (inplace=True)
        :param input_image: numpy array (or list of numpy array) that contains the xs data
        :param network: network object
        :param model_type: type of the model to be used
        :return: array. Image array adjusted to the format required by the network
        """
        assert network.expected_input_values_range is not None, \
            "Make sure you set a value for 'expected_input_values_range' property in your network"

        min_output, max_output = network.expected_input_values_range

        final_shape = network.img_shape[:-1]
        reformatted_img = np.zeros(((input_image.shape[2],) + final_shape), dtype=np.float32)
        cc = final_shape[2] // 2

        for ii in range(2, input_image.shape[2]-2):
            if model_type == '3Dto2D':
                img = input_image[:, :, ii - cc:ii + cc + 1].astype(np.float32)
            else:
                img = input_image[:, :, ii].astype(np.float32)

            if img.shape[0] != final_shape[0] or img.shape[1] != final_shape[1]:
                img = self.resize_image(img, final_shape)

            reformatted_img[ii] = DataProcessing.normalize_CT_image_intensity(img,
                                                                              min_value=-1024,
                                                                              max_value=1000,
                                                                              min_output=min_output,
                                                                              max_output=max_output)
        reformatted_img = np.expand_dims(reformatted_img, -1)

        return reformatted_img

    def keras_generator_h2l(self, ds_manager, network, batch_size, batch_type):
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
            xs, ys = self.format_data_to_network_h2l(xs, ys, network)
            yield xs, ys

    @staticmethod
    def resize_image(image, out_shape, output_type=sitk.sitkInt16):
        image_io = ImageReaderWriter()
        if len(out_shape) == 3:
            image_sitk = image_io.numpy_to_sitkImage(image)
            resampled_sitk, _ = DataProcessing.resample_image_itk(image_sitk, np.asarray(out_shape), output_type,
                                                                  interpolator=sitk.sitkBSpline)
        else:
            image_sitk = image_io.numpy_to_sitkImage_2d(image)
            resampled_sitk, _ = DataProcessing.resample_image_itk(image_sitk, np.asarray(out_shape), output_type,
                                                                  interpolator=sitk.sitkLinear)

        if len(out_shape) == 3:
            return image_io.sitkImage_to_numpy(resampled_sitk)
        else:
            return image_io.sitkImage_to_numpy_2d(resampled_sitk)

    @staticmethod
    def de_normalize_CT_image_intensity(image_array, min_value=-1024, max_value=1000):
        """
        Threshold and adjust contrast range in a CT image.
        :param image_array: int numpy array (CT or partial CT image)
        :param min_value: int. Min threshold (everything below that value will be thresholded). If None, ignore
        :param max_value: int. Max threshold (everything below that value will be thresholded). If None, ignore
        :param out: numpy array. Array that will be used as an output
        :return: New numpy array unless 'out' parameter is used. If so, reference to that array
        """
        image = image_array.astype(np.float32)
        np.clip(image, 0., 1., image)

        image *= (max_value - min_value)
        image += min_value

        return image

    def test(self):
        """
        Evaluate a dataset
        """
        # Build network models

        generator_LH_path = os.path.join(self.model_folder, self.model_name_GLH + '.h5')
        generator_HL_path = os.path.join(self.model_folder, self.model_name_GHL + '.h5')

        params = self.read_parameters_from_model_h5(generator_LH_path)

        p = dict()
        p['img_shape'] = params['image_shape']
        p['gf'] = params['generator_filters']
        p['df'] = params['discriminator_filters']
        p['lambda_cycle'] = params['lambda_cycle']

        network = LowToHighDoseNetwork(**p)

        img_shape = params['image_shape']

        if len(img_shape) == 2:
            generator_LH = network.build_generator_unet(pretrained_weights_file_path=generator_LH_path)
            generator_HL = network.build_generator_unet(pretrained_weights_file_path=generator_HL_path)
        else:
            generator_LH = network.build_generator_3d(pretrained_weights_file_path=generator_LH_path, generator_type='l2h')
            generator_HL = network.build_generator_3d(pretrained_weights_file_path=generator_HL_path, generator_type='h2l')

        low_dataset_manager = H5Manager(h5_file_path=self.parameters_dict['low_dataset_path'],
                                        batch_size=int(self.parameters_dict['batch_size']),
                                        train_ixs=None, validation_ixs=None, test_ixs=None,
                                        xs_dataset_names=('low_dose_CT',),
                                        ys_dataset_names=(), use_pregenerated_augmented_train_data=False)
        low_dataset_manager.generate_ixs_from_dataset('train_status')

        low_test_data = low_dataset_manager.get_all_test_data()[0]
        low_test_data_cid = h5py.File(self.parameters_dict['low_dataset_path'])['low_dose_CT.key'][:, 1]
        low_test_data_slice_nb = h5py.File(self.parameters_dict['low_dataset_path'])['low_dose_slice_number'][:, 0]

        imgs_L, _ = self.format_data_to_network(low_test_data, None, network, intensity_checking=False)
        fake_H = generator_LH.predict(imgs_L)
        reconstr_L = generator_HL.predict(fake_H)

        for ii in range(imgs_L.shape[0]):
            if fake_H[ii].shape[0] == 256:
                fake_H_original_shape = self.resize_image(fake_H[ii].squeeze(), img_shape, output_type=sitk.sitkFloat32)
                reconstr_L_original_shape = self.resize_image(reconstr_L[ii].squeeze(), img_shape,
                                                              output_type=sitk.sitkFloat32)
            else:
                fake_H_original_shape = fake_H[ii].squeeze()
                reconstr_L_original_shape = reconstr_L[ii].squeeze()

            fake_H_original_shape = self.de_normalize_CT_image_intensity(fake_H_original_shape,
                                                                         min_value=-1024,
                                                                         max_value=1000)
            reconstr_L_original_shape = self.de_normalize_CT_image_intensity(reconstr_L_original_shape,
                                                                             min_value=-1024,
                                                                             max_value=1000)

            if len(img_shape) == 2:
                gen_imgs_L2H = np.concatenate([low_test_data[0][ii, :, :, 2].astype(np.short),
                                               fake_H_original_shape.astype(np.short),
                                               reconstr_L_original_shape.astype(np.short)], axis=0)
            else:
                gen_imgs_L2H = np.concatenate(
                    [low_test_data[0][ii, :, :, 2].astype(np.short), fake_H_original_shape.astype(np.short),
                     reconstr_L_original_shape.astype(np.short)], axis=0)

            low_output_img_path = os.path.join(self.l2h_results_folder,
                                               '{}_{}.nrrd'.format(low_test_data_cid[ii],
                                                                   str(low_test_data_slice_nb[ii])))
            nrrd.write(low_output_img_path, gen_imgs_L2H)

        high_dataset_manager = H5Manager(h5_file_path=self.parameters_dict['high_dataset_path'],
                                         batch_size=int(self.parameters_dict['batch_size']),
                                         train_ixs=None, validation_ixs=None, test_ixs=None,
                                         xs_dataset_names=('high_dose_CT',),
                                         ys_dataset_names=(), use_pregenerated_augmented_train_data=False)
        high_dataset_manager.generate_ixs_from_dataset('train_status')

        high_test_data = high_dataset_manager.get_all_test_data()[0]
        high_test_data_cid = h5py.File(self.parameters_dict['high_dataset_path'])['high_dose_CT.key'][:, 1]
        high_test_data_slice_nb = h5py.File(self.parameters_dict['high_dataset_path'])['high_dose_slice_number'][:, 0]

        imgs_H, _ = self.format_data_to_network_h2l(high_test_data, None, network, intensity_checking=False)

        fake_L = generator_HL.predict(imgs_H)
        reconstr_H = generator_LH.predict(fake_L)

        for ii in range(imgs_H.shape[0]):
            if fake_L[ii].shape[0] == 256:
                fake_L_original_shape = self.resize_image(fake_L[ii].squeeze(), [img_shape[0], img_shape[1]],
                                                          output_type=sitk.sitkFloat32)
                reconstr_H_original_shape = self.resize_image(reconstr_H[ii].squeeze(), [img_shape[0], img_shape[1]],
                                                              output_type=sitk.sitkFloat32)
            else:
                fake_L_original_shape = fake_L[ii].squeeze()
                reconstr_H_original_shape = reconstr_H[ii].squeeze()

            fake_L_original_shape = self.de_normalize_CT_image_intensity(fake_L_original_shape,
                                                                         min_value=-1024,
                                                                         max_value=1000)

            reconstr_H_original_shape = self.de_normalize_CT_image_intensity(reconstr_H_original_shape,
                                                                             min_value=-1024,
                                                                             max_value=1000)

            gen_imgs_H2L = np.concatenate([high_test_data[0][ii, :, :, 2].astype(np.short),
                                           fake_L_original_shape.astype(np.short),
                                           reconstr_H_original_shape.astype(np.short)], axis=0)

            high_output_img_path = os.path.join(self.h2l_results_folder,
                                                '{}_{}.nrrd'.format(high_test_data_cid[ii],
                                                                    str(high_test_data_slice_nb[ii])))
            nrrd.write(high_output_img_path, gen_imgs_H2L)

    def predict_low_to_high_dose_from_ct(self, in_ct, cnn_model_path):
        params = self.read_parameters_from_model_h5(cnn_model_path)
        img_shape = params['image_shape']

        if len(img_shape) == 2:
            model_type = '2D'
        else:
            model_type = '3Dto2D'

        p = dict()
        p['img_shape'] = params['image_shape']
        p['gf'] = params['generator_filters']
        p['df'] = params['discriminator_filters']
        p['lambda_cycle'] = params['lambda_cycle']

        network = LowToHighDoseNetwork(**p)

        if model_type == '2D':
            L2H_generator = network.build_generator_unet(pretrained_weights_file_path=cnn_model_path)
        else:
            L2H_generator = network.build_generator_3d(pretrained_weights_file_path=cnn_model_path,
                                                       generator_type='l2h')

        imgs_L = self.format_ct_image_to_network(in_ct, network, model_type=model_type)

        imgs_H = L2H_generator.predict(imgs_L, batch_size=1)

        filtered_image = np.zeros(in_ct.shape, dtype=np.short)

        if model_type == '2D':
            for ii in range(imgs_H.shape[0]):
                if img_shape[0] != in_ct.shape[0] or img_shape[1] != in_ct.shape[1]:
                    img_H_original_shape = self.resize_image(imgs_H[ii].squeeze(), [in_ct.shape[0], in_ct.shape[1]],
                                                             output_type=sitk.sitkFloat32)
                else:
                    img_H_original_shape = imgs_H[ii].squeeze()

                img_H_original_shape = self.de_normalize_CT_image_intensity(img_H_original_shape,
                                                                            min_value=-1024,
                                                                            max_value=1000)
                filtered_image[:, :, ii] = img_H_original_shape.astype(np.short)
        else:
            for ii in range(2, imgs_H.shape[0] - 2):
                if img_shape[0] != in_ct.shape[0] or img_shape[1] != in_ct.shape[1]:
                    img_H_original_shape = self.resize_image(imgs_H[ii].squeeze(), [in_ct.shape[0], in_ct.shape[1]],
                                                             output_type=sitk.sitkFloat32)
                else:
                    img_H_original_shape = imgs_H[ii].squeeze()

                img_H_original_shape = self.de_normalize_CT_image_intensity(img_H_original_shape,
                                                                            min_value=-1024,
                                                                            max_value=1000)
                filtered_image[:, :, ii+1] = img_H_original_shape.astype(np.short)

            filtered_image[:, :, 0:3] = in_ct[:, :, 0:3]
            filtered_image[:, :, -1] = in_ct[:, :, -1]

        return filtered_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Airway cartilage sizing via deep learning')
    parser.add_argument('--operation', help="TRAIN / TEST, etc.", type=str, required=True,
                        choices=['TRAIN', 'TEST'])
    parser.add_argument('--output_folder', type=str, help="Program output logging folder (additional)", required=True)
    parser.add_argument('--params_file', type=str, help="Parameters file. Required for both TRAIN/TEST operations.",
                        required=True)
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

    parameters_dict = Utils.read_parameters_dict(args.params_file)
    e = LowDoseToHighDoseEngine(parameters_dict, output_folder=output_folder)

    if args.operation == 'TRAIN':
        e.train()
    elif args.operation == 'TEST':
        e.test()
