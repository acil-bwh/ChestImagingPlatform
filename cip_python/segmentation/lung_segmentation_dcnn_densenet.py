import os
import argparse
import numpy as np
import time

from skimage.transform import resize

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dense, add

from cip_python.common import ChestConventions, ChestRegion, ChestType, DeepLearningModelsManager
from cip_python.input_output import ImageReaderWriter




class LungSegmentationDCNNDensenet(object):
    #def __init__(self, axial_model_path, coronal_model_path, dims=(512, 512, 512)):
    def __init__(self, axial_model_path, dims=(512, 512, 512)):
        """
        Constructor

        Parameters
        ----------
        model_path: path to the Keras model11

        """
        self._axial_model_weights_path_ = axial_model_path
        # self._coronal_model_weights_path_ = coronal_model_path
        self.dims = dims

    def run(self, input_path, output_path):
        """
        Run a full segmentation.
        Generate a labelmap

        Parameters
        ----------
        input_path: local path to a nrrd volume
        output_path: path where the output result (nrrd volume labelmap) will be stored

        """
        print("Reading input data...")
        #input_axial, input_coronal, metainfo = self._read_input_(input_path)
        input_axial, metainfo = self._read_input_(input_path)

        # print ("AXIAL...")
        print ("... Building model...")
        model = self.build_model_axial()
        print("... Predicting results...")
        lm = np.zeros_like(input_axial[:, :, :, 0], dtype=np.uint8)
        num_slices = input_axial.shape[0]
        for i in range(num_slices):
            t1 = time.time()
            output = model.predict(input_axial[i:i+1, :, :, :])
            t2 = time.time()
            lm[i] = np.argmax(output, axis=3)[0]
            if i % 5 == 0:
                print("{}/{}: {}s".format(i, num_slices, t2-t1))

        # Update the needed CIP codes
        # lm == 2: No need to update (ChestRegion.RIGHTLUNG)
        lm[lm == 3] = ChestConventions.GetValueFromChestRegionAndType(ChestRegion.TRACHEA2, ChestType.UNDEFINEDTYPE) # 512
        lm[lm == 1] = ChestRegion.LEFTLUNG  # 3

        # print ("CORONAL...")
        # print ("... Building model...")
        # K.clear_session()
        # model = self.build_model_coronal()
        # print("... Predicting results...")
        # output_coronal = model.predict(input_coronal, batch_size=1)
        #
        # print("Composing labelmap...")
        # output_labelmap = self.__compose_labelmap__(output_axial, output_coronal)

        print("Writing output...")
        self._write_output_(lm, metainfo, output_path)

        print("Done!")

    def build_model_axial(self):
        """
        Build a model with an Axial input shape
        Returns
        -------
        Keras model
        """
        inputs = Input((self.dims[0], self.dims[1], 1))
        model = self.build_model(inputs)
        model.load_weights(self._axial_model_weights_path_)
        return model

    def build_model_coronal(self):
        """
        Build a model with a Coronal input shape
        Returns
        -------
        Keras Model

        """
        raise NotImplementedError("Update needed")
        inputs = Input((self.dims[2], self.dims[0], 1))
        model = self.build_model(inputs)
        model.load_weights(self._coronal_model_weights_path_)
        return model

    def build_model(self, inputs):
        """
        Build the model given the inputs in a fixed dimension. It does NOT load the weights
        because there are
        Parameters
        ----------
        inputs: keras Input in a fixed size

        Returns
        -------
        Keras model with the weights loaded

        """
        n_labels = 3
        conv1 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv1)
        conv1 = add([inputs, conv1])
        down1 = Conv2D(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv1)

        conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(down1)
        conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(conv2)
        conv2 = add([down1, conv2])
        down2 = Conv2D(128, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv2)

        conv3 = Conv2D(128, (4, 4), activation='relu', padding='same')(down2)
        conv3 = Conv2D(128, (4, 4), activation='relu', padding='same')(conv3)
        conv3 = add([down2, conv3])
        down3 = Conv2D(256, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv3)

        conv4 = Conv2D(256, (4, 4), activation='relu', padding='same')(down3)
        conv4 = Conv2D(256, (4, 4), activation='relu', padding='same')(conv4)
        conv4 = add([down3, conv4])
        down4 = Conv2D(512, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(down4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = add([down4, conv5])
        down5 = Conv2D(1024, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)

        conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(down5)
        conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = add([down5, conv6])

        up7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6)
        conc7 = concatenate([up7, conv5], axis=3)
        conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conc7)
        conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = add([up7, conv7])

        up8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7)
        conc8 = concatenate([up8, conv4], axis=3)
        conv8 = Conv2D(256, (4, 4), activation='relu', padding='same')(conc8)
        conv8 = Conv2D(256, (4, 4), activation='relu', padding='same')(conv8)
        conv8 = add([up8, conv8])

        up9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8)
        conc9 = concatenate([up9, conv3], axis=3)
        conv9 = Conv2D(128, (4, 4), activation='relu', padding='same')(conc9)
        conv9 = Conv2D(128, (4, 4), activation='relu', padding='same')(conv9)
        conv9 = add([up9, conv9])

        up10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9)
        conc10 = concatenate([up10, conv2], axis=3)
        conv10 = Conv2D(64, (5, 5), activation='relu', padding='same')(conc10)
        conv10 = Conv2D(64, (5, 5), activation='relu', padding='same')(conv10)
        conv10 = add([up10, conv10])

        up11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10)
        conc11 = concatenate([up11, conv1], axis=3)
        conv11 = Conv2D(32, (5, 5), activation='relu', padding='same')(conc11)
        conv11 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv11)
        conv11 = add([up11, conv11])

        conv12 = Conv2D(n_labels + 1, (1, 1), activation='sigmoid')(conv11)

        model = Model(inputs=[inputs], outputs=[conv12])
        return model

    def _read_input_(self, input_volume_path):
        """
        Read a nrrd volume path and convert the data to the format that is expected by the network

        Parameters
        ----------
        input_volume_path

        Returns
        -------
        Tuple of two numpy arrays in the format that is expected by the network as an inputs (axial and coronal)
        plus original volume metainfo
        """
        reader = ImageReaderWriter()
        img, metadata = reader.read_in_numpy(input_volume_path)
        img = img.astype(np.float)

        mean = np.mean(img)  # mean for data centering
        std = np.std(img)  # std for data normalization

        img -= mean
        img /= std

        input_axial = img.transpose((2, 0, 1))
        input_axial = input_axial.reshape((input_axial.shape[0],) + (512, 512, 1))
        return input_axial, metadata

        #input_coronal = img.transpose((1, 2, 0))
        #input_coronal = resize(input_coronal, (512, 512, 512)).reshape((512, 512, 512, 1))

        #return input_axial, input_coronal, metadata

    def _write_output_(self, output_array, output_path, metainfo):
        """
        Convert the output of the network to a labelmap nrrd file.
        Write the output in "output_path"

        Parameters
        ----------
        output_array: numpy array returned by the network
        output_path: output path to the labelmap

        """
        writer = ImageReaderWriter()
        writer.write_from_numpy(output_array, metainfo, output_path)

    #### Additional functions
    def __compose_labelmap__(self, axial, coronal):
        """
        Go from a 4-channel x 2 models output to a single labelmap.
        Chosen operation: sum of values
        Parameters
        ----------
        axial: numpy array X,512,512,4
        coronal: numpy array X,512,512,4

        Returns
        -------

        """
        # 0 Background
        # 1 3
        # 2 2
        # 3 512
        lm = np.argmax(axial + coronal, axis=3).astype(np.uint32)
        lm[lm == 1] = 3
        lm[lm == 3] = 512
        return lm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lung segmentation based on a DCNN Densenet network')
    parser.add_argument(dest='input_path', type=str, help="Input volume path")
    parser.add_argument('-m', dest='model_path', type=str, help="Path to a pretrained keras model", required=False)
    parser.add_argument('-o', dest='output_file', type=str, help="Path to the output file. Default: current folder and case_dcnnLungSegmentationLabelmap.nrrd", 
                        required=False)

    arguments = parser.parse_args()
    input = arguments.input_path
    
    if arguments.model_path:
        model_path = arguments.model_path
    else:
        # Use the Models Manager to download the predefined model
        manager = DeepLearningModelsManager()
        model_path=manager.get_model(manager.LUNG_SEGMENTATION_AXIAL)
    
    if arguments.output_file:
        output_file = arguments.output_file
    else:
        output_file = os.path.basename(input)
        i = output_file.rfind('.')
        if i != -1:
            output_file = output_file[:i]
        output_file += "_dcnnLungSegmentationLabelmap.nrrd"

    l = LungSegmentationDCNNDensenet(model_path)
    l.run(input, output_file)
        

