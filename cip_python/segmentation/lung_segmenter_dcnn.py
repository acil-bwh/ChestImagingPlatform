import argparse

from cip_python.common import DeepLearningModelsManager

import os
import numpy as np
import SimpleITK as sitk
from scipy import signal

from cip_python.dcnn import metrics, utils

from keras.models import load_model
import keras.backend as K

class LungSegmenterDCNN:
    def __init__(self):
        self.input_ct = None
        self.output_lm = None

    @staticmethod
    def load_network_model(model_path):
        """
        Load a pretrained Keras model from a file that will contain both configuration and weights
        :param model_path:
        :return:
        """
        if not os.path.exists(model_path):
            raise Exception("{} not found".format(model_path))

        network_model = load_model(model_path, custom_objects={'dice_coef': metrics.dice_coef,
                                                               'dice_coef_loss': metrics.dice_coef_loss})
        return network_model

    def lung_segmentation(self, network_model, patch_size, image_np):
        if patch_size[0] != patch_size[1]:  # Coronal orientation
            image_np = image_np.transpose([0, 2, 1])

        if patch_size[0] != image_np.shape[0] or patch_size[1] != image_np.shape[1]:
            cnn_images = np.zeros((patch_size[0], patch_size[1], image_np.shape[2]), dtype=np.float32)
            for ii in range(image_np.shape[2]):
                slice_sitk = sitk.GetImageFromArray(image_np[:, :, ii].transpose())
                resampled_sitk = utils.resample_image(slice_sitk, patch_size, sitk.sitkFloat32,
                                                      interpolator=sitk.sitkBSpline)
                cnn_images[:, :, ii] = sitk.GetArrayFromImage(resampled_sitk).transpose()
        else:
            cnn_images = image_np

        for ii in range(cnn_images.shape[2]):
            cnn_images[:, :, ii] = utils.standardization(cnn_images[:, :, ii])
            # cnn_images[:, :, ii] = utils.range_normalization(cnn_images[:, :, ii])

        cnn_images = cnn_images.transpose([2, 0, 1])
        cnn_images = np.expand_dims(cnn_images, axis=3)

        predictions = network_model.predict(cnn_images, batch_size=10, verbose=1)
        predictions = np.asarray(predictions)

        if patch_size[0] != image_np.shape[0] or patch_size[1] != image_np.shape[1]:
            final_predictions = np.zeros((predictions.shape[0], predictions.shape[1], image_np.shape[0],
                                          image_np.shape[1]))
            for ii in range(predictions.shape[0]):
                for jj in range(predictions.shape[1]):
                    label_sitk = sitk.GetImageFromArray(predictions[ii, jj, :, :, 0].transpose())
                    output_shape = np.asarray([image_np.shape[0], image_np.shape[1]])
                    resampled_label_sitk = utils.resample_image(label_sitk, output_shape, sitk.sitkFloat32,
                                                                interpolator=sitk.sitkBSpline)
                    final_predictions[ii, jj, :, :] = sitk.GetArrayFromImage(resampled_label_sitk).transpose()
        else:
            final_predictions = predictions.squeeze()

        return final_predictions

    @staticmethod
    def sum_up_to_one(prob):
        # Avoid division by 0
        prob += 0.000001
        xx = 1.0 / np.sum(prob, axis=0)
        return prob * xx

    @staticmethod
    def compute_map(prob):  # Maximum a posteriori probability
        map_id = np.argmax(prob, axis=0)

        return map_id.astype(np.int32)

    def combine_planes(self, axial_prob, coronal_prob):
        ss_a = int((int(axial_prob.shape[3] / 2.0) - 10) / 3.0)
        la_z = signal.gaussian(axial_prob.shape[3], std=ss_a)

        ss_c = int((int(coronal_prob.shape[2] / 2.0) - 10) / 3.0)
        lc_y = signal.gaussian(coronal_prob.shape[2], std=ss_c)

        axial_prob_gauss = np.zeros(axial_prob.shape, dtype=np.float32)
        for ii in range(axial_prob_gauss.shape[0]):
            for aa in range(axial_prob_gauss.shape[2]):
                for bb in range(axial_prob_gauss.shape[1]):
                    axial_prob_gauss[ii, bb, aa, :] = axial_prob[ii, bb, aa, :] * la_z

        coronal_prob_gauss = np.zeros(coronal_prob.shape, dtype=np.float32)
        for ii in range(coronal_prob_gauss.shape[0]):
            for aa in range(coronal_prob_gauss.shape[3]):
                for bb in range(coronal_prob_gauss.shape[1]):
                    coronal_prob_gauss[ii, bb, :, aa] = coronal_prob[ii, bb, :, aa] * lc_y

        combined_pp = np.zeros(axial_prob.shape, dtype=np.float32)
        for nn in range(combined_pp.shape[0]):
            if nn != 3:
                combined_pp[nn, :, :, :] = axial_prob_gauss[nn, :, :, :] + coronal_prob_gauss[nn, :, :, :]
            else:
                cc = np.zeros(combined_pp[nn].shape, dtype=np.float32)
                cc[coronal_prob[nn] >= 0.5] = 5.0
                combined_pp[nn] = cc

        return combined_pp

    def execute(self, input_ct, output_lm,
                axial_model_path, coronal_model_path, segmentation_type='combined'):
        self.input_ct = input_ct
        self.output_lm = output_lm

        image_sitk = sitk.ReadImage(self.input_ct)
        image_np = sitk.GetArrayFromImage(image_sitk).transpose([2, 1, 0])

        if segmentation_type == 'combined' or segmentation_type == 'axial':
            # Axial Lung Segmentation
            print ('    Predicting axial probabilities...')
            axial_model = self.load_network_model(axial_model_path)

            a_batch_input_shape = axial_model.layers[0].get_config()['batch_input_shape']

            if K.image_data_format() == 'channels_last':
                a_patch_size = np.asarray([a_batch_input_shape[1], a_batch_input_shape[2]])
            else:
                a_patch_size = np.asarray([a_batch_input_shape[2], a_batch_input_shape[3]])

            axial_predictions = self.lung_segmentation(axial_model, a_patch_size, image_np)
            axial_predictions = axial_predictions.transpose([0, 2, 3, 1])
            axial_predictions = self.sum_up_to_one(axial_predictions)

        if segmentation_type == 'combined' or segmentation_type == 'coronal':
            # Coronal Lung Segmentation
            print ('    Predicting coronal probabilities...')
            coronal_model = self.load_network_model(coronal_model_path)

            c_batch_input_shape = coronal_model.layers[0].get_config()['batch_input_shape']

            if K.image_data_format() == 'channels_last':
                c_patch_size = np.asarray([c_batch_input_shape[1], c_batch_input_shape[2]])
            else:
                c_patch_size = np.asarray([c_batch_input_shape[2], c_batch_input_shape[3]])

            coronal_predictions = self.lung_segmentation(coronal_model, c_patch_size, image_np)
            coronal_predictions = coronal_predictions.transpose([0, 2, 1, 3])
            coronal_predictions = self.sum_up_to_one(coronal_predictions)

        if segmentation_type == 'combined':
            print ('    Combining axial and coronal probabilities...')
            combined_predictions = self.combine_planes(axial_predictions, coronal_predictions)
            output_labels = self.compute_map(combined_predictions)
        elif segmentation_type == 'axial':
            output_labels = self.compute_map(axial_predictions)
        else:
            output_labels = self.compute_map(coronal_predictions)

        output_labels[output_labels == 3] = 512
        output_labels[output_labels == 1] = 3

        print ('    Writing lung segmentation file...')
        output_labels_sitk = sitk.GetImageFromArray(output_labels.transpose([2, 1, 0]))
        output_labels_sitk.CopyInformation(image_sitk)

        sitk.WriteImage(output_labels_sitk, self.output_lm, True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN method (2D) for lung segmentation (unet)')
    parser.add_argument('--i', dest='in_ct', help="Input CT file (.nrrd)", type=str, required=True)
    parser.add_argument('--t', dest='segmentation_type', choices=['axial','coronal','combined'],
                        help='Options: axial, coronal, combined', type=str, default='combined')
    parser.add_argument('--am', dest='axial_model', help='CNN model for axial lung segmentation (.hdf5).', required=False,
                        type=str, default=None)
    parser.add_argument('--cm', dest='coronal_model', help='CNN model for coronal lung segmentation (.hdf5).', required=False,
                        type=str, default=None)

    parser.add_argument('--o', dest='out_lm', help='Output labelmap file (.nrrd).', required=True, type=str,
                        default=None)
    op = parser.parse_args()

    axial = op.segmentation_type in ('axial', 'combined')
    coronal = op.segmentation_type in ('coronal', 'combined')

    if axial:
        if op.axial_model:
            axial_model = op.axial_model
        else:
            # Load with model manager
            manager = DeepLearningModelsManager()
            axial_model = manager.get_model_path('LUNG_SEGMENTATION_AXIAL')
    else:
        axial_model = None

    if coronal:
        if op.coronal_model:
            coronal_model = op.coronal_model
        else:
            # Load with model manager
            manager = DeepLearningModelsManager()
            coronal_model = manager.get_model_path('LUNG_SEGMENTATION_CORONAL')
    else:
        coronal_model = None

    lung_segmenter = LungSegmenterDCNN()
    lung_segmenter.execute(op.in_ct, op.out_lm, axial_model, coronal_model, segmentation_type=op.segmentation_type)

