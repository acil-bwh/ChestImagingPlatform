import argparse

import os
import numpy as np
import SimpleITK as sitk
from scipy import signal

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter
from cip_python.dcnn.logic import DeepLearningModelsManager, MetricsManager, Utils
from cip_python.dcnn.data import DataProcessing


class LungSegmenterDCNN:
    def __init__(self):
        self.input_ct = None
        self.output_lm = None

        self.prob_eps = 1e-7
        c = ChestConventions()
        self.RightLabel = c.GetChestRegionValueFromName('RightLung')
        self.LeftLabel = c.GetChestRegionValueFromName('LeftLung')
        self.AirwayLabel = c.GetValueFromChestRegionAndType(0, 2)

    @staticmethod
    def load_network_model(model_path):
        """
        Load a pretrained Keras model from a file that will contain both configuration and weights
        :param model_path:
        :return:
        """
        if not os.path.exists(model_path):
            raise Exception("{} not found".format(model_path))

        network_model = load_model(model_path, custom_objects={'_multiclass_2D_tversky_loss_': MetricsManager.multiclass_2D_tversky_loss(smooth=1, alpha=0.75),
                                                               '_multiclass_2D_tversky_index_': MetricsManager.multiclass_2D_tversky_index(smooth=1, alpha=0.75)})
        return network_model

    def normalized_convolution_lp(self, signal, certainty, filter_func, *args):

        numerator = filter_func(signal * certainty, *args)
        denominator = filter_func(certainty)
        index = denominator > self.prob_eps

        output = numerator
        output[index] = numerator[index] / denominator[index]

        return output

    @staticmethod
    def filtra3D(im, window=(1,1,1), spacing=(1,1,1), method=2):
        im_sitk = sitk.GetImageFromArray(im)
        if method == 1:
            # Calculate the sigma parameters for each dimensionssf
            spacing_sitk = (spacing[2], spacing[1], spacing[0])
            im_sitk.SetSpacing(spacing_sitk)
            sigma_filter = np.mean(window) * 0.78 * min(spacing_sitk)  # Order of sitk spacing

            ff = sitk.DiscreteGaussianImageFilter()
            ff.SetUseImageSpacing = True
            ff.SetVariance(sigma_filter**2)

            out_im = ff.Execute(im_sitk)
            return sitk.GetArrayFromImage(out_im)
        else:
            ff = sitk.MeanImageFilter()
            out_im = ff.Execute(im_sitk,window)
            return sitk.GetArrayFromImage(out_im)

    def lung_segmentation(self, network_model, patch_size, image_sitk, N_subsampling=10, orientation='axial'):
        image_spacing = image_sitk.GetSpacing()
        image_np = sitk.GetArrayFromImage(image_sitk).transpose([2, 1, 0])
        image_np = image_np.astype(np.float32)

        if orientation == 'coronal':  # Transpose image
            image_np = image_np.transpose([0, 2, 1])
            image_spacing = np.asarray([image_spacing[0], image_spacing[2], image_spacing[1]])

        img_sitk = sitk.GetImageFromArray(image_np.transpose([2, 1, 0]))
        img_sitk.SetSpacing([image_spacing[0], image_spacing[1], image_spacing[2]])

        z_shape = image_np.shape[2]
        if patch_size[0] != image_np.shape[0] or patch_size[1] != image_np.shape[1]:
            if orientation == 'coronal':
                z_shape /= 2
            output_size = np.asarray([int(patch_size[0]), int(patch_size[1]), int(z_shape)])

            resampled_sitk, output_spacing = DataProcessing.resample_image_itk(img_sitk, output_size, sitk.sitkFloat32,
                                                                               interpolator=sitk.sitkLinear)
            cnn_img = sitk.GetArrayFromImage(resampled_sitk).transpose([2, 1, 0])
            cnn_img = cnn_img.astype(np.float32)
        else:
            cnn_img = sitk.GetArrayFromImage(img_sitk).transpose([2, 1, 0])
            output_spacing = img_sitk.GetSpacing()

        z_samples = range(0, cnn_img.shape[2], N_subsampling)
        if not cnn_img.shape[2] - 1 in z_samples:
            z_samples.append(cnn_img.shape[2] - 1)

        predictions = np.zeros((int(z_shape), int(patch_size[0]), int(patch_size[1]), 4), dtype=np.float32)
        certainty_map = np.zeros((int(z_shape), int(patch_size[0]), int(patch_size[1])), dtype=np.float32)

        for ii in range(cnn_img.shape[2]):
            cnn_img[:, :, ii] = DataProcessing.standardization(cnn_img[:, :, ii], mean_value=-700,
                                                               std_value=450)

        for ii in z_samples:
            print ('Segmenting slice {} of {}'.format(ii, cnn_img.shape[2]))

            pred_img = cnn_img[:, :, ii]

            pred_img = np.expand_dims(pred_img, axis=0)
            pred_img = np.expand_dims(pred_img, axis=-1)
            cnn_predictions = np.squeeze(network_model.predict(pred_img, batch_size=N_subsampling))
            predictions[ii] = cnn_predictions
            certainty_map[ii] = np.ones([patch_size[0], patch_size[1]])

        if N_subsampling > 1:
            for cc in range(4):
                predictions[cc] = self.normalized_convolution_lp(predictions[cc], certainty_map, self.filtra3D,
                                                                 [1, 1, N_subsampling], [1, 1, 1])
        return predictions, output_spacing

    @staticmethod
    def compute_map(prob):  # Maximum a posteriori probability
        map_id = np.argmax(prob, axis=-1)

        return map_id.astype(np.uint16)

    @staticmethod
    def combine_planes(axial_prob, coronal_prob):
        ss_a = int((int(axial_prob.shape[2] / 2.0) - 5) / 3.0)
        la_z = signal.gaussian(axial_prob.shape[2], std=ss_a)

        ss_c = int((int(coronal_prob.shape[3] / 2.0) - 5) / 3.0)
        lc_y = signal.gaussian(coronal_prob.shape[3], std=ss_c)

        axial_prob_gauss = np.zeros(axial_prob.shape, dtype=np.float32)
        for ii in range(axial_prob_gauss.shape[0]):
            for aa in range(axial_prob_gauss.shape[3]):
                for bb in range(axial_prob_gauss.shape[1]):
                    axial_prob_gauss[ii, bb, :, aa] = axial_prob[ii, bb, :, aa] * la_z

        coronal_prob_gauss = np.zeros(coronal_prob.shape, dtype=np.float32)
        for ii in range(coronal_prob_gauss.shape[0]):
            for aa in range(coronal_prob_gauss.shape[2]):
                for bb in range(coronal_prob_gauss.shape[1]):
                    coronal_prob_gauss[ii, bb, aa, :] = coronal_prob[ii, bb, aa, :] * lc_y

        combined_pp = np.zeros(axial_prob.shape, dtype=np.float32)
        for nn in range(combined_pp.shape[0]):
            if nn != 3:
                combined_pp[nn, :, :, :] = axial_prob_gauss[nn, :, :, :] + coronal_prob_gauss[nn, :, :, :]
            else:
                cc = np.zeros(combined_pp[nn].shape, dtype=np.float32)
                cc[coronal_prob[nn] >= 0.5] = 5.0
                combined_pp[nn] = cc

        return combined_pp

    @staticmethod
    def resample_predictions(predictions_image, predictions_spacing, output_size):
        output_size = np.asarray(output_size)
        pp_sitk = sitk.GetImageFromArray(predictions_image.transpose([2, 1, 0]))
        pp_sitk.SetSpacing(predictions_spacing)
        resampled_sitk, _ = DataProcessing.resample_image_itk(pp_sitk, output_size, sitk.sitkUInt16,
                                                              interpolator=sitk.sitkNearestNeighbor)
        resampled_np = sitk.GetArrayFromImage(resampled_sitk).transpose([2, 1, 0])
        return resampled_np

    def execute(self, input_ct, axial_model_path, coronal_model_path, segmentation_type='combined', N_subsampling=10):
        image_sitk = sitk.ReadImage(input_ct)
        img_size = image_sitk.GetSize()

        if segmentation_type == 'combined' or segmentation_type == 'axial':
            # Axial Lung Segmentation
            print ('    Predicting axial probabilities...')
            axial_model = self.load_network_model(axial_model_path)
            a_patch_size = np.asarray([256, 256])

            axial_predictions, axial_spacing = self.lung_segmentation(axial_model, a_patch_size, image_sitk,
                                                                      N_subsampling, orientation='axial')
            axial_predictions = axial_predictions.transpose([1, 2, 0, 3])

        if segmentation_type == 'combined' or segmentation_type == 'coronal':
            # Coronal Lung Segmentation
            print ('    Predicting coronal probabilities...')
            coronal_model = self.load_network_model(coronal_model_path)
            c_patch_size = np.asarray([256, 320])

            if N_subsampling > 1:
                N_subsampling_coronal = 2
            else:
                N_subsampling_coronal = 1

            coronal_predictions, coronal_spacing = self.lung_segmentation(coronal_model, c_patch_size, image_sitk,
                                                                          N_subsampling_coronal, orientation='coronal')
            coronal_predictions = coronal_predictions.transpose([1, 0, 2, 3])

        if segmentation_type == 'combined':
            print ('    Combining axial and coronal probabilities...')
            if axial_predictions.shape[2] != coronal_predictions.shape[2]:
                for kk in range(coronal_predictions.shape[3]):
                    tmp = np.zeros((axial_predictions.shape))
                    tmp[:, :, :, kk] = self.resample_predictions(coronal_predictions[:, :, :, kk], coronal_spacing,
                                                                 np.asarray(axial_predictions.shape[0:3]))
                    coronal_predictions = tmp.copy()

            combined_predictions = self.combine_planes(axial_predictions, coronal_predictions)
            output_labels = self.compute_map(combined_predictions)
            if output_labels.shape[0] != img_size[0] or output_labels[1] != img_size[1]:
                output_size = np.asarray(img_size)
                output_labels = self.resample_predictions(output_labels, coronal_spacing, output_size)
        elif segmentation_type == 'axial':
            output_labels = self.compute_map(axial_predictions)
            if a_patch_size[0] != img_size[0] or a_patch_size[1] != img_size[1]:
                output_size = np.asarray(img_size)
                output_labels = self.resample_predictions(output_labels, axial_spacing, output_size)
        else:
            output_labels = self.compute_map(coronal_predictions)
            if c_patch_size[0] != img_size[0] or c_patch_size[1] != img_size[1]:
                output_size = np.asarray(img_size)
                output_labels = self.resample_predictions(output_labels, coronal_spacing, output_size)

        output_labels = output_labels.astype(np.uint16)
        output_labels[output_labels == 3] = self.AirwayLabel
        output_labels[output_labels == 2] = self.LeftLabel
        output_labels[output_labels == 1] = self.RightLabel

        output_labels_sitk = sitk.GetImageFromArray(output_labels.transpose([2, 1, 0]))
        output_labels_sitk.SetSpacing(image_sitk.GetSpacing())
        output_labels_sitk.SetDirection(image_sitk.GetDirection())
        output_labels_sitk.SetOrigin(image_sitk.GetOrigin())

        return output_labels_sitk


class LungThirdSplitter():
    def __init__(self):
        self.size_th = 0.05
        self.coordinate_system = 'lps'
        c = ChestConventions()
        self.RightLabel = c.GetChestRegionValueFromName('RightLung')
        self.LeftLabel = c.GetChestRegionValueFromName('LeftLung')
        self.WholeLung = c.GetChestRegionValueFromName('WholeLung')
        self.UpperThird = c.GetChestRegionValueFromName('UpperThird')
        self.MiddleThrid = c.GetChestRegionValueFromName('MiddleThird')
        self.LowerThird = c.GetChestRegionValueFromName('LowerThird')
        self.LeftUpperThird = c.GetChestRegionValueFromName('LeftUpperThird')
        self.LeftMiddleThird = c.GetChestRegionValueFromName('LeftMiddleThird')
        self.LeftLowerThrid = c.GetChestRegionValueFromName('LeftLowerThird')
        self.RightUpperThird = c.GetChestRegionValueFromName('RightUpperThird')
        self.RightMiddleThrid = c.GetChestRegionValueFromName('RightMiddleThird')
        self.RightLowerThrid = c.GetChestRegionValueFromName('RightLowerThird')

        self.cc_f = sitk.ConnectedComponentImageFilter()
        self.r_f = sitk.RelabelComponentImageFilter()
        self.ls = sitk.LabelShapeStatisticsImageFilter()

    def execute(self, lm):
        # Get Region/Type Information
        lm_np = sitk.GetArrayFromImage(lm)
        lm_np = lm_np.astype(np.uint16)

        # Output holder copy
        olm_tmp = sitk.Image(lm)
        olm_np = sitk.GetArrayFromImage(olm_tmp)
        olm_np = olm_np.astype(np.uint16)

        present_labels = np.unique(lm_np)

        for ll in present_labels:
            lm_target_np = np.zeros(lm_np.shape, dtype=lm_np.dtype)
            lm_target_np[lm_np == ll] = 1

            tmp_itk = sitk.GetImageFromArray(lm_target_np)
            tmp_cc = self.cc_f.Execute(tmp_itk)
            tmp_rl = self.r_f.Execute(tmp_cc)

            tmp_rl_np = sitk.GetArrayFromImage(tmp_rl)
            olm_np[tmp_rl_np > 1] = 0

        lm_type_np = olm_np >> 8
        lm_region_np = olm_np & 255

        # Splitting in Thirds
        size = lm.GetSize()
        vol_right = np.sum(olm_np == self.RightLabel)
        vol_left = np.sum(olm_np == self.LeftLabel)
        target_vol_right = 0
        target_vol_left = 0
        for zz in range(size[2]):
            cut = olm_np[zz, :, :]
            right_mask = (cut == self.RightLabel)
            left_mask = (cut == self.LeftLabel)

            slice_vol_right = np.sum(right_mask)
            slice_vol_left = np.sum(left_mask)
            if target_vol_right <= vol_right / 3:
                cut[right_mask] = self.RightLowerThrid
            elif target_vol_right > vol_right / 3 and target_vol_right <= 2 * vol_right / 3:
                cut[right_mask] = self.RightMiddleThrid
            else:
                cut[right_mask] = self.RightUpperThird

            target_vol_right = target_vol_right + slice_vol_right

            if target_vol_left <= vol_left / 3:
                cut[left_mask] = self.LeftLowerThrid
            elif target_vol_left > vol_left / 3 and target_vol_left <= 2 * vol_left / 3:
                cut[left_mask] = self.LeftMiddleThird
            else:
                cut[left_mask] = self.LeftUpperThird

            target_vol_left = target_vol_left + slice_vol_left

        # Transfer type labels to output LM
        wl_mask = (lm_region_np == self.WholeLung) | (lm_region_np == self.RightLabel) | \
                  (lm_region_np == self.LeftLabel)

        olm_np[np.logical_not(wl_mask)] = lm_region_np[np.logical_not(wl_mask)]
        pp = olm_np + (lm_type_np << 8)

        olm = sitk.GetImageFromArray(pp)
        olm.CopyInformation(lm)
        return olm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN method (2D) for lung segmentation (modified unet)')
    parser.add_argument('--i', dest='in_ct', help="Input CT file (.nrrd)", type=str, required=True)
    parser.add_argument('--t', dest='segmentation_type', choices=['axial','coronal','combined'],
                        help='Options: [axial|coronal|combined]', type=str, default='combined')
    parser.add_argument('--am', dest='axial_model', help='Optional. CNN model for axial lung segmentation (.h5). '
                                                         'If not specified the default model will be used.',
                        required=False, type=str, default=None)
    parser.add_argument('--cm', dest='coronal_model', help='Optional. CNN model for coronal lung segmentation (.h5). '
                                                           'If not specified the default model will be used.',
                        required=False, type=str, default=None)
    parser.add_argument('--n', dest='n_subsampling', help='Number of slices to use for segmentation',
                        required=False, type=int, default=1)
    parser.add_argument('--o', dest='out_lm', help='Output labelmap file (.nrrd)', required=True, type=str,
                        default=None)
    parser.add_argument('-thirds', dest='thirds', help='Flag to split lung segmentation into thirds',
                        action='store_true')
    op = parser.parse_args()

    axial = op.segmentation_type in ('axial', 'combined')
    coronal = op.segmentation_type in ('coronal', 'combined')

    if axial:
        if op.axial_model:
            axial_model = op.axial_model
        else:
            # Load with model manager
            manager = DeepLearningModelsManager()
            axial_model = manager.get_model('LUNG_SEGMENTATION_AXIAL')
    else:
        axial_model = None

    if coronal:
        if op.coronal_model:
            coronal_model = op.coronal_model
        else:
            # Load with model manager
            manager = DeepLearningModelsManager()
            coronal_model = manager.get_model('LUNG_SEGMENTATION_CORONAL')
    else:
        coronal_model = None

    lung_segmenter = LungSegmenterDCNN()
    lung_segmentation = lung_segmenter.execute(op.in_ct, axial_model, coronal_model,
                                               segmentation_type=op.segmentation_type,
                                               N_subsampling = int(op.n_subsampling))

    if op.thirds:
        print ('    Splitting Segmentation in Thirds...')
        lung_splitter = LungThirdSplitter()
        lung_segmentation = lung_splitter.execute(lung_segmentation)

    sitk.WriteImage(lung_segmentation, op.out_lm, True)
