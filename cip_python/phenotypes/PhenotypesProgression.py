import platform
import datetime
import numpy as np
import pandas as pd
import scipy.stats as scp
import SimpleITK as sitk
from cip_python.common import ChestConventions


class PhenotypesProgression:
    """Base class for phenotype genearting classes in progression CT scans.

    Attributes
    ----------
    pheno_names_ : list of strings
        The names of the phenotypes

    key_names_ : list of strings
        The names of the keys that are associated with each of the phenotype
        values.

    valid_key_values_ : dictionary
        The keys of the dictionary_mask are the key_mask names. Each dictionary_mask key
        maps to a list of valid DB key_mask values that each key_mask name can
        assume

    Notes
    -----
    """
    def __init__(self):
        """
        """
        # To be changed to type standard types
        self.type_list_baseline = [1,35,56]
        self.type_list_followup = [1,35,56]
        self._spacing = None
        self.min_intensity = -1024.

    def get_transition_matrix(self, x, y, mask1, mask2, t, t_max=None):
        """
        Calculate the transition according to a threshold.
        Notation:
            0 Means below the threshold
            1 Means beyond the threshold
            First index is baseline
            Second index is followup
        """
        x_mask = x[mask1]
        y_mask = y[mask2]
        matrix_out = np.zeros([2,2], 'float')

        if t_max is not None:
            mask_tmax = x_max < t_max
        else:
            mask_tmax = 1.0

        matrix_out[0, 0] = np.sum((x_mask <= t) * (y_mask <= t) * mask_tmax)
        matrix_out[0, 1] = np.sum((x_mask <= t) * (y_mask > t) * mask_tmax)
        matrix_out[1, 0] = np.sum((x_mask > t) * (y_mask <= t) * mask_tmax)
        matrix_out[1, 1] = np.sum((x_mask > t) * (y_mask > t) * mask_tmax)

        return matrix_out

    ################
    # Transition matrices for different thresholds.
    #
    # list_TML = ['TMLAA950', 'TMLAA925', 'TMLAA910', 'TMLAA905', 'TMLAA900', 'TMLAA875', 'TMLAA865', 'TMHAA700',
    #             'TMHAA600', 'TMHAA500', 'TMHAA250', 'TMHAA600To250']
    # for aux in list_TML:
    #     t = int(aux[5:8])
    #     # print("def get_" + aux + "(self, x, y, mask1, mask2, mask_types1, mask_types2):")
    #     # print("\t return self.get_transition_matrix(x, y, mask1, mask2, t=-%d)" % t)
    #     # print("")
    #     print("paren_pheno.get_" + aux + "(x, y, mask1, mask2, None, None)")

    def get_TMLAA950(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-950)

    def get_TMLAA925(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-925)

    def get_TMLAA910(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-910)

    def get_TMLAA905(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-905)

    def get_TMLAA900(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-900)

    def get_TMLAA875(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-875)

    def get_TMLAA865(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-865)

    def get_TMHAA700(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-700)

    def get_TMHAA600(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-600)

    def get_TMHAA500(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-500)

    def get_TMHAA250(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-250)

    def get_TMHAA600To250(self, x, y, mask1, mask2, mask_types1, mask_types2):
        return self.get_transition_matrix(x, y, mask1, mask2, t=-600)

    # The thresholds is set with respect to the baseline
    def get_TMPERC10(self, x, y, mask1, mask2, mask_types1, mask_types2):
        x_mask = x[mask1]
        y_mask = y[mask2]
        matrix_out = np.zeros([2, 2])

        t_percentile = np.percentile(x_mask, 10)
        return self.get_transition_matrix(x, y, mask1, mask2, t=t_percentile)

    def get_TMPERC15(self, x, y, mask1, mask2, mask_types1, mask_types2):
        x_mask = x[mask1]
        y_mask = y[mask2]
        matrix_out = np.zeros([2, 2])

        t_percentile = np.percentile(x_mask, 15)
        return self.get_transition_matrix(x, y, mask1, mask2, t=t_percentile)

    def get_TMTypes(self, x, y, mask1, mask2, mask_types1, mask_types2):
        matrix_out = np.zeros([len(self.type_list_baseline), len(self.type_list_followup)], 'float')

        for index1, type1 in enumerate(self.type_list_baseline):
            for index2, type2 in enumerate(self.type_list_followup):
                matrix_out[index1, index2] = np.sum(x[mask_types1 == type1] * y[mask_types2 == type2])

        return matrix_out

    def get_Volume(self, x, y, mask1, mask2, mask_types1, mask_types2):
        x_vol = np.prod(self._spacing)*float(mask1.sum())/1e6
        y_vol = np.prod(self._spacing)*float(mask2.sum())/1e6
        return [x_vol, y_vol]

    def get_Paired_Delta(self, x, y, mask1, mask2, mask_types1, mask_types2):
        x_mask = x[mask1]
        y_mask = y[mask2]

        delta_xy = y_mask - x_mask
        delta_xyPositive = delta_xy[delta_xy >= 0]
        delta_xyNegative = delta_xy[delta_xy < 0]

        list_labels = ['DeltaPairedHUMean', 'DeltaPairedHUStd', 'DeltaPairedHUKurtosis', 'DeltaPairedHUSkewness',
                       'DeltaPairedHUMode', 'DeltaPairedHUMedian', 'DeltaPairedHUMin', 'DeltaPairedHUMax']
        list_functions = [np.mean, np.std, scp.stats.kurtosis, scp.stats.skew, scp.stats.mode, np.median,
                          np.min, np.max]

        list_suffix = ['','Positive','Negative']
        list_variables = [delta_xy, delta_xyPositive, delta_xyNegative]

        dic_paired_delta = dict()
        for suffix, variable in zip(list_suffix,list_variables):
            # print("")
            for prefix, func in zip(list_labels,list_functions):
                label = prefix + suffix
                # print("dic_paired_delta['%s'] = %s(%s)" % (label, func.__name__, "delta_xy" + suffix))
                if len(variable) == 0:
                    dic_paired_delta[label] = None
                    continue
                else:
                    dic_paired_delta[label] = func(variable)

        if len(delta_xyPositive) == 0:
            dic_paired_delta['VolumeDeltaPairedPositive'] = 0
        else:
            dic_paired_delta['VolumeDeltaPairedPositive'] = np.prod(self._spacing) * len(delta_xyPositive) / 1e6

        if len(delta_xyPositive) == 0:
            dic_paired_delta['VolumeDeltaPairedNegative'] = 0
        else:
            dic_paired_delta['VolumeDeltaPairedNegative'] = np.prod(self._spacing) * len(delta_xyNegative) / 1e6

        return dic_paired_delta

    def get_Paired_Ratio(self, x, y, mask1, mask2, mask_types1, mask_types2):
        x_mask = x[mask1]
        y_mask = y[mask2]
        x_mask.clip(self.min_intensity)
        y_mask.clip(self.min_intensity)

        # To prevent from division by zero
        x_mask = x_mask - self.min_intensity + 1
        y_mask = y_mask - self.min_intensity + 1
        x_mask = x_mask.astype('float')
        y_mask = y_mask.astype('float')

        ratio_xy = y_mask / x_mask
        ratio_xyAbove1 = ratio_xy[ratio_xy >= 1]
        ratio_xyBelow1 = ratio_xy[ratio_xy < 1]

        list_labels = ['RatioPairedHUMean', 'RatioPairedHUStd', 'RatioPairedHUKurtosis', 'RatioPairedHUSkewness',
                       'RatioPairedHUMode', 'RatioPairedHUMedian', 'RatioPairedHUMin', 'RatioPairedHUMax']

        list_functions = [np.mean, np.std, scp.stats.kurtosis, scp.stats.skew, scp.stats.mode, np.median,
                          np.min, np.max]

        list_suffix = ['','Above1','Below1']
        list_variables = [ratio_xy,ratio_xyAbove1,ratio_xyBelow1]

        dic_paired_ratio = dict()
        for suffix, variable in zip(list_suffix, list_variables):
            # print("")
            for prefix, func in zip(list_labels, list_functions):
                label = prefix + suffix
                # print("dic_paired_ratio['%s'] = %s(%s)" % (label,func.__name__,"ratio_xy"+suffix))
                if len(variable) == 0:
                    dic_paired_ratio[label] = None
                    continue
                else:
                    dic_paired_ratio[label] = func(variable)

        if len(ratio_xyAbove1) == 0:
            dic_paired_ratio['VolumeRatioPairedAbove1'] = None
        else:
            dic_paired_ratio['VolumeRatioPairedAbove1'] = np.prod(self._spacing) * len(ratio_xyAbove1) / 1e6

        if len(ratio_xyBelow1) == 0:
            dic_paired_ratio['VolumeRatioPairedBelow1'] = None
        else:
            dic_paired_ratio['VolumeRatioPairedBelow1'] = np.prod(self._spacing) * len(ratio_xyBelow1) / 1e6

        return dic_paired_ratio

    # Conditional statistics for local changes of density (T1 -T0). Condition applies to baseline scan.
    def get_Conditioned_Statistics(self, x, y, mask1, mask2, mask_types1, mask_types2):
        list_t = [-950, -500, -600]
        list_t_max = [None, None, -250]
        list_suffix = ['','Positive','Negative']
        list_prefix = ['DeltaPairedHUMean', 'DeltaPairedHUStd', 'DeltaPairedHUKurtosis', 'DeltaPairedHUSkewness',
                       'DeltaPairedHUMode', 'DeltaPairedHUMedian', 'DeltaPairedHUMin', 'DeltaPairedHUMax']

        list_functions = [np.mean, np.std, scp.stats.kurtosis, scp.stats.skew,
                          scp.stats.mode, np.median, np.min, np.max]

        x_mask = x[mask1]
        y_mask = y[mask2]

        dic_conditioned_stats = dict()
        for t, t_max in zip(list_t, list_t_max):
            # Condition in the baseline
            if t_max is not None:
                indices = x_mask < t_max
            else:
                indices1 = x_mask >= t
                indices2 = x_mask < t_max
                indices = indices1 * indices2

            delta_conditioned = y_mask[indices] - x_mask[indices]
            delta_conditionedPositive = delta_conditioned[delta_conditioned>=0]
            delta_conditionedNegative = delta_conditioned[delta_conditioned<0]

            list_variables = [delta_conditioned, delta_conditionedPositive, delta_conditionedNegative]

            for suffix, variable in zip(list_suffix,list_variables):
                # print("")
                for prefix, func in zip(list_prefix, list_functions):
                    if t_max is None:
                        label = prefix + suffix + str(-t)
                    else:
                        label = prefix + suffix + str(-t) + 'To' + str(-t_max)

                    # print("dic_conditioned_stats['%s'] = %s(%s)" % (label, func.__name__, "delta_conditioned" + suffix))
                    if len(variable) == 0:
                        dic_conditioned_stats[label] = None
                        continue
                    else:
                        dic_conditioned_stats[label] = func(variable)

            if t_max is None:
                suffix = 'str(-t)'
            else:
                suffix = str(-t) + 'To' + str(-t_max)

            if len(delta_conditionedPositive) == 0:
                dic_paired_delta['VolumeDeltaPairedPositive' + suffix] = 0
            else:
                dic_paired_delta['VolumeDeltaPairedPositive'+ suffix] = np.prod(self._spacing) * len(delta_conditionedPositive) / 1e6

            if len(delta_conditionedPositive) == 0:
                dic_paired_delta['VolumeDeltaPairedNegative' + suffix] = 0
            else:
                dic_paired_delta['VolumeDeltaPairedNegative'+ suffix] = np.prod(self._spacing) * len(delta_conditionedNegative) / 1e6

        return dic_conditioned_stats


########################################
# Debugging

# print("################")
# print("DEBUGGING MODE!")
# print("################")
#
# scans_dir = '/Users/gvegas/data/temp'
# cid1 = '19020F_INSP_STD_UIA_COPD'
# cid2 = '19020F_INSP_STD_UIA_COPD'
# ct1_path = scans_dir + '/' + cid1 + '.nrrd'
# lm1_path = scans_dir + '/' + cid1 + '_partialLungLabelMap.nrrd'
# ct2_path = scans_dir + '/' + cid2 + '.nrrd'
# lm2_path = scans_dir + '/' + cid2 + '_partialLungLabelMap.nrrd'
# ct1_sitk = sitk.ReadImage(ct1_path)
# ct1 = sitk.GetArrayFromImage(ct1_sitk)
# ct2_sitk = sitk.ReadImage(ct2_path)
# ct2 = sitk.GetArrayFromImage(ct2_sitk)
# lm1_sitk = sitk.ReadImage(lm1_path)
# lm1 = sitk.GetArrayFromImage(lm1_sitk).astype('bool')
# lm2_sitk = sitk.ReadImage(lm2_path)
# lm2 = sitk.GetArrayFromImage(lm2_sitk).astype('bool')
#
# regions = "WholeLung,LeftLung,RightLung,UpperThird,MiddleThird,LowerThird,LeftUpperThird,LeftMiddleThird,LeftLowerThird,RightUpperThird,RightMiddleThird,RightLowerThird"
# regions = regions.split(',')
# types = None
# pairs = None
#
# chest_regions = regions
#
# paren_pheno = PhenotypesProgression()
# paren_pheno._spacing = ct1_sitk.GetSpacing()
# self = paren_pheno
#
# x, y, mask1, mask2, mask_types1, mask_types2 = ct1, ct2, lm1, lm2, None, None
#
#
# paren_pheno.get_Paired_Delta(x, y, mask1, mask2, mask_types1, mask_types2)
# paren_pheno.get_Paired_Ratio(x, y, mask1, mask2, mask_types1, mask_types2)
# paren_pheno.get_Conditioned_Statistics(x, y, mask1, mask2, mask_types1, mask_types2)
