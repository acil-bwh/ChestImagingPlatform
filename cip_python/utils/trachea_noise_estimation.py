import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.ndimage import binary_erosion
from scipy.stats import truncnorm
from scipy.stats import skew
from scipy.stats import kurtosis

from cip_python.input_output.image_reader_writer import ImageReaderWriter


class TracheaNoiseEstimation:
    def __init__(self):
        self.io = ImageReaderWriter()

    @staticmethod
    def gauss_trunc_ls(m00, v00, ht, xx):
        def errorHist(aa, ht, xx):
            mu = aa[0]
            ss = aa[1]
            ww = aa[2]
            htn = ht / np.sum(ht)
            he = 1 / np.sqrt(2 * np.pi * ss * ss) * np.exp(-(xx - mu) ** 2 / (ss * ss))
            EE = np.sum(1 / (ht + 0.1) * (ht - he) ** 2)
            # EE=np.mean((ht-he)**2)
            # EE=np.mean((np.cumsum(ht)-np.cumsum(he))**2)
            return EE

        BB = (('-inf', 'inf'), (1e-5, 'inf'), ('-inf', 'inf'))
        aa = minimize(errorHist, (m00, v00, 1.0), args=(ht, xx), bounds=BB)
        m0 = aa.x[0]
        v0 = aa.x[1]
        return m0, v0, aa.x[2]

    def get_roi_from_trachea(self,ct, lm, erosion_kernel=5):
        lmtrachea = 1 * (lm == 512)

        #In case trachea leaks into surrounding tissue, set a hard threshold
        lmtrachea[ct>-600]=0
        roi_lm= binary_erosion(lmtrachea, structure=np.ones((erosion_kernel, erosion_kernel, erosion_kernel))).astype(lmtrachea.dtype)

        return roi_lm

    def get_raw_moments(self, ct, roi_lm):
        huroi = ct[roi_lm == 1]

        mean_raw=np.mean(huroi)
        std_raw=np.std(huroi)

        # Calculate skewness and kurtosis
        skewness_raw = skew(huroi)
        kurt_raw = kurtosis(huroi)

        # Calculate the 15th percentile
        perc15_raw = np.percentile(huroi[:], 15)

        return mean_raw, std_raw, skewness_raw, kurt_raw, perc15_raw

    def estimate_truncated_noise_params(self, ct, roi_lm, truncation_th=-1022):

        huroi = ct[roi_lm == 1]

        thist, xx = np.histogram(huroi, bins=1100, range=(-1100, 0), density=True)
        xx = xx[:-1]

        if np.min(xx)>truncation_th:
            idx=0
        else:
            idx = np.where(xx > truncation_th)[0][0]  # Remove everything below -1022 (truncation_th) to take away truncation
        xx_r = xx[idx:]
        ht = thist[idx:]    # Get trachea histogram without truncation
        ht = ht/np.sum(ht)  #Renormalize histogram to take into account the portion that we have remove

        # Estimate Noise from Trachea
        mean_t = np.sum(ht * xx_r)  # Initial estimation of mean
        std_t = np.sqrt(np.sum(ht * (xx_r ** 2)) - mean_t ** 2)  # Initial estimation of variance
        me2, ve2, ww2 = self.gauss_trunc_ls(mean_t, std_t, ht, xx_r)  # Moment fitting estimation

        return me2, ve2

    @staticmethod
    def truncation_correction(ct, mean, std_dev, th=-1023):
        mask = ct <= th

        lower = float('-inf')
        upper = th
        a, b = (lower - mean) / std_dev, (upper - mean) / std_dev

        normal_values = truncnorm.rvs(a=a, b=b, loc=mean, scale=std_dev, size=np.count_nonzero(mask))
        normal_values = normal_values.astype(ct.dtype)
        np.random.shuffle(normal_values)

        ct[mask] = normal_values
        return ct

    @staticmethod
    def is_case_truncated(ct, lm):
        lm2 = lm.astype('uint8')
        lmlung = 1 * (lm2 > 0)
        hulung = ct[lmlung == 1]

        lhist, xx = np.histogram(hulung, bins=1100, range=(-1100, 0), density=True)
        xx = xx[:-1]

        idx = np.where(xx > -1022)[0][0]
        ht = lhist[:idx]

        if ct.min() == -1024 and np.sum(ht) >= 0.00001:
            return True

        return False

    def execute(self, in_ct, in_lm, out_file,erosion_kernel,cid):
        ct, metainfo = self.io.read_in_numpy(in_ct)
        lm = self.io.read_in_numpy(in_lm)[0]

        roi_lm = self.get_roi_from_trachea(ct, lm, erosion_kernel)

        # Compute raw moments in roi assuming no truncated distribution
        mm_raw, std_raw, ss_raw,kk_raw, pp_raw = self.get_raw_moments(ct, roi_lm)

        # Compute mean, std, assuming a truncated gaussian model
        mm, vv  = self.estimate_truncated_noise_params(ct, roi_lm)

        is_truncated = self.is_case_truncated(ct, lm)

        # Impute truncated values with a gaussian model and compute moments
        corrected_ct = self.truncation_correction(ct, mm, vv, th=-1023)
        mm_cor, std_cor, ss_cor,kk_cor, pp_cor = self.get_raw_moments(corrected_ct, roi_lm)

        out_csv = dict()

        out_csv['CID'] = [cid]
        out_csv['Noise Mean'] = [mm]
        out_csv['Noise Sigma'] = [vv]
        out_csv['Noise Mean Imputed'] = [mm_cor]
        out_csv['Noise Sigma Imputed'] = [std_cor]
        out_csv['Noise Skew Imputed'] = [ss_cor]
        out_csv['Noise Kurtosis Imputed'] = [kk_cor]
        out_csv['Noise Perc15 Imputed'] = [pp_cor]
        out_csv['Noise Mean Raw'] = [mm_raw]
        out_csv['Noise Sigma Raw'] = [std_raw]
        out_csv['Noise Skew Raw'] = [ss_raw]
        out_csv['Noise Kurtosis Raw'] = [kk_raw]
        out_csv['Noise Perc15 Raw'] = [pp_raw]
        out_csv['Truncation Status'] = [is_truncated]

        df = pd.DataFrame.from_dict(out_csv)
        df.to_csv(out_file, index=False, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Method to compute mean, sigma, perc15, skewness, and kurtosis'
                                                 'of the CT noise (inside the trachea). Truncation correction  '
                                                 'will be applied if required.')
    parser.add_argument('--i_ct', help="Path to CT image (.nrrd)", type=str, required=True)
    parser.add_argument('--i_lm', help="Path to lung labelmap (.nrrd)", type=str, required=True)
    parser.add_argument('-o', help="Path to save .csv file with mean, sigma, perc15, skewness, and kurtosis of the CT "
                                    "noise", type=str, required=True)
    parser.add_argument('--cid', help="Case ID", type =str, required=True)
    parser.add_argument('-k', help="Erosion kernel size", type=int,default=5)

    args = parser.parse_args()

    tr = TracheaNoiseEstimation()
    tr.execute(args.i_ct, args.i_lm, args.o,args.k,args.cid)




