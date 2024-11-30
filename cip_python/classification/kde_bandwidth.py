from __future__ import division, absolute_import, print_function
import numpy as np
from scipy import fftpack, optimize


    
def variance_bandwidth(factor, xdata):
    """
    Returns the covariance matrix:

    .. math::

        \mathcal{C} = \tau^2 cov(X)

    where :math:`\tau` is a correcting factor that depends on the method.
    """
    data_covariance = np.atleast_2d(np.cov(xdata, rowvar=1, bias=False))
    sq_bandwidth = data_covariance * factor * factor
    return sq_bandwidth


def silverman_covariance(xdata, model=None):
    """
    The Silverman bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = \left( n \frac{d+2}{4} \right)^\frac{-1}{d+4}
    """
    xdata = np.atleast_2d(xdata)
    d, n = xdata.shape
    return variance_bandwidth(np.power(n * (d + 2.) / 4.,
                              -1. / (d + 4.)), xdata)


def scotts_covariance(xdata, model=None):
    """
    The Scotts bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = n^\frac{-1}{d+4}
    """
    xdata = np.atleast_2d(xdata)
    d, n = xdata.shape
    return variance_bandwidth(np.power(n, -1. / (d + 4.)), xdata)


class botev_bandwidth(object):
    """
    Implementation of the KDE bandwidth selection method outline in:

    Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

    Based on the implementation of Daniel B. Smith, PhD.

    The object is a callable returning the bandwidth for a 1D kernel.
    """
    def __init__(self, N=None, **kword):
        if 'lower' in kword or 'upper' in kword:
            print("Warning, using 'lower' and 'upper' for botev bandwidth is "
                  "deprecated. Argument is ignored")
        self.N = N

        if hasattr(np, 'float128'):
            self.large_float = np.float128
        elif hasattr(np, 'float96'):
            self.large_float = np.float96
        else:
            self.large_float = np.float64

    def run(self, data):
        """
        Returns the optimal bandwidth based on the data
        """
        N = 2 ** 10 if self.N is None else int(2 ** np.ceil(np.log2(self.N)))

        minimum = np.min(data)
        maximum = np.max(data)
        span = maximum - minimum
        lower = minimum - span / 10 
        upper = maximum + span / 10 
        # Range of the data
        span = upper - lower
        
        # Histogram of the data to get a crude approximation of the density
        M = len(data)
        DataHist, bins = np.histogram(data, bins=N, range=(lower, upper)) 
        DataHist = DataHist / M
        DCTData = fftpack.dct(DataHist, norm=None)

        I = np.arange(1, N, dtype=int) ** 2
        SqDCTData = (DCTData[1:] / 2) ** 2
        guess = 0.1

        try:
            t_star = optimize.brentq(self._botev_fixed_point, 0, guess,
                                     args=(M, I, SqDCTData))
        except ValueError:
            t_star = .28 * N ** (-.4)
            
        #pdb.set_trace()
        return np.sqrt(t_star) * span

    def _botev_fixed_point(self,t, M, I, a2):
        l = 7
        I = self.large_float(I)
        M = self.large_float(M)
        a2 = self.large_float(a2)
        f = 2 * np.pi ** (2 * l) * np.sum(I ** l * a2 * \
                                        np.exp(-I * np.pi ** 2 * t))
        for s in xrange(l, 1, -1):
            K0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
            const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
            time = (2 * const * K0 / M / f) ** (2 / (3 + 2 * s))
            f = 2 * np.pi ** (2 * s) * \
                  np.sum(I ** s * a2 * np.exp(-I * np.pi ** 2 * time))
        return t - (2 * M * np.sqrt(np.pi) * f) ** (-2 / 5)




