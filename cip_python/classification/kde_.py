# -*- encoding: utf-8 -*-
##!/usr/bin/python
from __future__ import division

import numpy as np
import scipy as sci
import scipy.fftpack
import scipy.io
import scipy.optimize


class botev_bandwidth2():
  """
    Implementation of the KDE bandwidth selection method outline in:
    Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.
    Based on the implementation of Daniel B. Smith, PhD.
    The object is a callable returning the bandwidth for a 1D kernel.
  """
  def __init__(self, N=None, lower=np.nan, upper=np.nan):
    self.N = N
    self.MIN = lower
    self.MAX = upper
  def kde_b(self, data):
    # Parameters to set up the mesh on which to calculate
    N = 2**14 if self.N is None else int(2**sci.ceil(sci.log2(self.N)))
    if self.MIN is None or self.MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        self.MIN = minimum - Range/10 if self.MIN is None else self.MIN
        self.MAX = maximum + Range/10 if self.MAX is None else self.MAX
    # Range of the data
    R = self.MAX-self.MIN
    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = sci.histogram(data, bins=N, range=(self.MIN,self.MAX))
    DataHist = DataHist/M
    DCTData = scipy.fftpack.dct(DataHist, norm=None)
    I = [iN*iN for iN in xrange(1, N)]
    SqDCTData = (DCTData[1:]/2)**2
    # The fixed point calculation finds the bandwidth = t_star
    guess = 0.1
    try:
        t_star = scipy.optimize.brentq(self.fixed_point, 0, guess, 
                                       args=(M, I, SqDCTData))
    except ValueError:
        print ('Oops!')
        return None
    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData*sci.exp(-sci.arange(N)**2*sci.pi**2*t_star/2)
    # Inverse DCT to get density
    density = scipy.fftpack.idct(SmDCTData, norm=None)*N/R
    mesh = [(bins[i]+bins[i+1])/2 for i in xrange(N)]
    bandwidth = sci.sqrt(t_star)*R
    density = density/sci.trapz(density, mesh)
    return bandwidth, mesh, density
  def fixed_point(self,t, M, I, a2):
      l=7
      I = sci.float128(I)
      M = sci.float128(M)
      a2 = sci.float128(a2)
      f = 2*sci.pi**(2*l)*sci.sum(I**l*a2*sci.exp(-I*sci.pi**2*t))
      for s in range(l, 1, -1):
          K0 = sci.prod(xrange(1, 2*s, 2))/sci.sqrt(2*sci.pi)
          const = (1 + (1/2)**(s + 1/2))/3
          time=(2*const*K0/M/f)**(2/(3+2*s))
          f=2*sci.pi**(2*s)*sci.sum(I**s*a2*sci.exp(-I*sci.pi**2*time))
      return t-(2*M*sci.sqrt(sci.pi)*f)**(-2/5)