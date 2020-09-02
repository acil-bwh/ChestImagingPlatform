from __future__ import division
import numpy as np


class MonogenicSignal2D:
    """
            Class to calculate the Monogenic Signal of a 2d-array.
            Authors: Marlon C. Hidalgo-Gato and Valeria C.F. Barbosa, 2016.
            Code from: https://github.com/pinga-lab/paper-monogenic-signal/blob/master/Code/monogenic.py
    """
    def __init__(self):
        pass

    @staticmethod
    def fft_wavenumbers(x, y, shape_dat, shape_pdat):
        """
        Calculates the u, v and r Fourier wavenumbers in the x, y and radial
        directions respectively.
        Parameters:
        * x, v: 2d-arrays
            Arrays with the x and y coordinates of the data points.
        * shape_dat: tube = (ny, nx)
            The number of data points in each direction before padding.
        * shape_pdat: tube = (ny, nx)
            The number of data points in each direction after padding.
        Returns:
        * u, v, r: 2d-arrays
            x, y and radial Fourier wavenumbers.
        """

        dx = (np.amax(x) - np.amin(x))/(shape_dat[1] - 1)
        dy = (np.amax(y) - np.amin(y))/(shape_dat[0] - 1)
        fx = np.fft.fftfreq(shape_pdat[1], dx)
        fy = np.fft.fftfreq(shape_pdat[0], dy)
        u, v = np.meshgrid(fx, fy)
        r = np.sqrt(u**2 + v**2)

        return u, v, r

    @staticmethod
    def riesz_to_attributes(vx, vy, vz):
        """
        Calculates the amplitude, phase and orientation of a given vector
        v = (vx, vy, vz).
        Parameters:
        * vx, vy, vz: 2d-arrays
            x, y and z components of the vector.
        Returns:
        * amp: 2d-array
            The vector amplitude.
        * phase: 2d-array
            The vector phase.
        * orientation: 2d-array
            The vector orientation.
        """
        vz[vz == 0] += 0.0000001
        amp = np.sqrt(vx**2 + vy**2 + vz**2)
        phase = np.arctan(np.sqrt(vx**2 + vy**2)/vz)
        orient = np.arctan(vy/vx)

        return amp, phase, orient

    @staticmethod
    def fft_pad_data(data, shape_dat, n_points=10, mode='linear_ramp'):
        """
        Padd data and calculates de FFT.
        Parameters:
        * data: 2d-array
            Array with the gridded data.
        * shape_dat: tube = (ny, nx)
            The number of points in each direction of data before padding.
        * pad_pt: int
            Number of array points to pad the data.
        * pad_mode: str
            Padding mode - {
               'linear_ramp': Pads with a linear ramp between edge value and zero.
               'edge': Pads with the edge values of the data.
               'mean': pads with the mean value of all the data.
                           }
        Returns:
        * fpad: 2d-array
            The FFT of the padded data.
        * mask: 2d-array
            Location of padding points - {
                 True: data points.
                 False: padded points.
                           }
        * shape_pdat: tube = (ny, nx)
            The number of data points in each direction after padding.
        """

        data_p = np.pad(data, n_points, mode)

        shape_pdat = (shape_dat[0] + 2*n_points, shape_dat[1] + 2*n_points)

        mask = np.zeros_like(data_p, dtype=bool)
        mask[n_points:n_points+shape_dat[0], n_points:n_points+shape_dat[1]] = True
        fpdat = np.fft.fft2(data_p)

        return fpdat, mask, shape_pdat

    @staticmethod
    def ifft_unpad_data(data_p, mask, shape_dat, shape_pdat):
        """
        Calculates de inverse Fourier Transform (iFFT) of a padded array and mask
        the data to the original shape.
        Parameters:
        * data_p: 2d-array
            Array with the padded data.
        * mask: 2d-array
            Location of padding points - {
                 True: Points to be kept .
                 False: Points to be removed.
                           }
        * shape_dat: tube = (ny, nx)
            The number of data points in each direction before padding.
        * shape_pdat: tube = (ny, nx)
            The number of data points in each direction after padding.
        Returns:
        * data: 2d-array
            The unpadded space-domain data.
        """

        ifft_data = np.real(np.fft.ifft2(data_p))
        data = ifft_data[mask]

        return np.reshape(data, shape_dat)

    def nss_monogenic_signal(self, x, y, data, pad_pt=10, pad_mode='linear_ramp'):
        """
        Calculates the local amplitude, local phase and local orientation in the
        non-scale monogenic signal of data.
        Parameters:
        * x, y: 2d-arrays
            Arrays with the x and y coordinates of the data points.
        * data: 2d-array
            Array with the gridded data.
        * pad_pt: int
            Number of array points to pad the data.
        * pad_mode: str
            Padding mode - {
               'linear_ramp': Pads with a linear ramp between edge value and zero.
               'edge': Pads with the edge values of data.
               'mean': pads with the mean value of all the data.
                           }
        Returns:
        * amplitude: 2d-array
            The local amplitude.
        * phase: 2d-array
            The local phase.
        * orientation: 2d-array
            The local orientation.
        """

        shape_dat = np.shape(data)

        # Data in the Fourier domain
        F, mask, shape_pdat = self.fft_pad_data(data, shape_dat, pad_pt, pad_mode)

        # Fourier wavenumbers
        u, v, r = self.fft_wavenumbers(x, y, shape_dat, shape_pdat)

        # Put 1 in r=0 to avoid singularity
        r[r == 0] = 1

        # Riesz components in the Wavenumber domain
        RX = 1j*(u/r)
        RY = 1j*(v/r)

        # Riesz components in the space domain
        rx = self.ifft_unpad_data(RX*F, mask, shape_dat, shape_pdat)
        ry = self.ifft_unpad_data(RY*F, mask, shape_dat, shape_pdat)

        # Returns the amplitude, phase and orientation
        return self.riesz_to_attributes(rx, ry, data)

    def pss_monogenic_signal(self, x, y, data, hc=None, hf=None, pad_pt=10, pad_mode='linear_ramp'):
        """
        Calculates the local amplitude, local phase and local orientation in the
        Poisson scale-space monogenic signal of data.
        Parameters:
        * x, y: 2d-arrays
            Arrays with the x and y coordinates of the data points.
        * data: 2d-array
            Array with the gridded data.
        * pad_pt: int
            Number of array points to pad the data.
        * pad_mode: str
            Padding mode - {
               'linear_ramp': Pads with a linear ramp between edge value and zero.
               'edge': Pads with the edge values of data.
               'mean': pads with the mean value of all the data.
                           }
        * hc: float
            The coarse Poisson scale-space parameter.
            None = default parameters calculation.
        * hf: float
            The fine Poisson scale-space parameter.
            None = default parameters calculation.
        Returns:
        * amplitude: 2d-array
            The local amplitude.
        * phase: 2d-array
            The local phase.
        * orientation: 2d-array
            The local orientation.
        """

        shape_dat = np.shape(data)

        # Data in the Fourier domain
        F, mask, shape_pdat = self.fft_pad_data(data, shape_dat, pad_pt, pad_mode)

        # Fourier wavenumbers
        u, v, r = self.fft_wavenumbers(x, y, shape_dat, shape_pdat)

        # Setting defaults parameters
        if hc is None:
            shape = np.shape(data)
            dx = (np.max(x) - np.min(x))/(shape[1] - 1)
            dy = (np.max(x) - np.min(x))/(shape[0] - 1)
            hc = np.amin([dx, dy])
        if hf is None:
            hf = 0.9*hc

        # Scale-space filter kernel
        p = np.exp(-2.*np.pi*r*hf) - np.exp(-2.*np.pi*r*hc)

        # Put 1 in r=0 to avoid singularity
        r[r == 0] = 1

        # Riesz components in the Wavenumber domain
        RX = 1j*(u/r)*p
        RY = 1j*(v/r)*p

        # Riesz components and data in the Poisson scale-space in the space domain
        fbp = self.ifft_unpad_data(p*F,  mask, shape_dat, shape_pdat)
        rxp = self.ifft_unpad_data(RX*F, mask, shape_dat, shape_pdat)
        ryp = self.ifft_unpad_data(RY*F, mask, shape_dat, shape_pdat)

        # Amplitude, phase and orientation in the Poisson scale-space
        return self.riesz_to_attributes(rxp, ryp, fbp)
