
from scipy import stats
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt



class CTHarmonization():

    def __init__(self):
        self.mu_air = -1000
        self.mu_blood = 50
        self.air_reference=-1024

    def myplot(self,I, title, **kwargs):
        plt.figure()
        plt.imshow(I, **kwargs)
        plt.title(title)


    def linear_fit_plot(self,np_air, hu_air, lfit, **kwargs):
        """Linera fit plot bewteen the noise variance and its density.

        Parameters
        ----------
        np_air : Noise variance
        hu_air : Density variance of the noise
        lfit : Linear fit between the noise variance and density

        Returns
        -------
        Return the plot axis

        """
        fig = None
        if "ax" in kwargs:
            ax = kwargs["ax"]
            ret_val = ax
        else:
            fig, ax = plt.subplots()
            ret_val = fig, ax
        ax.scatter(np_air, hu_air)
        ax.plot(np_air, lfit.intercept + lfit.slope * np_air, "-r")

        return ret_val


    def clip(self,img, lower_bound=-1024):
        """Clip the image so that all values below `lower_bound` are set to `lower_bound`.

        Parameters
        ----------
        img : sitk.Image
        lower_bound: set all the values below lower_bound to lower_bound

        Returns
        -------
        sitk.Image

        """
        maximum_filter = sitk.MinimumMaximumImageFilter()
        maximum_filter.Execute(img)
        upper_bound = maximum_filter.GetMaximum()
        img = sitk.Clamp(img, lowerBound=lower_bound, upperBound=upper_bound)
        return img


    def binary_erode(self,mask, kernel_radio=[5, 5, 1]):
        """Erode the segmentation according to the kernel_radio.

        Parameters
        ----------
        mask : sitk.Image
        kernel_radio : radio used to erode the segmentation, optional

        Returns
        -------
        sitk.Image

        """
        ff = sitk.BinaryErodeImageFilter()
        ff.SetKernelRadius(kernel_radio)
        mask = ff.Execute(mask)
        return mask


    def local_central_moment_estimator(self,X, X_mean=None, radius=[4, 4, 1], order=1):
        """Compute the local central moments of an image using a mean filter.

        Parameters
        ----------
        X : Image sitk.Image
        X_mean : Local mean/estimated image sitk.Image, optional. If not provided the mean filter is used to compute the mean.
        radius : radio of the local kernel to estimate the mean value, optional
        order : Moment order (Mean: order 1, Variance: order 2, etc..)

        Returns
        -------
        Local central Moment estimated using a low pass filter (MeanFilter)

        """
        if order > 1 and X_mean is None:
            mean_filter = sitk.MeanImageFilter()
            mean_filter.SetRadius(radius)
            # Local mean: E[X]
            X_mean = mean_filter.Execute(X)

        # Local Centered Moment: E[(X - X_mean)^order]
        if X_mean is not None:
            X = X - X_mean

        mean_filter = sitk.MeanImageFilter()
        mean_filter.SetRadius(radius)
        local_moment = mean_filter.Execute(X**order)

        return local_moment


    def spatial_bias_estimation(self,lfit_air, Is, Inp, delta):
        """Spatially-variant bias estimation.

        Parameters
        ----------
        lfit_air : Linear fit between the nosie variance and signal density in air
        Is : Image signal (noise free) (np.array)
        Inp : Variance noise image (np.array)
        delta : air reference

        Returns
        -------
        Spatialy-variant bias image estimatino (np.array)

        """
        Ilf = lfit_air.slope * Inp + lfit_air.intercept
        Ib = np.full_like(Inp, lfit_air.slope)
        idx = (Is > Ilf) & ~(Is == delta)  # Avoid division by 0
        Ib[idx] = lfit_air.slope * (Ilf[idx] - delta) / (Is[idx] - delta)

        return Ib


    def bias_correction(self,
        mask_air,
        mask_blood,
        signal,
        variance_noise,
        spatial_bias=True,
        systematic_bias=True,
        air_reference=-1024,
        lfit_plot=False,
    ):
        """Correction of spatially-variant and systematic biases.

        Parameters
        ----------
        mask_air : Mask image of the trachea used to measure the air density
        (sitk.Image)
        mask_blood : Mask image of the aorta used to measure the blood density
        (sitk.Image)
        signal : Denoised CT scan image (sitk.Image)
        variance_noise : Spatially variance of the CT noise (sitk.Image)
        spatial_bias : Correct the signal by the spatially-variant bias
        systematic_bias : Correct the signal by the systemic bias
        air_reference : air HU reference value, optional
        lfit_plot : The linear fit between the air density and noise variance is
        pltted if it is True, optional

        Returns
        -------
        Signal corrected by tge spatially-variand and/or systematic bias (sitk.Image)

        """
        # Get Array view
        Is = sitk.GetArrayFromImage(signal)  # Make a copy
        Ivn = sitk.GetArrayViewFromImage(variance_noise)
        Im_air = sitk.GetArrayViewFromImage(mask_air)
        Im_blood = sitk.GetArrayViewFromImage(mask_blood)

        idx_air = np.where(Im_air)
        idx_blood = np.where(Im_blood)

        hu_air = Is[idx_air]  # Density of the trachea
        # Just in case we discard wrong density values
        idx = hu_air < -890
        idx_air = tuple([idx_air[i][idx] for i in range(len(idx_air))])

        Ix = Is
        if spatial_bias:
            hu_air = Is[idx_air]  # Density of the trachea
            vn_air = Ivn[idx_air]  # Noise power of the trachea

            lfit = stats.linregress(vn_air, hu_air)
            if lfit_plot:
                linear_fit_plot(vn_air, hu_air, lfit)
            Ibias = self.spatial_bias_estimation(lfit, Is, Ivn, air_reference)
            #Remove local bias term: Ibias * Ivn
            Ix -= Ibias * Ivn

        if systematic_bias:
            E_air = Ix[idx_air].mean()
            E_blood = Ix[idx_blood].mean()

            I_lambda = (Ix - E_air) / (E_blood - E_air)
            mu_air = -1000
            mu_blood = 50
            Ix = (1 - I_lambda) * mu_air + I_lambda * mu_blood

        hsignal = sitk.GetImageFromArray(Ix)
        hsignal.CopyInformation(signal)

        localbias = sitk.GetImageFromArray(Ibias*Ivn)
        localbias.CopyInformation(signal)

        return hsignal, localbias

    def adpative_epsilon(self,sigma_noise,c_sigma,epsilon_min):
        """
        Compute a spatially adaptive epsilon offset for low-signal stabilization.

        This function defines a locally adaptive offset ε(r) used to regularize
        square-root–based variance stabilization in low-signal regions. The offset
        is defined pointwise as:

            ε(r) = max(ε_min, c_sigma · σ(r))

        where σ(r) is the local noise standard deviation.

        The adaptive epsilon prevents numerical instability and breakdown of
        Taylor-based approximations near air or other low-density regions, where
        the signal-to-noise ratio is low and nonlinear transforms become ill-conditioned.

        Parameters
        ----------
        sigma_noise : np.ndarray
            Array containing the local noise standard deviation σ(r), expressed
            in the same units as the signal (e.g., HU or density units).

        c_sigma : float
            Scaling factor controlling how aggressively ε adapts to local noise
            levels. Typical values are in the range [1, 3].

        epsilon_min : float
            Minimum allowable epsilon value. Acts as a global lower bound to
            ensure stability even in regions with very low estimated noise.

        Returns
        -------
        np.ndarray
            Array of the same shape as `sigma_noise` containing the spatially
            adaptive epsilon ε(r).

        Notes
        -----
        - The operation is performed element-wise.
        - Choosing ε proportional to σ ensures that stabilization strength
          increases naturally in low-SNR regions.
        - This epsilon is intended to be added inside nonlinear transforms,
          e.g., sqrt(X + ε), to improve accuracy near zero.
        """
        return np.maximum(epsilon_min,c_sigma * sigma_noise)

    def E1sqrtX_taylor_expansion(self,img, signal, E, order=1,use_adpative_epsilon=True,c_sigma=2,epsilon_min=10):
        """Apply the Taylor series expansion to estimate E[sqrt(X)]

        Apply the Taylor expansion of the random variable X around a:
        sqrt(X) = sqrt(a) + coeff1 * (X-a) - coeff2 * (X-a)^2 + coeff3 * (X-a)^3
                   - coeff4 * (X-a)^4
        where
         coeff1 : 1/2 * a^-1/2
         coeff2 : 1/8 * a^-3/2
         coeff3 : 1/16 * a^-5/2
         coeff4 : 5/128 a^-7/2

        E[sqrt(X)] = sqrt(a) + coeff1 * E[X-a] - coeff2 * E[(X-a)^2]
                       + coeff3 * E[(X-a)^3] - coeff4 * E[(X-a)^4}

        where X is the noisy signal, i.e. the random variable, 'a' is the
        denoised signal of X, i.e. the constant signal, and E[(X-a)^k] is the
        moment of order k centered on 'a'

        Parameters
        ----------
        img : Noisy image (sitk.Image)
        signal : Denoised image (sitk.Image)
        E : local moments E1, E2, ...

        Returns
        -------
        return an image (sitk.Image) of the E[sqrt(X)] estimation
        """
        # Ensure all images have the same size
        if img.GetSize() != signal.GetSize():
            raise ValueError(
                "Noisy and estimated images must have the same dimensions."
            )

        taylor_order = [1, 2, 3, 4]
        taylor_coeff = [1.0 / 2.0, -1.0 / 8.0, 1.0 / 16.0, -5.0 / 128.0]
        taylor_exp = [1, 3, 5, 7]

        if use_adpative_epsilon:
            sigma_noise=np.sqrt(sitk.GetArrayViewFromImage(E[1]))
            Is_sqrt = np.sqrt(sitk.GetArrayViewFromImage(signal) + self.adpative_epsilon(sigma_noise,c_sigma,epsilon_min))
        else:
            Is_sqrt = np.sqrt(sitk.GetArrayViewFromImage(signal))

        E1sqrtX = Is_sqrt

        Is_sqrt_inv = 1.0 / Is_sqrt
        # Handle potential divisions by zero or negative values
        Is_sqrt_inv = np.nan_to_num(Is_sqrt_inv, nan=0.0, posinf=0.0, neginf=0.0)

        for torder, tcoeff, texp, Ek in zip(
            taylor_order, taylor_coeff, taylor_exp, E
        ):
            Is_sqrt_inv_k = Is_sqrt_inv**texp
            IEk = sitk.GetArrayViewFromImage(Ek)
            E1sqrtX += tcoeff * IEk * Is_sqrt_inv_k

        # Clamp small negative values introduced by numerical errors in the expansion.
        # These values arise primarily from kernel border effects and are not
        # physically meaningful.
        E1sqrtX[E1sqrtX < 0] = 0.0

        # Convert the result back to a SimpleITK image
        imgE1sqrtX = sitk.GetImageFromArray(E1sqrtX)
        imgE1sqrtX.CopyInformation(img)

        return imgE1sqrtX


    def E2sqrtX_taylor_expansion(self,img, signal, E,use_adpative_epsilon=True,c_sigma=2,epsilon_min=10):
        """
        Estimate Var[sqrt(X)] using a Taylor series expansion.

        This function applies a Taylor expansion of the nonlinear transform
        sqrt(X) around the mean signal a = E[X], and propagates the centered
        moments of X to obtain an approximation of the variance of sqrt(X):

            Var(sqrt(X)) = E[(sqrt(X) - E[sqrt(X)])^2]

        Using the expansion of sqrt(a + noise), with noise = X - a and E[noise] = 0,
        the variance can be approximated (up to fourth-order central moments) as:

            Var(sqrt(X)) ≈
                (1 / (4a))   μ2
              - (1 / (8a^2)) μ3
              - (1 / (64a^3)) μ2^2
              + (5 / (64a^3)) μ4

        where:
            μk = E[(X - a)^k]

        and:
            a = E[X]        (mean / denoised signal)
            μ2 = Var(X)     (second central moment)
            μ3              third central moment
            μ4              fourth central moment

        This approximation assumes that fluctuations around a are small
        relative to a, and that X is predominantly positive.

        Parameters
        ----------
        img : sitk.Image
            Noisy image representing the random variable X.

        signal : sitk.Image
            Denoised or mean image representing a = E[X].

        E : sequence of sitk.Image
            Local centered moments of X around a, where:
                E[1] = μ1 = E[X - a] (≈ 0),
                E[2] = μ2,
                E[3] = μ3,
                E[4] = μ4.

        Returns
        -------
        sitk.Image
            Image containing the Taylor-based approximation of Var(sqrt(X)).

        Notes
        -----
        - The leading term (μ2 / (4a)) corresponds to the delta-method result.
        - Higher-order terms account for skewness and kurtosis of the noise.
        - If X is approximately Gaussian, μ3 ≈ 0 and μ4 ≈ 3μ2^2, simplifying
          the expression.
        """

        # Ensure all images have the same size
        if img.GetSize() != signal.GetSize():
            raise ValueError(
                "Noisy and estimated images must have the same dimensions."
            )

        #Shift singal to improve Taylor expansion stability for low values 
        #based on multiplier of local variance
        if use_adpative_epsilon:
            sigma_noise=np.sqrt(sitk.GetArrayViewFromImage(E[1]))
            Is = sitk.GetArrayViewFromImage(signal) + self.adpative_epsilon(sigma_noise,c_sigma,epsilon_min)
        else:
            Is = sitk.GetArrayViewFromImage(signal)
 
        Is2 = Is**2
        Is3 = Is**3
        Isinv = 1.0 / Is
        Is2inv = 1.0 / Is2
        Is3inv = 1.0 / Is3
        
        # Handle potential divisions by zero or negative values
        Isinv = np.nan_to_num(Isinv, nan=0.0, posinf=0.0, neginf=0.0)
        Is2inv = np.nan_to_num(Is2inv, nan=0.0, posinf=0.0, neginf=0.0)
        Is3inv = np.nan_to_num(Is3inv, nan=0.0, posinf=0.0, neginf=0.0)

        taylor_coeff = [Isinv / 4.0, -Is2inv / 8.0, -Is3inv / 64.0, Is3inv * 5.0 / 64.0]
        taylor_momens = [E[1], E[2], E[1] ** 2, E[3]]

        E2sqrtX = np.zeros(Is.shape)
        for tcoeff, Ek in zip(taylor_coeff, taylor_momens):
            E2sqrtX += tcoeff * sitk.GetArrayViewFromImage(Ek)

        # Convert the result back to a SimpleITK image
        imgE2sqrtX = sitk.GetImageFromArray(E2sqrtX)
        imgE2sqrtX.CopyInformation(img)

        return imgE2sqrtX


    def noise_stabilization(self,img, signal, E, verbose=False):
        """Compute the noise stabilization

        Parameters
        ----------
        img : Noisy image (sitk.Image)
        signal : Denoisy image (sitk.Image)
        E : local moments list(sitk.Image)

        Returns
        -------
        Stabilized detail (sitk.Image)

        """
        """
        Compute a stabilized (approximately variance-normalized) detail/noise field.

        This function constructs a dimensionless stabilized residual D(r) by working
        in a square-root–transformed intensity domain and normalizing by the local
        standard deviation of sqrt(X). To improve robustness at low signal levels
        (near air) and prevent instabilities when the local mean approaches zero,
        a spatially adaptive offset ε(r) is added prior to applying the square-root
        transform.

        Let X denote the noisy image (in HU), a denote the denoised/mean signal image. 
        This code assumes that the input a clamped and shifted intensity (density-like) variable: 
        Y{r] = Y(r) = max(X(r) - δ, 0).
       
         where δ is a fixed shift (typically δ = -1000 HU). 
        To ensure numerical stability at low signal levels (near air), a spatially
        adaptive offset ε(r) is introduced:
    
            Z(r) = Y(r) + ε(r)

        where ε(r) is computed from the local noise standard deviation:

            ε(r) = max(ε_min, c_sigma * σ(r))

        with σ(r) estimated from the second local central moment (variance).

        The stabilized detail is then computed voxel-wise as:

            D(r) = ( sqrt(Z(r)) - E[sqrt(Z(r))] ) / sqrt( Var[sqrt(Z(r))] )

        where E[sqrt(Z)] and Var[sqrt(Z)] are approximated using Taylor/moment
        expansions around the local mean signal a (via precomputed local moments E).

        The implementation includes safeguards for:
        - negative or near-zero estimated variances,
        - numerical NaNs/Infs,
        - and removal of voxels outside the field-of-view (FOV) using a mask derived
          from (Y - signal).

        Parameters
        ----------
        img : sitk.Image
            Noisy input image representing the random variable X (typically in HU or
            shifted intensity units depending on upstream processing).

        signal : sitk.Image
            Denoised / expected signal image (a = E[X]) used as the expansion center
            and for the FOV mask.

        E : list[sitk.Image]
            Local centered moments of X around the signal, typically:
                E[1] : μ2 = E[(X - a)^2] (local variance)
                E[2] : μ3 = E[(X - a)^3]
                E[3] : μ4 = E[(X - a)^4]
            (Indexing may vary depending on how E is assembled upstream.)

        verbose : bool, optional
            If True, print diagnostic information about numerical failures
            (e.g., negative variance estimates) and optionally write debugging
            volumes. Default is False.

        Returns
        -------
        sitk.Image
            Stabilized detail image D(r) (float32), intended to behave approximately
            like a standardized noise field (mean ~ 0, variance ~ 1) in regions where
            the model assumptions hold.

        Notes
        -----
        - This function currently uses NumPy array views for intermediate steps.
          Any modifications occur on NumPy arrays and are converted back to a
          SimpleITK image before returning.
        - In low-signal regions, the adaptive ε(r) term is essential to bound the
          behavior of the square-root transform and Taylor coefficients.
        - If the estimated Var[sqrt(Z)] is <= machine epsilon, the corresponding
          voxel is assigned D(r)=0 to avoid unstable amplification.

    """


        # Apply Taylor expansion for E{sqrt(X)} and Var{sqrt(X)}
        E1sqrtX = self.E1sqrtX_taylor_expansion(img, signal, E,use_adpative_epsilon=True,c_sigma=2,epsilon_min=10)
        E2sqrtX = self.E2sqrtX_taylor_expansion(img, signal, E,use_adpative_epsilon=True,c_sigma=2,epsilon_min=10)

        # Get numpy arrays
        #Use adaptive epsilon for image with noise.
        sigma_noise=np.sqrt(sitk.GetArrayViewFromImage(E[1]))
        I = sitk.GetArrayViewFromImage(img) + self.adpative_epsilon(sigma_noise,c_sigma=2,epsilon_min=10)
        Is = sitk.GetArrayViewFromImage(signal)
        IE1sqrtX = sitk.GetArrayViewFromImage(E1sqrtX)
        IE2sqrtX = sitk.GetArrayViewFromImage(E2sqrtX)

        # Stabilized detail
        sigma_sqrtX = np.sqrt(IE2sqrtX)
        mask_zero_var = sigma_sqrtX<np.finfo(np.float64).eps
        sigma_sqrtX[mask_zero_var]=1.0

        D = (np.sqrt(I) - IE1sqrtX) / sigma_sqrtX
        #Set detail to zero
        D[mask_zero_var]=0.0

        # Handle potential divisions by zero or negative values
        D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

        # Remove values outside the FOV
        Imask = (I - Is) == 0
        D = D * (1 - Imask)

        imgD = sitk.GetImageFromArray(D)
        imgD.CopyInformation(img)
        imgD = sitk.Cast(imgD, sitk.sitkFloat32)
        return imgD


    def noise_stabilization_G_assumption(self,img, signal, variance_noise):
        """Compute the noise estabilization assuming the noise follows a Gaussian
        distribution

        Parameters
        ----------
        img : Noisy image (sitk.Image)
        signal : Denoisy image (sitk.Image)
        variance_noise : variance of the noise (sitk.Image)

        Returns
        -------
        Estabilized details (sitk.Image)

        """
        # Get numpy arrays
        I = sitk.GetArrayViewFromImage(img)
        Is = sitk.GetArrayViewFromImage(signal)
        # We use the gaussian normalization on those pixels where the taylor
        # expansion fails
        sigmag = np.sqrt(sitk.GetArrayViewFromImage(variance_noise))
        Dg = (I - Is) / sigmag
        # Handle potential divisions by zero or negative values
        Dg = np.nan_to_num(Dg, nan=0.0, posinf=0.0, neginf=0.0)

        imgDg = sitk.GetImageFromArray(Dg)
        imgDg.CopyInformation(img)
        return imgDg

    def execute(self,img,signal,noise_std_ref,label_map,label_air,label_blood,
                gaussian_noise=False, systematic_bias=False, spatial_bias=False, air_reference=-1024):
        """
        Perform signal harmonization, bias correction, and noise stabilization
        on a CT image using local moment estimation.

        This method implements a three-stage pipeline:
        (1) local moment estimation for noise characterization,
        (2) optional correction of systematic and/or spatial bias using air and
            blood reference regions,
        (3) noise stabilization and reconstruction of an adjusted image with
            controlled noise properties.

        Parameters
        ----------
        img : sitk.Image
            Input CT image to be harmonized (e.g., raw or reconstructed image).

        signal : sitk.Image
            Reference signal image used to estimate local moments and guide
            noise stabilization (typically a denoised or model-predicted signal).

        noise_std_ref : float
            Reference noise standard deviation used to scale the stabilized
            noise component when reconstructing the adjusted image.

        label_map : sitk.Image
            Segmentation or label image used to identify tissue classes for
            bias correction.

        label_air : int
            Label value in `label_map` corresponding to air regions.

        label_blood : int
            Label value in `label_map` corresponding to blood regions.

        gaussian_noise : bool, optional (default=False)
            If True, additionally compute a noise-stabilized image under the
            assumption of Gaussian noise statistics.

        systematic_bias : bool, optional (default=False)
            If True, apply a global (systematic) bias correction using air and
            blood reference regions.

        spatial_bias : bool, optional (default=False)
            If True, apply a spatially varying bias correction using air and
            blood reference regions.

        air_reference : int, optional (default=-1024)
            Hounsfield Unit (HU) reference value for air. Used for clipping,
            shifting, and bias correction.

        Returns
        -------
        adj_img : sitk.Image
            Noise-stabilized and bias-corrected CT image, reconstructed as:
            hsignal + noise_std_ref * imgD.
            Returned as a 16-bit signed integer image.

        imgD : sitk.Image
            Dimensionless stabilized noise field derived from the local
            moment-based noise model.

        local_bias : sitk.Image or None
            Estimated bias field from the bias correction step.
            Returned as None if neither `systematic_bias` nor `spatial_bias`
            is enabled.

        adj_img_g : sitk.Image or None
            Noise-stabilized image reconstructed under the assumption of
            Gaussian noise statistics. Returned only if `gaussian_noise=True`.

        imgDg : sitk.Image or None
            Dimensionless stabilized noise field assuming Gaussian noise.
            Returned only if `gaussian_noise=True`.

        E : List[stik.Image]
            List of central moment images  (order 1, 2, 3 and 4)

        Notes
        -----
        - Input images are clipped above `air_reference + 24 HU` and shifted
          by the air reference to improve numerical stability near air.
        - Local central moments up to 4th order are estimated to characterize
          noise statistics.
        - The function always returns a fixed-length tuple; optional outputs
          are returned as None when not computed.
        - This method is designed for quantitative CT harmonization workflows
          and assumes consistent image geometry across inputs.

        """

        # Clipping and shifting. We shift the clip to avoid unstability around the
        # air
        lower_bound = air_reference + 24
        img_clip = self.clip(img, lower_bound=lower_bound) - air_reference
        signal_clip = self.clip(signal, lower_bound=lower_bound) - air_reference

        # (1) Estimate the Local momentos used for the signal harmonization and the
        # noise stabilization.  E[1] = variance of the noise
        E = [
            self.local_central_moment_estimator(img_clip, X_mean=signal_clip, order=i)
            for i in range(1, 5)
        ]

        if systematic_bias or spatial_bias:
            # Reading blood and air mask for bias correction
            mask_air = self.binary_erode(label_map == label_air)
            mask_blood = self.binary_erode(label_map == label_blood)

            # (2) Correction of spatially-variant and systematic bias
            hsignal,local_bias = self.bias_correction(
                mask_air,
                mask_blood,
                signal,
                E[1],
                spatial_bias=spatial_bias,
                systematic_bias=systematic_bias,
                air_reference=air_reference,
            )
        else:
            hsignal = signal
            localbias = None

        # (3) Noise stabilization
        imgD = self.noise_stabilization(img_clip, signal_clip, E)

        # (4) Harmonize image using the bias corrected estimation an dthe noise detail
        adj_img = hsignal + noise_std_ref * imgD
        adj_img = sitk.Cast(sitk.Round(adj_img), sitk.sitkInt16)

        if gaussian_noise:
            # (3.1) Noise stabilization under the assumption of a Gaussian noise
            imgDg = self.noise_stabilization_G_assumption(img_clip, signal_clip, E[1])
            adj_img_g = hsignal + noise_std_ref * imgDg
        else:
            adj_img_g = None
            imgDg = None

        return adj_img, imgD, local_bias, adj_img_g, imgDg, E


def print_args(args, flush=True):
    msg = "-" * 10 + " Configuration " + "-" * 10 + "\n"
    for k, v in vars(args).items():
        msg += "{}: {}\n".format(k, v)
    msg += "-" * 35 + "\n"
    print(msg, flush=flush)
    return msg


def main(args):

    print_args(args)

    # Inputs
    fname = args.ct_path
    sfname = args.estimated_ct_path
    lfname = args.label_map_path
    # Outputs
    output_image_path = args.output_image_path
    output_noise_path = args.output_noise_path
    output_bias_path = args.output_bias_path
    output_noise_std_path = args.output_noise_std_path
    # Config.
    label_air = args.label_air
    label_blood = args.label_blood
    spatial_bias = args.spatial_bias_correction
    systematic_bias = args.systematic_bias_correction
    noise_std_ref = args.noise_std_ref
    gaussian_noise = args.gaussian_noise

    #  fname = ('/Users/ahc44/Datos/ACIL/DeepHarmonization/' +
    #           '023579003169_INSP_STD_L1_ECLIPSE_0000.nrrd')
    #  sfname = ('/Users/ahc44/Datos/ACIL/DeepHarmonization/' +
    #            '023579003169_INSP_STD_L1_ECLIPSE_NoiseAugmentation_signal.nrrd')
    #  lfname = ('/Users/ahc44/Datos/ACIL/DeepHarmonization/' +
    #            '023579003169_INSP_STD_L1_ECLIPSE_totalSegmentation_0000.nrrd')
    #  output_image_path = (
    #      '/Users/ahc44/Datos/ACIL/DeepHarmonization/' +
    #      '023579003169_INSP_STD_L1_ECLIPSE_0000_harmonized.nrrd')
    #  output_noise_path = ('/Users/ahc44/Datos/ACIL/DeepHarmonization/' +
    #                       '023579003169_INSP_STD_L1_ECLIPSE_0000_noise.nrrd')

    #  label_air = 43
    #  label_blood = 7
    #  spatial_bias = True
    #  systematic_bias = True
    #  noise_std_ref = 20
    #  gaussian_noise = False


    # Read image and signal
    img = sitk.Cast(sitk.ReadImage(fname), sitk.sitkFloat32)
    signal = sitk.Cast(sitk.ReadImage(sfname), sitk.sitkFloat32)
    
    if systematic_bias or spatial_bias:
        label_map = sitk.ReadImage(lfname)
    else: label_map = None

    harmonizer=CTHarmonization()

    adj_img,imgD,local_bias,adj_img_g,imgDg, E_img= harmonizer.execute(img,signal,noise_std_ref,label_map,label_air,label_blood,
           gaussian_noise, systematic_bias, spatial_bias)

    # Save the result
    sitk.WriteImage(adj_img, output_image_path, useCompression=True)
    print("Adjusted image saved to %s" % output_image_path)

    if output_noise_path != "":
        imgD_int = sitk.Cast(imgD * 1e3, sitk.sitkInt16)
        sitk.WriteImage(imgD_int, output_noise_path, useCompression=True)
        print("Stabilized noise saved to %s" % output_noise_path)

    if output_bias_path !="":
        sitk.WriteImage(local_bias, output_bias_path, useCompression=True)
        print("Local spatially variant bias field saved to %s" % output_noise_path)

    if output_noise_std_path != "":
        sitk.WriteImage(sitk.Sqrt(E_img[1]), output_noise_std_path, useCompression=True)
        print("Noise std saved to %s" % output_noise_path)

    if gaussian_noise:
        # (3.1) Noise stabilization under the assumption of a Gaussian noise
        oimg_path = output_image_path.split(".")
        output_imageG_path = oimg_path[0] + "_Gnoise." + oimg_path[1]

        sitk.WriteImage(adj_img_g, output_imageG_path, useCompression=True)
        print("Harmonized image (Gaussian assumption) saved to %s" % output_imageG_path)

        if output_noise_path != "":
            onoise_path = output_noise_path.split(".")
            output_noiseG_path = onoise_path[0] + "_Gnoise." + onoise_path[1]
            imgDG_int = sitk.Cast(imgDg * 1e3, sitk.sitkInt16)
            sitk.WriteImage(imgDG_int, output_noiseG_path, useCompression=True)
            print("Stabilized noise (Gaussian assumption) saved to %s" % output_noiseG_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser( description=(
            "CT harmonization using (i) optional air/blood-based bias correction and "
            "(ii) noise stabilization driven by a denoised/reference CT."
        ),
         epilog=(
            "Examples:\n"
            "  (1) Noise stabilization only:\n"
            "    ct_harmonization.py --ct_path noisy.nrrd --estimated_ct_path signal.nrrd \\\n"
            "      --output_image_path out_harmonized.nrrd --noise_std_ref 20\n\n"
            "  (2) With systematic + spatial bias correction (requires label map):\n"
            "    ct_harmonization.py --ct_path noisy.nrrd --estimated_ct_path signal.nrrd --label_map_path labels.nrrd \\\n"
            "      --systematic_bias_correction --spatial_bias_correction --label_air 43 --label_blood 7 \\\n"
            "      --output_image_path out_harmonized.nrrd --output_bias_path out_bias.nrrd\n\n"
            "  (3) Save stabilized noise and noise std:\n"
            "    ct_harmonization.py --ct_path noisy.nrrd --estimated_ct_path signal.nrrd \\\n"
            "      --output_image_path out_harmonized.nrrd --output_noise_path out_noise.nrrd \\\n"
            "      --output_noise_std_path out_noise_std.nrrd --noise_scale 1e4\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-ct",
        "--ct_path",
        type=str,
        required=True,
        default="",
        help="Path to the noisy NRRD image.",
    )
    parser.add_argument(
        "-sct",
        "--estimated_ct_path",
        type=str,
        required=True,
        default="",
        help="Path to the estimated (noise-free) NRRD image.",
    )
    parser.add_argument(
        "-n",
        "--noise_std_ref",
        type=float,
        default=20,
        help="Standard deviation of the stabilized noise in HU",
    )
    parser.add_argument(
        "-sp",
        "--spatial_bias_correction",
        action="store_true",
        help="Apply spatially varying bias correction (requires label map).",
    )
    parser.add_argument(
        "-sy",
        "--systematic_bias_correction",
        action="store_true",
        help="AApply global/systematic bias correction (requires label map).",
    )
    parser.add_argument(
        "-lm",
        "--label_map_path",
        type=str,
        default="",
        help="Path to label map (NRRD) for air and blood references. Required if bias correction is enabled.",
    )
    parser.add_argument(
        "-la", "--label_air", 
        type=int,
        default=43, 
        help="Label value corresponding to air"
    )
    parser.add_argument(
        "-lb",
        "--label_blood",
        type=int,
        default=7,
        help="Label value corresponding to blood"
    )
    parser.add_argument(
        "-gn",
        "--gaussian_noise",
        action="store_true",
        help=(
            "Additionally compute outputs under the assumption of Gaussian noise."
        )
    )
    parser.add_argument(
        "-o",
        "--output_image_path",
        type=str,
        help="Path to save the harmonized (stabilized+bias correction) output NRRD image.",
    )
    parser.add_argument(
        "-on",
        "--output_noise_path",
        type=str,
        default="",
        help=(
            "Path to save the stabilized noise. The noise is multiplied by"
            + "1e3 and saved as int16 to reduce the storage size."
        )
    )
    parser.add_argument(
        "-onstd",
        "--output_noise_std_path",
        type=str,
        default="",
        help=(
            "Path to save the noise std image"
        ),
    )
    parser.add_argument(
        "-ob",
        "--output_bias_path",
        type=str,
        default="",
        help=(
            "Path to save the spatially-variant bias image"
        ),
    )

    args = parser.parse_args()
    main(args)
