import SimpleITK as sitk
import argparse
import numpy as np
from PIL import Image  # Import Pillow
import pandas as pd
from skimage.metrics import structural_similarity as ssim


class AffineRegistration():

    def __init__(self):
        self._lr = 0.1 #(without scaling the images)
        #self._lr = 0.001

        self._numIter=1100
        self._shrinkFactors=[8, 4, 2]
        self._smoothingSigma=[4,2,1]


    def adjust_densities(self,fixed_image,moving_image,fixed_mask,moving_mask,sponge_model=True):

        """
        Adjust the intensity of the moving image based on volume differences inferred from masks.

        Parameters:
        - fixed_image: SimpleITK.Image, fixed reference image.
        - moving_image: SimpleITK.Image, moving image to register.
        - fixed_mask: SimpleITK.Image or None, mask for fixed image.
        - moving_mask: SimpleITK.Image or None, mask for moving image.
        - sponge_model: bool, whether to apply volume-based intensity scaling.

        Returns:
        - Tuple[SimpleITK.Image, SimpleITK.Image], intensity-adjusted fixed and moving images.
        """

        #Estimate volume change from masks
        intensity_factor=1
        if sponge_model==True:
            if fixed_mask is not None and moving_mask is not None:

                f_np=sitk.GetArrayFromImage(fixed_mask)
                m_np=sitk.GetArrayFromImage(moving_mask)

                vol_m=np.sum(m_np>0)*np.prod(moving_mask.GetSpacing())/1000.0
                vol_f=np.sum(f_np>0)*np.prod(fixed_mask.GetSpacing())/1000.0

                intensity_factor=vol_m/vol_f
        print("Intensity factor: {}".format(intensity_factor))
        fixed_image_adjusted=sitk.Multiply(sitk.Threshold(sitk.Add(fixed_image, 1000),lower=0, upper=float('inf'), outsideValue=0),1.0)
        moving_image_adjusted=sitk.Multiply(sitk.Threshold(sitk.Add(fixed_image, 1000),lower=0, upper=float('inf'), outsideValue=0),intensity_factor)

        return fixed_image_adjusted,moving_image_adjusted

    def scale_factor_from_mask(self,fixed_mask,moving_mask,factor=0.5):
        """
        Computes a scaling factor for affine registration based on the volume difference
        between the fixed and moving masks. This scaling factor adjusts the registration
        transformation by incorporating a specified factor of the volume change.

        Parameters:
        ----------
        fixed_mask : sitk.Image
            The binary mask image for the fixed image. Non-zero values represent the region of interest (ROI).
        moving_mask : sitk.Image
            The binary mask image for the moving image. Non-zero values represent the region of interest (ROI).
        factor : float, optional
            A user-defined factor to adjust the contribution of the volume change to the final scale factor. 
            Default is 0.5, which means the computed scale factor will be an average of 1 (no scaling) and the 
            ratio of the moving mask's volume to the fixed mask's volume.

        Returns:
        -------
        scale_factor : float
            The computed scaling factor that can be used to adjust the affine registration. If either the 
            fixed or moving mask is None, a scale factor of 1.0 is returned (i.e., no scaling).

        Notes:
        -----
        - The function calculates the volume of the mask in cubic millimeters (mm³) by summing the number of 
          non-zero voxels in each mask and multiplying by the voxel spacing.
        - The scale factor is computed as the ratio of the moving mask's volume to the fixed mask's volume, 
          modified by the user-provided `factor`. The formula used is:

            scale_factor = (vol_m / vol_f) + (1 - vol_m / vol_f) * factor

          where `vol_m` is the volume of the moving mask, and `vol_f` is the volume of the fixed mask.
        - This scale factor can be used in affine transformations to adjust for volume differences between 
          the two images being registered.

        Example Usage:
        --------------
        scale = self.scale_factor_from_mask(fixed_mask, moving_mask, factor=0.7)
        """
        if fixed_mask is not None and moving_mask is not None:

            f_np=sitk.GetArrayFromImage(fixed_mask)
            m_np=sitk.GetArrayFromImage(moving_mask)

            vol_m=np.sum(m_np>0)*np.prod(moving_mask.GetSpacing())/1000.0
            vol_f=np.sum(f_np>0)*np.prod(fixed_mask.GetSpacing())/1000.0

            scale_factor = vol_m/vol_f + (1-vol_m/vol_f)*(1-factor) 

        else:
            scale_factor = 1.0

        return scale_factor


    def multiresolution_registration(self,fixed_image, moving_image, metric="ncc", transform_type="affine", fixed_mask=None,moving_mask=None,initial_transform=None,center_images=False, 
                                    use_mask_to_center=False,centering_type="geometry",scaling_from_mask=False,scaling_factor=0.4,scaling_th=2):

        """
        Perform multi-resolution image registration between fixed and moving images.

        Parameters:
        - fixed_image: SimpleITK.Image, reference image.
        - moving_image: SimpleITK.Image, image to register.
        - metric: str, similarity metric to use ("ncc", "mse", "mmi", "antsncc").
        - transform_type: str, type of transformation ("affine", "rigid", etc.).
        - fixed_mask: SimpleITK.Image or None, mask for fixed image.
        - moving_mask: SimpleITK.Image or None, mask for moving image.
        - initial_transform: SimpleITK.Transform or None, initial transformation.
        - center_images: bool, whether to center the images initially.
        - use_mask_to_center: bool, whether to use masks for centering.
        - centering_type: str, "geometry" or "moments" centering.
        - scaling_from_mask: bool, whether to scale based on volume ratio.
        - scaling_factor: float, factor to modulate the mask-based scaling.

        Returns:
        - Tuple[SimpleITK.Transform, SimpleITK.ImageRegistrationMethod], final transform and registration method object.
        """

        registration_method = sitk.ImageRegistrationMethod()

        # Set the metric
        if metric == "ncc":
            registration_method.SetMetricAsCorrelation()
        elif metric == "mse":
            registration_method.SetMetricAsMeanSquares()
        elif metric == "mmi":
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        elif metric == "antsncc":
            registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=3)
        else:
            raise ValueError(f"Invalid metric: {metric}.")

        # Set the transformation type (force the desired transformation type)
        if transform_type == "affine":
            transform = sitk.AffineTransform(fixed_image.GetDimension())
        elif transform_type == "rigid":
            transform = sitk.Euler3DTransform() if fixed_image.GetDimension() == 3 else sitk.Euler2DTransform()
        elif transform_type == "similarity":
            transform = sitk.Similarity3DTransform() if fixed_image.GetDimension() == 3 else sitk.Similarity2DTransform()
        elif transform_type == "scaling":
            transform = sitk.ScaleTransform(fixed_image.GetDimension())
        elif transform_type == "rigidwithscaling":
            transform = sitk.ScaleVersor3DTransform() if fixed_image.GetDimension() == 3 else sitk.ScaleVersor2DTransform()

        # Set the initial transform (from file or from centering)
        if initial_transform is not None:
            # Compose the initial transform with the new transform (apply the initial transform first)
            transform.SetMatrix(initial_transform.GetMatrix())  # Initialize the matrix from the initial transform
            transform.SetTranslation(initial_transform.GetTranslation())
            transform.SetCenter(initial_transform.GetCenter())

        elif center_images:
            center_transform = sitk.ScaleVersor3DTransform() if fixed_image.GetDimension() == 3 else sitk.ScaleVersor2DTransform()
            
            if use_mask_to_center:
                fixed_im_center=fixed_mask
                moving_im_center=moving_mask
            else:
                fixed_im_center=fixed_image
                moving_im_center=moving_image

            if fixed_im_center is None or moving_im_center is None:
                raise ValueError("Missing images for centering. Likely one of the masks was not provided")

            if centering_type == "geometry":
                center_mode = sitk.CenteredTransformInitializerFilter.GEOMETRY
            elif centering_type == "moments":
                center_mode = sitk.CenteredTransformInitializerFilter.MOMENTS
            else:
                raise ValueError(f"Invalid centering type: {centering_type}.")


            print("Computing centered transform initializer...")
            initial_transform_centering= sitk.CenteredTransformInitializer(fixed_im_center,
                                                                      moving_im_center,
                                                                      center_transform,  # Pure translation
                                                                      center_mode)

            initial_transform_centering = sitk.ScaleVersor3DTransform(initial_transform_centering) if fixed_image.GetDimension() == 3 else sitk.ScaleVersor2DTransform(initial_transform_centering)
            print("Done..")
            #Setting up initial transformation
            #transform.SetMatrix(sitk.Euler3DTransform(initial_transform_centering).GetMatrix())
            transform.SetCenter(initial_transform_centering.GetCenter())
            if isinstance(transform,sitk.ScaleTransform) == False:
                transform.SetTranslation(initial_transform_centering.GetTranslation())

            #Perform scale adjustment along the z axis to compensate.
            if scaling_from_mask:
                scale_z = self.scale_factor_from_mask(fixed_mask,moving_mask,factor=scaling_factor)
                print("Adjusting scale by factor {}".format(scale_z))
                if scale_z > scaling_th or scale_z < 1.0 / scaling_th:
                    print("Adjusting scale factor: value out of bounds. Setting value to 1.")
                    scale_z=1
                if isinstance(transform, sitk.ScaleVersor3DTransform):
                    transform.SetScale((1,1,scale_z))
                if isinstance(transform, sitk.AffineTransform):
                    initial_transform_centering.SetScale((1,1,scale_z))
                    transform.SetMatrix(initial_transform_centering.GetMatrix())

        registration_method.SetInitialTransform(transform, inPlace=False)

        # Apply the mask if provided
        if fixed_mask is not None:
            registration_method.SetMetricFixedMask(fixed_mask)

        if moving_mask is not None:
            registration_method.SetMetricMovingMask(moving_mask)

        # Set optimizer: Regular step gradient is lower without much improvement
        #registration_method.SetOptimizerAsGradientDescent(
        #    learningRate=self._lr, numberOfIterations=self._numIter, convergenceMinimumValue=1e-6, convergenceWindowSize=15
        #)

        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=self._lr, minStep=0.0001, numberOfIterations=self._numIter, gradientMagnitudeTolerance=1e-6,relaxationFactor=0.5)


        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Multi-resolution framework
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=self._shrinkFactors)
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=self._smoothingSigma)
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        #Adjust densities
        #fixed_image_adjusted,moving_image_adjusted=self.adjust_densities(fixed_image,moving_image,fixed_mask,moving_mask,sponge_model=False)
        fixed_image_adjusted=fixed_image
        moving_image_adjusted=moving_image

        # Execute the registration
        final_transform = registration_method.Execute(sitk.Cast(fixed_image_adjusted, sitk.sitkFloat32),
                                                      sitk.Cast(moving_image_adjusted, sitk.sitkFloat32))

        print(f"Final metric value: {registration_method.GetMetricValue()}")
        print(f"Optimizer's stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")

        # Check if the transform is actually a CompositeTransform
        if isinstance(final_transform, sitk.CompositeTransform):

            output_transform=final_transform.GetNthTransform(0)
        else:
            output_transform=final_transform

        return output_transform,registration_method


    def harden_composite_transformation(self, final_transform):
        """
        Convert a CompositeTransform into a single AffineTransform.

        Parameters:
        - final_transform: SimpleITK.CompositeTransform or Transform

        Returns:
        - SimpleITK.Transform
        """
        if isinstance(final_transform, sitk.CompositeTransform):
            print("Number of composite transformations {}.".format(final_transform.GetNumberOfTransforms()))
            if final_transform.GetNumberOfTransforms() > 1 :
                output_transform = sitk.AffineTransform(fixed_image.GetDimension())
                out_matrix = np.matrix(np.eye(fixed_image.GetDimension()))
                out_tt = np.array(np.zeros(fixed_image.GetDimension())) 

                for nn in range(final_transform.GetNumberOfTransforms()):
                    tt_tmp=final_transform.GetNthTransform(nn)

                    out_matrix=out_matrix @ np.matrix(tt_tmp.GetMatrix())
                    out_tt = out_tt + np.array(tt_tmp.GetTranslation())

                    output_transform.SetMatrix(out_matrix)
                    output_transform.SetTranslation(out_tt)

            else:
                output_transform=final_transform.GetNthTransform(0)

        else:
            output_transform = final_transform

        return output_transform


    def apply_transform(self,fixed_image, moving_image, transform):
        """
        Apply a transformation to the moving image.

        Parameters:
        - fixed_image: SimpleITK.Image, used as reference grid.
        - moving_image: SimpleITK.Image, image to resample.
        - transform: SimpleITK.Transform, transformation to apply.

        Returns:
        - SimpleITK.Image, resampled image.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)

        # Resample the moving image
        out_image = resampler.Execute(moving_image)
        return out_image



    def save_transform(self,transform, output_transform_path):
        """
        Save the transformation to disk.

        Parameters:
        - transform: SimpleITK.Transform, transformation object.
        - output_transform_path: str, file path to save the transform.
        """
        sitk.WriteTransform(transform, output_transform_path)
        print(f"Transformation saved to {output_transform_path}")


    def load_transform(self,transform_path,dimension=3):
        """
        Load a transformation from disk.

        Parameters:
        - transform_path: str, file path of the transform.
        - dimension: int, spatial dimension (2 or 3).

        Returns:
        - SimpleITK.Transform, loaded transformation.
        """
        # If it's a composite transform, you can get the list of transforms
        final_transform=sitk.ReadTransform(transform_path)
        if isinstance(final_transform, sitk.CompositeTransform):
            for i in range(final_transform.GetNumberOfTransforms()):
                print(f"Transform {i}: {final_transform.GetNthTransform(i)}")
        
        return final_transform

        # # If the transform is already an AffineTransform, just return it
        # if isinstance(final_transform, sitk.AffineTransform):
        #     return final_transform
        
        # # If it's a SimilarityTransform (which includes uniform scaling), we can treat it like Affine
        # if isinstance(final_transform, sitk.Similarity2DTransform) or isinstance(final_transform, sitk.Similarity3DTransform):
        #     affine_transform = sitk.AffineTransform(final_transform)
        #     return affine_transform
        
        # # For Euler (rigid body) or Translation transforms, convert to an affine transform
        # elif isinstance(final_transform, sitk.Euler2DTransform) or isinstance(final_transform, sitk.Euler3DTransform):
        #     # Create a new affine transform and copy the parameters
        #     affine_transform = sitk.AffineTransform(dimension)
        #     affine_transform.SetMatrix(final_transform.GetMatrix())  # Extract the matrix
        #     affine_transform.SetTranslation(final_transform.GetTranslation())
        #     affine_transform.SetCenter(final_transform.GetCenter())
        #     return affine_transform
        
        # else:
        #     raise TypeError(f"Unsupported transform type: {type(final_transform)}")


    def compute_metrics(self,reg_filter):
        """
        Compute multiple registration metrics from a registration method object.

        Parameters:
        - reg_filter: SimpleITK.ImageRegistrationMethod

        Returns:
        - dict: Dictionary with metric values (MSE, NCC, MMI).
        """
        metrics=dict()

        reg_filter.SetMetricAsMeanSquares()
        metrics['mse']=[reg_filter.GetMetricValue()]
        reg_filter.SetMetricAsCorrelation()
        metrics['ncc']=[reg_filter.GetMetricValue()]
        reg_filter.SetMetricAsMattesMutualInformation()
        metrics['mmi']=[reg_filter.GetMetricValue()]

        return metrics

    def pipeline_affine_lung_registration(self,fixed_image,moving_image,fixed_mask,moving_mask,metric='mmi'):
        """
        Execute a rigid + affine lung registration pipeline with predefined parameters.

        Parameters:
        - fixed_image: SimpleITK.Image
        - moving_image: SimpleITK.Image
        - fixed_mask: SimpleITK.Image
        - moving_mask: SimpleITK.Image
        - metric: str, similarity metric

        Returns:
        - Tuple[Transform, Transform, float], affine transform, rigid transform, and metric value
        """
        rigid_t,reg_filter=self.multiresolution_registration(fixed_image,moving_image,metric,'rigidwithscaling',fixed_mask,moving_mask,None,True,True,"moments",True,0.4)
        final_t,reg_filter=self.multiresolution_registration(fixed_image,moving_image,metric,'affine',fixed_mask,moving_mask,rigid_t)

        return final_t,rigid_t,reg_filter.GetMetricValue()

    def qc_image(self,fixed_image, moving_image, registered_image):
        """
        Generate QC image by concatenating orthogonal views of the fixed, moving, and registered images.

        Returns:
        - 2D numpy array for visualization or saving.
        """
        cor_view=list()
        axial_view=list()
        sag1_view = list()
        sag2_view = list()
        for im in [fixed_image,moving_image,registered_image]:
            im_np=sitk.GetArrayFromImage(im)
            #Cor
            slice_image = im_np[:, im_np.shape[1]//2, :]
            slice_image =np.flip(slice_image,axis=0)
            cor_view.append(slice_image)
            #Ax
            slice_image = im_np[im_np.shape[0]//2,: , :]
            axial_view.append(slice_image)
            #Sag views
            slice_image = im_np[:,:,im_np.shape[2]//4]
            slice_image =np.flip(slice_image,axis=0)
            sag1_view.append(slice_image)
            slice_image = im_np[:,:,3*im_np.shape[2]//4]
            slice_image =np.flip(slice_image,axis=0)
            sag2_view.append(slice_image)

        cor_row=self.pad_concatenate(cor_view,axis=1,pad_value=-1024)
        axial_row=self.pad_concatenate(axial_view,axis=1,pad_value=-1024)
        sag1_row=self.pad_concatenate(sag1_view,axis=1,pad_value=-1024)
        sag2_row=self.pad_concatenate(sag2_view,axis=1,pad_value=-1024)

        image=self.pad_concatenate([cor_row,axial_row,sag1_row,sag2_row],axis=0,pad_value=-1024)

        return image

    def window_image(self, ct_array,window_center=-500,window_width=1000):
        """
        Apply window/level settings to a CT image.

        Parameters:
        - ct_array: ndarray, raw CT data
        - window_center: int
        - window_width: int

        Returns:
        - ndarray (uint8): image windowed and scaled to 0-255
        """
        lower_bound = window_center - window_width // 2
        upper_bound = window_center + window_width // 2
        windowed_image = np.clip(ct_array, lower_bound, upper_bound)
        
        # Normalize to 0-255 for PNG
        windowed_image = (windowed_image - lower_bound) / (upper_bound - lower_bound) * 255.0
        windowed_image = windowed_image.astype(np.uint8)
        
        return windowed_image

    def pad_concatenate(self,arrays,axis=0, pad_value=0):
        """
        Concatenate 2D arrays with padding along the specified axis.

        Parameters:
        - arrays: list of 2D numpy arrays
        - axis: int (0 or 1)
        - pad_value: int, fill value for padding

        Returns:
        - numpy array, padded concatenated output
        """

        # Check that all arrays are 2D
        for arr in arrays:
            if arr.ndim != 2:
                raise ValueError("All input arrays must be 2D.")

        if axis == 0:
            # Concatenating along rows, so pad along columns
            max_columns = max(arr.shape[1] for arr in arrays)  # Find the max number of columns
            padded_arrays = [np.pad(arr, ((0, 0), (0, max_columns - arr.shape[1])), constant_values=pad_value)
                             for arr in arrays]
            return np.concatenate(padded_arrays, axis=0)

        elif axis == 1:
            # Concatenating along columns, so pad along rows
            max_rows = max(arr.shape[0] for arr in arrays)  # Find the max number of rows
            padded_arrays = [np.pad(arr, ((0, max_rows - arr.shape[0]), (0, 0)), constant_values=pad_value)
                             for arr in arrays]
            return np.concatenate(padded_arrays, axis=1)

        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")

    def ssmi_metric(self,fixed_image,registered_image,fixed_mask=None):
        """
        Compute mean SSIM (structural similarity) between fixed and registered image using representative slices.

        Parameters:
        - fixed_image: SimpleITK.Image
        - registered_image: SimpleITK.Image
        - fixed_mask: Optional[SimpleITK.Image]

        Returns:
        - float: mean SSIM value across views
        """
        cor_view=list()
        axial_view=list()
        sag1_view = list()
        sag2_view = list()
        for im in [fixed_image,registered_image,fixed_mask]:
            if im is not None:
                im_np=sitk.GetArrayFromImage(im)
                #Cor
                slice_image = im_np[:, im_np.shape[1]//2, :]
                slice_image =np.flip(slice_image,axis=0)
                cor_view.append(slice_image)
                #Ax
                slice_image = im_np[im_np.shape[0]//2,: , :]
                axial_view.append(slice_image)
                #Sag views
                slice_image = im_np[:,:,im_np.shape[2]//4]
                slice_image =np.flip(slice_image,axis=0)
                sag1_view.append(slice_image)
                slice_image = im_np[:,:,3*im_np.shape[2]//4]
                slice_image =np.flip(slice_image,axis=0)
                sag2_view.append(slice_image)

        ssim_values=list()
        for view in [cor_view,axial_view,sag1_view,sag2_view]:
            if fixed_mask is not None:

                # Apply mask to both slices
                slice1 = view[0][view[2]>0]
                slice2 = view[1][view[2]>0]
            else:
                slice1 = view[0]
                slice2 = view[1]


            ssim_value = ssim(slice1, slice2, data_range=slice2.max() - slice2.min())
            ssim_values.append(ssim_value)
        
        # Filter out None values (slices with no masked region) and calculate mean SSIM
        valid_ssim_values = [val for val in ssim_values if val is not None]
        mean_ssim = np.mean(valid_ssim_values) if valid_ssim_values else None
    
        return mean_ssim

def volumechange_verification(fixed_mask,moving_mask,transform):

    f_np=sitk.GetArrayFromImage(fixed_mask)
    m_np=sitk.GetArrayFromImage(moving_mask)

    vol_m=np.sum(m_np>0)*np.prod(moving_mask.GetSpacing())/1000.0
    vol_f=np.sum(f_np>0)*np.prod(fixed_mask.GetSpacing())/1000.0
    vol_ratio=vol_m/vol_f

    mm=transform.GetMatrix()

    scaling_f=np.prod(np.array([mm[0],mm[4],mm[8]]))

    det=np.linalg.det(np.reshape(transform.GetMatrix(),[3,3]))

    return vol_f,vol_m,vol_ratio,det

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image registration with different transformation types and metrics")
    parser.add_argument("--fixed", required=True, help="Path to the fixed (reference) image")
    parser.add_argument("--moving", required=True, help="Path to the moving image")
    parser.add_argument("--output_transform", required=False, help="Path to save the resulting transformation")
    parser.add_argument("--mask_fixed", required=False, help="Path to the fixed image mask (optional)")
    parser.add_argument("--mask_moving", required=False, help="Path to the moving image mask (optional)")
    parser.add_argument("--transform_type", required=False, choices=["affine", "rigid", "similarity","scaling","rigidwithscaling"], default="affine",
                        help="Transformation type to apply during registration")
    parser.add_argument("--metric", required=False, default="ncc", choices=["ncc", "mse", "mmi", "antsncc"],
                        help="Similarity metric to use: ncc (Normalized Cross Correlation), mse (Mean Squares), mmi (Matte's Mutual Information) or antsncc (ANTS Neighborhood Correlation)")
    parser.add_argument("--centering", dest="center_images", action='store_true', required=False, help="Center the images before processing.")
    parser.add_argument("--use_mask_centering", dest="use_mask_to_center", action='store_true', required=False, help="Use mask images for centering.")
    parser.add_argument("--centering_type", dest="centering_type", required=False, default="geometry", choices=["geometry","moments"])
    parser.add_argument("--scaling_from_mask", dest="scaling_from_mask", required=False,action='store_true', help="Set an initial scaling factor along the z coordinate based on the mask volume ratio (optional).")
    parser.add_argument("--scaling_factor", dest="scaling_factor", required=False, default=0.4, help="Scale factor to apply to the mask volume ratio to define initial scaling (optional).")
    parser.add_argument("--initial_transform", required=False, 
        help="Optional: Path to an initial transformation file. If provided, the optimization will start from this transform, and the final output transform will be relative to the original space (optional)")
    parser.add_argument("--qc_image", required=False, help="Path to qc image for registration result (optional)")
    parser.add_argument("--report", required=False, help="Path to a csv file containing the performnace metrics of the registration (optional)")
    parser.add_argument("--cid_fixed", required=False, help="CID of fixed image to save with performance metrics (optional).", default=None)
    parser.add_argument("--cid_moving", required=False, help="CID of moving image to save with performance metrics (optional).", default=None)
    parser.add_argument("--inverse", required=False, help="Perform registration in the inverse direction (i.e fixed image is moving and the other way around). However, deformation is provided as moving to fix)", action='store_true')
    parser.add_argument("--lungpipeline", required=False, help="Perform affine lung registration based on the predefined pipeline optimized for our worked. It requires lung masks for both fixed and moving images)", action='store_true')
    parser.add_argument("--prefix_output_transform", required=False, help="Prefix name for the output transformation when using the lung pipeline option")

    args = parser.parse_args()

    # Load images
    fixed_image = sitk.ReadImage(args.fixed, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(args.moving, sitk.sitkFloat32)

    # Load mask if provided
    fixed_mask = sitk.ReadImage(args.mask_fixed, sitk.sitkUInt16) if args.mask_fixed else None
    moving_mask = sitk.ReadImage(args.mask_moving, sitk.sitkUInt16) if args.mask_moving else None

    reg_engine=AffineRegistration()

    # Load initial transformation if provided
    initial_transform = reg_engine.load_transform(args.initial_transform,fixed_image.GetDimension()) if args.initial_transform else None

    if args.lungpipeline:
        if fixed_mask is None or moving_mask is None:
            #Raised error because lung pipeline needs lung masks for registration
            raise ValueError("Lung pipeline requires both fixed and moving lung masks for registration.")
        else:
            final_transform,rigid_transform,reg_filter=reg_engine.pipeline_affine_lung_registration(fixed_image,moving_image,fixed_mask,moving_mask)
    else:
        # Perform registration
        if args.inverse:

            if initial_transform is not None:
                tmp_tt = sitk.AffineTransform(fixed_image.GetDimension())
                tmp_tt.SetMatrix(initial_transform.GetMatrix())
                tmp_tt.SetCenter(initial_transform.GetCenter())
                tmp_tt.SetTranslation(initial_transform.GetTranslation())
                initial_transform=tmp_tt.GetInverse()

            final_transform,reg_filter = reg_engine.multiresolution_registration(moving_image,fixed_image, metric=args.metric,
                                                           transform_type=args.transform_type, fixed_mask=moving_mask, moving_mask=fixed_mask,
                                                           initial_transform=initial_transform,center_images=args.center_images,use_mask_to_center=args.use_mask_to_center,centering_type=args.centering_type,
                                                           scaling_from_mask=args.scaling_from_mask,scaling_factor=args.scaling_factor)
            if isinstance(final_transform,sitk.ScaleVersor3DTransform):
                #Cast to a AffineTransform to compute the inverse
                transform=sitk.AffineTransform(fixed_image.GetDimension())
                transform.SetMatrix(final_transform.GetMatrix())  # Initialize the matrix from the initial transform
                transform.SetTranslation(final_transform.GetTranslation())
                transform.SetCenter(final_transform.GetCenter())
                final_transform=transform.GetInverse()
            else:
                final_transform=final_transform.GetInverse()

        else:

            final_transform,reg_filter = reg_engine.multiresolution_registration(fixed_image, moving_image, metric=args.metric,
                                                       transform_type=args.transform_type, fixed_mask=fixed_mask, moving_mask=moving_mask,
                                                       initial_transform=initial_transform,center_images=args.center_images,use_mask_to_center=args.use_mask_to_center,centering_type=args.centering_type,
                                                       scaling_from_mask=args.scaling_from_mask,scaling_factor=args.scaling_factor)

    # Apply transformation
    #registered_image = apply_transform(fixed_image, moving_image, final_transform)

    # Display images
    #show_images(fixed_image, moving_image, registered_image)

    # Save the transformation if output path is provided
    if args.lungpipeline:
        prefix_file=args.prefix_output_transform
        reg_engine.save_transform(final_transform, prefix_file+"_0GenericAffine.tfm")
        reg_engine.save_transform(rigid_transform, prefix_file+"_0GenericRigidWithScaling.tfm")
    else:
        if args.output_transform:
            reg_engine.save_transform(final_transform, args.output_transform)

    if args.qc_image:
        registered_image = reg_engine.apply_transform(fixed_image, moving_image, final_transform)
        qc_array=reg_engine.qc_image(fixed_image, moving_image, registered_image)
        qc_array=reg_engine.window_image(qc_array,window_center=-500,window_width=1000)
        pil_image = Image.fromarray(qc_array)
        # Save the image as PNG
        pil_image.save(args.qc_image, format='PNG')

    if args.report:
        registered_image = reg_engine.apply_transform(fixed_image, moving_image, final_transform)
        metric_dict=dict()
        metric_dict['cid-fixed']=[args.cid_fixed]
        metric_dict['cid-moving']=[args.cid_moving]
        metric_dict['transform']=[args.transform_type]
        metric_dict['metric']=[args.metric]
        metric_dict['metric_val']=[reg_filter.GetMetricValue()]
        metric_dict['ssim']=[reg_engine.ssmi_metric(fixed_image,registered_image)]
        if fixed_mask is not None and moving_mask is not None:
            vol_f,vol_m,vol_ratio,det=volumechange_verification(fixed_mask,moving_mask,final_transform)
            metric_dict['vol-fixed']=[vol_f]
            metric_dict['vol-moving']=[vol_m]
            metric_dict['vol-ratio']=[vol_ratio]
            metric_dict['determinant']=[det]
            metric_dict['ssim-masked']=[reg_engine.ssmi_metric(fixed_image,registered_image,fixed_mask)]
        pd.DataFrame(metric_dict).to_csv(args.report,index=False)


