import SimpleITK as sitk
import numpy as np
from PIL import Image  # Import Pillow
import os
import argparse


class ExtractQCImageFromRegion():
    def __init__(self, ct_image, label_map, window_center=-700, window_width=1000, extend_by=0,plane="coronal"):
        """
        Initialize the CTCoronalExtractor class with paths and parameters.
        """
        self.ct_image = ct_image
        self.label_map = label_map
        self.window_center = window_center
        self.window_width = window_width
        self.extend_by = extend_by
        self.plane = plane.lower()

    def extract_bounding_box(self,label_id):
        """
        Extract the bounding box of the segmented region and optionally extend it.
        """
        label_map_np = sitk.GetArrayFromImage(self.label_map)
        non_zero_indices = np.nonzero(label_map_np==label_id)
        
        min_z, max_z = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
        min_y, max_y = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_x, max_x = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])
        
        # Extend bounding box if requested
        min_x = max(min_x - self.extend_by, 0)
        max_x = min(max_x + self.extend_by, label_map_np.shape[2] - 1)
        min_y = max(min_y - self.extend_by, 0)
        max_y = min(max_y + self.extend_by, label_map_np.shape[1] - 1)
        min_z = max(min_z - self.extend_by, 0)
        max_z = min(max_z + self.extend_by, label_map_np.shape[0] - 1)
        
        return (min_x, max_x, min_y, max_y, min_z, max_z)

    def window_image(self, ct_array):
        """
        Apply window level and width to a CT array.
        """
        lower_bound = self.window_center - self.window_width // 2
        upper_bound = self.window_center + self.window_width // 2
        windowed_image = np.clip(ct_array, lower_bound, upper_bound)
        
        # Normalize to 0-255 for PNG
        windowed_image = (windowed_image - lower_bound) / (upper_bound - lower_bound) * 255.0
        windowed_image = windowed_image.astype(np.uint8)
        
        return windowed_image

    def extract_slice(self, slice_idx,plane):
        """
        Extract a slice based on the specified plane, apply windowing, and save as PNG.
        """
        # Get numpy array from CT image
        ct_array = sitk.GetArrayFromImage(self.ct_image)
        
        # Select the appropriate slice based on the plane
        if plane == "coronal":
            slice_image = ct_array[:, slice_idx, :]
            slice_image =np.flip(slice_image,axis=0)
        elif plane == "axial":
            slice_image = ct_array[slice_idx, :, :]
        elif plane == "sagittal":
            slice_image = ct_array[:, :, slice_idx]
            slice_image =np.flip(slice_image,axis=0)

        else:
            raise ValueError(f"Invalid plane: {self.plane}. Choose from 'coronal', 'axial', or 'sagittal'.")
        
        return slice_image

    def save_slice(self, slice_image,output_path):
        # Apply windowing
        windowed_slice = self.window_image(slice_image)
        
        # Save as PNG
        #plt.imsave(output_path, windowed_slice, cmap='gray')

        # Convert the windowed image (NumPy array) to a PIL image
        pil_image = Image.fromarray(windowed_slice)
        
        # Save the image as PNG
        pil_image.save(output_path, format='PNG')

    def pad_concatenate(self,arrays,axis=0, pad_value=0):
        """
        Concatenate N 2D arrays along the specified axis and pad the other dimension.

        Parameters:
        - arrays: List of 2D NumPy arrays to concatenate.
        - axis: Axis along which to concatenate (0 for rows, 1 for columns).
        - pad_value: The value to use for padding the arrays (default: 0).

        Returns:
        - Concatenated and padded array.
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

    def process(self, label_id,output_path):
        """
        Main process to extract bounding box, select a slice based on the plane, and save as PNG.
        """
        # Extract the bounding box with optional extension
        min_x, max_x, min_y, max_y, min_z, max_z = self.extract_bounding_box(label_id)
        
        # Select a slice index based on the plane
        if self.plane == "coronal":
            slice_idx = (min_y + max_y) // 2  # Midpoint in the coronal axis
            slice_image=self.extract_slice(slice_idx,self.plane)
        elif self.plane == "axial":
            slice_idx = (min_z + max_z) // 2  # Midpoint in the axial axis
            slice_image=self.extract_slice(slice_idx,self.plane)
        elif self.plane == "sagittal":
            slice_idx = (min_x + max_x) // 2  # Midpoint in the sagittal axis
            slice_image=self.extract_slice(slice_idx,self.plane)
        elif self.plane == "all":
            slices=list()
            slice_idx = (min_z + max_z) // 2  # Midpoint in the axial axis
            slices.append(self.extract_slice(slice_idx,"axial"))
            slice_idx = (min_y + max_y) // 2  # Midpoint in the coronal axis
            slices.append(self.extract_slice(slice_idx,"coronal"))
            slice_idx = (min_x + max_x) // 2  # Midpoint in the sagittal axis
            slices.append(self.extract_slice(slice_idx,"sagittal"))
            slice_image=self.pad_concatenate(slices,axis=1,pad_value=-1024)
        else:
            raise ValueError(f"Invalid plane: {self.plane}. Choose from 'coronal', 'axial', or 'sagittal'.")

        self.save_slice(slice_image,output_path)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Extract a coronal slice from a CT volume and save as PNG with specified window level.")
    
    # Define command-line arguments
    parser.add_argument("--ct", dest="ct_image_path", type=str, help="Path to the CT image.")
    parser.add_argument("--lm", dest="lm_path", type=str, help="Path to the segmentation label map.")
    parser.add_argument("-o", dest="output_path", type=str, help="Path to save the output PNG file.")
    parser.add_argument("-l", dest="label_id", type=int, help="Label id to used to defined bounding box. Axial, sag, and/or coronal images will be extracted from the mid point of the bounding box.")
    parser.add_argument("--window_center", type=int, default=-700, help="Window center for HU (default: -700).")
    parser.add_argument("--window_width", type=int, default=1000, help="Window width for HU (default: 1000).")
    parser.add_argument("--extend_bbox", type=int, default=0, help="Number of voxels to extend the bounding box in all directions (default: 0).")
    parser.add_argument("--plane", type=str, default="all", choices=["coronal", "axial", "sagittal", "all"], help="Plane to extract the slice from (default: all).")

    # Parse the arguments
    args = parser.parse_args()

    # Read the CT image and segmentation label map
    ct_image = sitk.ReadImage(args.ct_image_path)
    label_map = sitk.ReadImage(args.lm_path)
    
    # Initialize the extractor class
    extractor = ExtractQCImageFromRegion(
        ct_image=ct_image,
        label_map=label_map,
        window_center=args.window_center,
        window_width=args.window_width,
        extend_by=args.extend_bbox,
        plane=args.plane
    )
    
    # Run the process
    extractor.process(label_id=args.label_id,output_path=args.output_path)
