"""
Extracting TACs from masks or regions, getting statistics, and writing to file.
"""
import ants
import numpy as np

from ..utils.useful_functions import check_physical_space_for_ants_image_pair


def extract_roi_voxel_tacs_from_image_using_mask(input_image: ants.core.ANTsImage,
                                                 mask_image: ants.core.ANTsImage,
                                                 verbose: bool = False) -> np.ndarray:
    """
    Function to extract ROI voxel tacs from an image using a mask image.

    This function returns all the voxel TACs, and unlike
    :func:`extract_mean_roi_tac_from_nifti_using_segmentation` does not calculate the mean over
    all the voxels.

    Args:
        input_image (ants.core.ANTsImage): Input 4D-image from which to extract ROI voxel tacs.
        mask_image (ants.core.ANTsImage): Mask image which determines which voxels to extract.
        verbose (bool, optional): If True, prints information about the shape of extracted voxel
            tacs.

    Returns:
        out_voxels (np.ndarray): Array of voxel TACs of shape (num_voxels, num_frames)

    Raises:
         AssertionError: If input image is not 4D-image.
         AssertionError: If mask image is not in the same physical space as the input image.

    Example:

        .. code-block:: python

            import ants
            import numpy as np

            from petpal.preproc import regional_tac_extraction
            tac_func = regional_tac_extraction.extract_roi_voxel_tacs_from_image_using_mask
            
            # Read images
            pet_img = ants.image_read("/path/to/pet.nii.gz")
            masked_region_img = ants.image_read("/path/to/mask_region.nii.gz")

            # Run ROI extraction and save
            time_series = tac_func(input_image=pet_img, mask_image=masked_region_img).T
            np.savetxt("time_series.tsv", time_series, delimiter='\t')
            
    """
    assert len(input_image.shape) == 4, "Input image must be 4D."
    assert check_physical_space_for_ants_image_pair(input_image, mask_image), (
        "Images must have the same physical dimensions.")

    out_voxels = apply_mask_4d(input_arr=input_image.numpy(),
                               mask_arr=mask_image.numpy(),
                               verbose=verbose)
    return out_voxels


def apply_mask_4d(input_arr: np.ndarray,
                  mask_arr: np.ndarray,
                  verbose: bool = False) -> np.ndarray:
    """
    Function to extract ROI voxel tacs from an array using a mask array.

    This function applies a 3D mask to a 4D image, returning the time series for each voxel in a
    single flattened numpy array.

    Args:
        input_arr (np.ndarray): Input 4D-image from which to extract ROI voxel tacs.
        mask_arr (np.ndarray): Mask image which determines which voxels to extract.
        verbose (bool, optional): If True, prints information about the shape of extracted voxel
            tacs.

    Returns:
        out_voxels (np.ndarray): Time series of each voxel in the mask, as a flattened numpy array.

    Raises:
         AssertionError: If input array is not 4D.
         AssertionError: If input and mask array shapes are mismatched.

    Example:

        .. code-block:: python

            import ants
            import numpy as np

            from petpal.preproc.regional_tac_extraction import apply_mask_4d
            
            # Read images
            pet_img = ants.image_read("/path/to/pet.nii.gz")
            masked_region_img = ants.image_read("/path/to/mask_region.nii.gz")

            # Get underlying arrays
            pet_arr = pet_img.numpy()
            masked_region_arr = masked_region_img.numpy()

            # Run ROI extraction and save
            time_series = apply_mask_4d(input_arr=pet_arr, mask_arr=masked_region_arr).T
            np.savetxt("time_series.tsv", time_series, delimiter='\t')

    """
    assert len(input_arr.shape) == 4, "Input array must be 4D."
    assert input_arr.shape[:3] == mask_arr.shape, (
            "Array must have the same physical dimensions.")

    x_inds, y_inds, z_inds = mask_arr.nonzero()
    out_voxels = input_arr[x_inds, y_inds, z_inds, :]
    if verbose:
        print(f"(ImageOps): Output TACs have shape {out_voxels.shape}")
    return out_voxels
