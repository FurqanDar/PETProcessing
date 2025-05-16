"""
Extracting TACs from masks or regions, getting statistics, and writing to file.
"""
import os
import ants
import numpy as np

from ..preproc.image_operations_4d import extract_mean_roi_tac_from_nifti_using_segmentation
from ..utils import image_io
from ..utils.scan_timing import ScanTimingInfo
from ..utils.time_activity_curve import TimeActivityCurve
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


def write_tacs(input_image_path: str,
               label_map_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               verbose: bool = False,
               out_tac_prefix: str = ''):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and label map. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.

    Args:
        input_image_path (str): Path to the 4D PET image from which regional TACs will be
            extracted.
        label_map_path (str): Path to the dseg file linking regions to their mappings in the
            segmentation image.
        segmentation_image_path (str): Path to the segmentation image containing ROIs. Must be in
            the same space as input_image.
        out_tac_dir (str): Path to the directory where regional TACs will be written to.
        verbose (bool): If true, outputs information during processing. Default False.
        out_tac_prefix (str): Prefix for output TAC files. Typically the participant ID.


    Examples:

        .. code-block:: python

            from petpal.preproc.regional_tac_extraction import write_tacs

            # get files
            segmentation_path = '/path/to/aparc+aseg.nii.gz'
            pet_path = '/path/to/pet.nii.gz'

            # run write_tacs
            write_tacs(input_image_path=pet_path,
                       label_map_path='dseg.tsv',
                       segmentation_image_path=segmentation_path,
                       out_tac_dir='/path/to/output/',
                       verbose=False)

    """
    label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
    regions_abrev = label_map['abbreviation']
    regions_map = label_map['mapping']

    tac_extraction_func = extract_mean_roi_tac_from_nifti_using_segmentation
    pet_numpy = ants.image_read(input_image_path).numpy()
    seg_numpy = ants.image_read(segmentation_image_path).numpy()

    scan_timing_info = ScanTimingInfo.from_nifti(image_path=input_image_path)
    tac_times_in_mins = scan_timing_info.center_in_mins

    for i, region in enumerate(regions_map):
        extracted_tac = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                            segmentation_image_numpy=seg_numpy,
                                            region=int(region),
                                            verbose=verbose,
                                            with_std=True)

        region_tac = TimeActivityCurve(times=tac_times_in_mins,
                                       activity=extracted_tac[0],
                                       uncertainty=extracted_tac[1])
        if out_tac_prefix:
            out_tac_path = os.path.join(out_tac_dir,
                                        f'{out_tac_prefix}_seg-{regions_abrev[i]}_tac.tsv')
        else:
            out_tac_path = os.path.join(out_tac_dir, f'seg-{regions_abrev[i]}_tac.tsv')
        region_tac.to_tsv(filename=out_tac_path)
