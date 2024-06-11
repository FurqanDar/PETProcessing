"""
Methods applying to segmentations.

Available methods:
* :meth:`region_blend`: Merge regions in a segmentation image into a mask with value 1
* :meth:`resample_segmentation`: Resample a segmentation image to the affine of a 4D PET image.
* :meth:`vat_wm_ref_region`: Compute the white matter reference region for the VAT radiotracer.

TODO:

"""
import numpy as np
import nibabel
from nibabel import processing
from . import image_operations_4d
from . import math_lib


def region_blend(segmentation_numpy: np.ndarray,
                 regions_list: list):
    """
    Takes a list of regions and a segmentation, and returns a mask with only the listed regions.
    """
    regions_blend = np.zeros(segmentation_numpy.shape)
    for region in regions_list:
        region_mask = segmentation_numpy == region
        region_mask_int = region_mask.astype(int)
        regions_blend += region_mask_int
    return regions_blend


def segmentations_merge(segmentation_primary: np.ndarray,
                        segmentation_secondary: np.ndarray,
                        regions_to_reassign: list) -> np.ndarray:
    """
    Merge segmentations by assigning regions to a primary segmentation image from a secondary
    segmentation. Region indices are pulled from the secondary into the primary from a list.

    Primary and secondary segmentations must have the same shape and orientation.
    """
    for region in regions_to_reassign:
        region_mask = np.where(segmentation_secondary==region)
        segmentation_primary[region_mask] = region
    return segmentation_primary


def binarize(segmentation_numpy: np.ndarray,
             out_val: float=1):
    """
    Set all non-zero values to a given output, typically 1.
    """
    nonzero_voxels = np.where(segmentation_numpy!=0)
    bin_mask = np.zeros(segmentation_numpy.shape)
    bin_mask[nonzero_voxels] = out_val
    return bin_mask


def parcellate_right_left(segmentation_numpy: np.ndarray,
                          region: int,
                          new_right_region: int,
                          new_left_region: int) -> np.ndarray:
    """
    Divide a region within a segmentation image into right and left values.
    Assumes left and right sides are neatly subdivided by the image midplane,
    with right values below the mean value of the x-axis (zeroth axis) and left
    values above the mean value of the x-axis (zeroth axis).

    Intended to work with FreeSurfer segmentations on images loaded with
    nibabel. Use outside of these assumptions at your own risk.

    Args:
        segmentation_numpy (np.ndarray): Segmentation image array loaded with Nibabel, RAS+ orientation
        region (int): Region index in segmentation image to be split into left and right.
        new_right_region (int): Region on the right side assigned to previous region.
        new_left_region (int): Region on the left side assined to previous region.

    Returns:
        split_segmentation (np.ndarray): Original segmentation image array with new left and right values.
    """
    seg_shape = segmentation_numpy.shape
    x_mid = (seg_shape[0] - 1) // 2

    seg_region = np.where(segmentation_numpy==region)
    right_region = seg_region[0] <= x_mid
    seg_region_right = tuple((seg_region[0][right_region],
                              seg_region[1][right_region],
                              seg_region[2][right_region]))

    left_region = seg_region[0] > x_mid
    seg_region_left = tuple((seg_region[0][left_region],
                             seg_region[1][left_region],
                             seg_region[2][left_region]))
    
    split_segmentation = segmentation_numpy
    split_segmentation[seg_region_right] = new_right_region
    split_segmentation[seg_region_left] = new_left_region

    return split_segmentation


def resample_segmentation(input_image_4d_path: str,
                          segmentation_image_path: str,
                          out_seg_path: str,
                          verbose: bool):
    """
    Resamples a segmentation image to the resolution of a 4D PET series image. Takes the affine 
    information stored in the PET image, and the shape of the image frame data, as well as the 
    segmentation image, and applies NiBabel's ``resample_from_to`` to resample the segmentation to
    the resolution of the PET image. This is used for extracting TACs from PET imaging where the 
    PET and ROI data are registered to the same space, but have different resolutions.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image, registered to anatomical space, to which the segmentation file is resampled.
        segmentation_image_path (str): Path to a .nii or .nii.gz file containing a 3D segmentation
            image, where integer indices label specific regions.
        out_seg_path (str): Path to a .nii or .nii.gz file to which the resampled segmentation
            image is written.
        verbose (bool): Set to ``True`` to output processing information.
    """
    pet_image = nibabel.load(input_image_4d_path)
    seg_image = nibabel.load(segmentation_image_path)
    pet_series = pet_image.get_fdata()
    image_first_frame = pet_series[:, :, :, 0]
    seg_resampled = processing.resample_from_to(from_img=seg_image,
                                                to_vox_map=(image_first_frame.shape, pet_image.affine),
                                                order=0)
    nibabel.save(seg_resampled, out_seg_path)
    if verbose:
        print(f'Resampled segmentation saved to {out_seg_path}')


def vat_wm_ref_region(input_segmentation_path: str,
                      out_segmentation_path: str):
    """
    Generates the cortical white matter reference region described in O'Donnell
    JL et al. (2024) PET Quantification of [18F]VAT in Human Brain and Its 
    Test-Retest Reproducibility and Age Dependence. J Nucl Med. 2024 Jun 
    3;65(6):956-961. doi: 10.2967/jnumed.123.266860. PMID: 38604762; PMCID:
    PMC11149597. Requires FreeSurfer segmentation with original label mappings.

    Args:
        input_segmentation_path (str): Path to segmentation on which white
            matter reference region is computed.
        out_segmentation_path (str): Path to which white matter reference
            region mask image is saved.
    """
    wm_regions = [2,41,251,252,253,254,255,77,3000,3001,3002,3003,3004,3005,
                  3006,3007,3008,3009,3010,3011,3012,3013,3014,3015,3016,3017,
                  3018,3019,3020,3021,3022,3023,3024,3025,3026,3027,3018,3029,
                  3030,3031,3032,3033,3034,3035,4000,4001,4002,4003,4004,4005,
                  4006,4007,4008,4009,4010,4011,4012,4013,4014,4015,4016,4017,
                  4018,4019,4020,4021,4022,4023,4024,4025,4026,4027,4028,4029,
                  4030,4031,4032,4033,4034,4035,5001,5002]
    csf_regions = [4,14,15,43,24]

    segmentation = nibabel.load(input_segmentation_path)
    seg_image = segmentation.get_fdata()
    seg_resolution = segmentation.header.get_zooms()

    wm_merged = region_blend(segmentation_numpy=seg_image,
                                                 regions_list=wm_regions)
    csf_merged = region_blend(segmentation_numpy=seg_image,
                                                  regions_list=csf_regions)
    wm_csf_merged = wm_merged + csf_merged

    wm_csf_blurred = math_lib.gauss_blur_computation(input_image=wm_csf_merged,
                                                     blur_size_mm=9,
                                                     input_zooms=seg_resolution,
                                                     verbose=True,
                                                     use_FWHM=True)
    
    wm_csf_eroded = image_operations_4d.threshold(input_image_numpy=wm_csf_blurred,
                                                  lower_bound=0.95)
    wm_csf_eroded_keep = np.where(wm_csf_eroded>0)
    wm_csf_eroded_mask = np.zeros(wm_csf_eroded.shape)
    wm_csf_eroded_mask[wm_csf_eroded_keep] = 1

    wm_erode = wm_csf_eroded_mask * wm_merged

    wm_erode_save = nibabel.nifti1.Nifti1Image(dataobj=wm_erode,
                                               affine=segmentation.affine,
                                               header=segmentation.header)
    nibabel.save(img=wm_erode_save,
                 filename=out_segmentation_path)
