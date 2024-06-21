"""
Methods applying to segmentations.

Available methods:
* :meth:`region_blend`: Merge regions in a segmentation image into a mask with value 1
* :meth:`resample_segmentation`: Resample a segmentation image to the affine of a 4D PET image.
* :meth:`vat_wm_ref_region`: Compute the white matter reference region for the VAT radiotracer.

TODO:
 * Find a more efficient way to find the region mask in :meth:`segmentations_merge` that works for region=1

"""
import numpy as np
import nibabel
from nibabel import processing
from . import image_operations_4d
from ..utils import math_lib


def region_blend(segmentation_numpy: np.ndarray,
                 regions_list: list):
    """
    Takes a list of regions and a segmentation, and returns a mask with only the listed regions.

    Args:
        segmentation_numpy (np.ndarray): Segmentation image data array
        regions_list (list): List of regions to include in the mask

    Returns:
        regions_blend (np.ndarray): Mask array with value one where
            segmentation values are in the list of regions provided, and zero
            elsewhere.
    """
    regions_blend = np.zeros(segmentation_numpy.shape)
    for region in regions_list:
        region_mask = segmentation_numpy == region
        region_mask_int = region_mask.astype(int)
        regions_blend += region_mask_int
    return regions_blend


def segmentations_merge(segmentation_primary: np.ndarray,
                        segmentation_secondary: np.ndarray,
                        regions: list) -> np.ndarray:
    """
    Merge segmentations by assigning regions to a primary segmentation image from a secondary
    segmentation. Region indices are pulled from the secondary into the primary from a list.

    Primary and secondary segmentations must have the same shape and orientation.

    Args:
        segmentation_primary (np.ndarray): The main segmentation to which new
            regions will be added.
        segmentation_secondary (np.ndarray): Distinct segmentation with regions
            to add to the primary.
        regions (list): List of regions to pull from the secondary to add to
            the primary.
    
    Returns:
        segmentation_primary (np.ndarray): The input segmentation with new
            regions added.
    """
    for region in regions:
        region_mask = (segmentation_secondary > region - 0.1) & (segmentation_secondary < region + 0.1)
        segmentation_primary[region_mask] = region
    return segmentation_primary


def binarize(input_image_numpy: np.ndarray,
             out_val: float=1):
    """
    Convert a segmentation image array into a mask by setting nonzero values
    to a uniform output value, typically one.

    Args:
        input_image_numpy (np.ndarray): Input image to be binarized to zero and
            another value.
        out_val (float): Uniform value output image is set to.
    
    Returns:
        bin_mask (np.ndarray): Image array of same shape as input, with values
            only zero and ``out_val``.
    """
    nonzero_voxels = (input_image_numpy > 1e-37) & (input_image_numpy < -1e-37)
    bin_mask = np.zeros(input_image_numpy.shape)
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


def replace_probabilistic_region(segmentation_numpy: np.ndarray,
                                 segmentation_zooms: list,
                                 blur_size_mm: float,
                                 regions: list,
                                 regions_to_replace: list):
    """
    Runs a correction on a segmentation by replacing a list of regions with
    a set of nearby regions. This is accomplished by creating masks of the
    nearby regions, blurring them to create probabilistic segmentation maps,
    finding the highest probability nearby region in the region to replace,
    and replacing values with the respective nearby region.

    This is useful for protocols where there residual regions not intended to
    be carried forward after generating new regions or merging segmentations.

    Args:
        segmentation_numpy (np.ndarray): Input segmentation array.
        segmentation_zooms (list): X,Y,Z side length of voxels in mm.
        blur_size_mm (float): FWHM of Gaussian kernal used to blur regions.
        regions (list): List of region indices to replace residual regions.
        regions_to_replace (list): List of regions to be replaced by nearby
            regions listed in ``regions``.

    Returns:
        segmentation_numpy (np.ndarray): The input segmentation with replaced
            regions.
    """
    segmentations_combined = []
    for region in regions:
        region_mask = region_blend(segmentation_numpy=segmentation_numpy,
                                   regions_list=[region])

        region_blur = math_lib.gauss_blur_computation(input_image=region_mask,
                                                      blur_size_mm=blur_size_mm,
                                                      input_zooms=segmentation_zooms,
                                                      use_FWHM=True)
        segmentations_combined += [region_blur]
    
    segmentations_combined_np = np.array(segmentations_combined)
    probability_map = np.argmax(segmentations_combined_np,axis=0)
    blend = region_blend(segmentation_numpy=segmentation_numpy,
                         regions_list=regions_to_replace)

    for i, region in enumerate(regions):
        region_match = (probability_map == i) & (blend > 0)
        segmentation_numpy[region_match] = region
    
    return segmentation_numpy


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

def vat_wm_region_merge(wmparc_segmentation_path: str,
                        bs_segmentation_path: str,
                        wm_ref_segmentation_path: str,
                        out_image_path: str):
    """
    Merge subcortical structures into a merged segmentation image according to
    the protocol for processing the VAT radiotracer.

    Args:
        wmparc_segmentation_path (str): Path to `wmparc` segmentation generated
            by FreeSurfer.
        bs_segmentation_path (str): Path to brainstem segmentation generated by
            FreeSurfer.
        wm_ref_segmentation_path (str): Path to eroded white matter reference
            region generated by :meth:`vat_wm_ref_region`.
        out_image_path (str): Path to which output fused segmentation is saved.
    """
    wmparc = nibabel.load(wmparc_segmentation_path)
    bs = nibabel.load(bs_segmentation_path)
    wm_ref = nibabel.load(wm_ref_segmentation_path)

    wmparc_img = wmparc.get_fdata()
    bs_img = bs.get_fdata()
    wm_ref_img = wm_ref.get_fdata()

    zooms = wmparc.header.get_zooms()

    wmparc_split = parcellate_right_left(segmentation_numpy=wmparc_img,
                                         region=77,
                                         new_right_region=2,
                                         new_left_region=41)

    wmparc_bs = segmentations_merge(segmentation_primary=wmparc_split,
                                    segmentation_secondary=bs_img,
                                    regions=[173,174,175])
    wmparc_bs_prob = replace_probabilistic_region(segmentation_numpy=wmparc_bs,
                                                  segmentation_zooms=zooms,
                                                  blur_size_mm=6,
                                                  regions=[257,15,165],
                                                  regions_to_replace=[16])

    wmparc_bs_wmref = segmentations_merge(segmentation_primary=wmparc_bs_prob,
                                          segmentation_secondary=wm_ref_img,
                                          regions=[1])

    out_file = nibabel.nifti1.Nifti1Image(dataobj=wmparc_bs_wmref,
                                          header=wmparc.header,
                                          affine=wmparc.affine)
    nibabel.save(out_file,
                 out_image_path)
