"""
Methods applying to segmentations.

Available methods:
* :meth:`region_blend`: Merge regions in a segmentation image into a mask with value 1
* :meth:`resample_segmentation`: Resample a segmentation image to the affine of a 4D PET image.
* :meth:`vat_wm_ref_region`: Compute the white matter reference region for the VAT radiotracer.

TODO:
 * Find a more efficient way to find the region mask in :meth:`segmentations_merge` that works for region=1

"""
import functools
import numpy as np
import ants
import nibabel
from nibabel import processing

from petpal.preproc import motion_corr
import pandas as pd

from . import image_operations_4d, motion_corr
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
                                                      use_fwhm=True)
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
                                                     use_fwhm=True)
    
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
                        out_image_path: str,
                        wm_ref_segmentation_path: str,
                        bs_segmentation_path: str = None):
    """
    Merge subcortical structures into a merged segmentation image according to
    the protocol for processing the VAT radiotracer.

    Args:
        wmparc_segmentation_path (str): Path to `wmparc` segmentation generated
            by FreeSurfer.
        out_image_path (str): Path to which output fused segmentation is saved.
        wm_ref_segmentation_path (str): Path to eroded white matter reference
            region generated by :meth:`vat_wm_ref_region`.
        bs_segmentation_path (str): Path to brainstem segmentation generated by
            FreeSurfer. If None, then skip.
    """
    wmparc = nibabel.load(wmparc_segmentation_path)
    wm_ref = nibabel.load(wm_ref_segmentation_path)

    wmparc_img = wmparc.get_fdata()
    wm_ref_img = wm_ref.get_fdata()

    zooms = wmparc.header.get_zooms()

    wmparc_split = parcellate_right_left(segmentation_numpy=wmparc_img,
                                         region=77,
                                         new_right_region=2,
                                         new_left_region=41)

    if bs_segmentation_path is None:
        wmparc_bs = wmparc_split
    else:
        bs = nibabel.load(bs_segmentation_path)
        bs_img = bs.get_fdata()
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
    if len(wmparc_bs_wmref.shape)==4:
        out_array = wmparc_bs_wmref[:,:,:,0]
    else:
        out_array = wmparc_bs_wmref
    out_file = nibabel.nifti1.Nifti1Image(dataobj=out_array,
                                          header=wmparc.header,
                                          affine=wmparc.affine)
    nibabel.save(out_file,out_image_path)


def gw_segmentation(freesurfer_path: str,
                    dseg_path: str,
                    output_path: str):
    """
    Creates a gray matter and a white matter mask based on FreeSurfer regions. Useful for PVC.
    """
    dseg = pd.read_csv(dseg_path,sep=r'\s+')
    freesurfer = ants.image_read(freesurfer_path)
    freesurfer_np = freesurfer.numpy()
    gm_map = np.zeros_like(freesurfer_np)
    wm_map = np.zeros_like(freesurfer_np)

    for i,mapping in enumerate(dseg['mapping']):
        gw_label = dseg['gray_white_matter'].iloc[i]
        region_seg = np.where(freesurfer_np==mapping)
        if gw_label == 0:
            gm_map[region_seg] = 1
        elif gw_label == 1:
            wm_map[region_seg] = 1

    gm_img = ants.from_numpy(data=gm_map,
                             origin=freesurfer.origin,
                             spacing=freesurfer.spacing,
                             direction=freesurfer.direction)
    wm_img = ants.from_numpy(data=wm_map,
                             origin=freesurfer.origin,
                             spacing=freesurfer.spacing,
                             direction=freesurfer.direction)
    gw_map_template = motion_corr._gen_nd_image_based_on_image_list([gm_img,wm_img])
    gw_map_4d = ants.list_to_ndimage(image=gw_map_template,image_list=[gm_img,wm_img])
    ants.image_write(gw_map_4d,output_path)


def subcortical_mask(input_seg_path: str,
                     output_seg_path: str=None,
                     subcortical_regions: list=None):
    """
    Gets a mask for subcortical regions from a FreeSurfer label image.

    Args:
        input_seg_path (str): Path to segmentation label image.
        output_seg_path (str): Path to which subcortical mask is saved.
        subcortical regions (list): Regions to include in the subcortical mask. Uses a built in
            list of subcortical mappings unless overridden by the user.

    Returns:
        subcortical_img (ants.ANTsImage): Subcortical mask image.

    .. important::
        * Subcortical mappings are assumed to correspond to a subset of FreeSurfer subcortical
            regions
        * Default regions: whole cerebellum, thalamus, caudate, putamen, pallidum, brainstem.
    """
    subcortical_mappings = [7,8,10,11,12,13,16,49,50,51,52,173,174,175]

    if subcortical_regions is None:
        subcortical_regions = subcortical_mappings

    segmentation = ants.image_read(input_seg_path)
    segmentation_np = segmentation.numpy()
    subcortical_mask_arr = np.zeros(segmentation_np.shape)
    for region in subcortical_regions:
        region_seg = np.where(segmentation_np==region)
        subcortical_mask_arr[region_seg] = 1
    subcortical_img = ants.from_numpy(
        data=subcortical_mask_arr,
        origin=segmentation.origin,
        spacing=segmentation.spacing,
        direction=segmentation.direction
    )

    if output_seg_path is not None:
        ants.image_write(subcortical_img,output_seg_path)

    return subcortical_img


def ANTsImageToANTsImage(func):
    """
    A decorator for functions that process an ANTs image and output another ANTs image.
    Assumes that the argument of the passed in function is an ANTs image.

    This decorator is designed to extend functions that take an ANTs image as input
    and output another ANTs image. It supports seamless handling of input images
    provided as either file paths (str) or `ants.core.ANTsImage` objects. The resulting
    processed image can optionally be saved to a specified file path.

    Args:
        func (Callable): The function to be decorated. It should accept an ANTs image as
            the first argument and return a processed ANTs image.

    Returns:
        Callable: A wrapper function that:
            - Reads the input image if a file path (str) is provided.
            - Passes an `ants.core.ANTsImage` object to the decorated function.
            - Saves the output image to the specified file path if `out_path` is provided.

    Wrapper Parameters:
        in_img (ants.core.ANTsImage | str): Input image, either an ANTsImage object or
            the file path to a NIfTI image.
        out_path (str): File path to save the output image. If `None`, the output image
            is not saved.
        *args: Additional positional arguments for the decorated function.
        **kwargs: Additional keyword arguments for the decorated function.

    Raises:
        TypeError: If `in_img` is not a string or `ants.core.ANTsImage`.

    Notes:
        - If `in_img` is provided as a file path, the image is read using `ants.image_read`.
        - The output image is written to the desired path using `ants.image_write` if
          `out_path` is specified.
    """

    @functools.wraps(func)
    def wrapper(in_img:ants.core.ANTsImage | str,
                out_path: str,
                *args, **kwargs):
        if isinstance(in_img, str):
            in_image = ants.image_read(in_img)
        elif isinstance(in_img, ants.core.ANTsImage):
            in_image = in_img
        else:
            raise TypeError('in_img must be str or ants.core.ANTsImage')
        out_img = func(in_image, *args, **kwargs)
        if out_path is not None:
            ants.image_write(out_img, out_path)
        return out_img
    return wrapper


def calc_vesselness_measure_image(input_image: ants.core.ANTsImage,
                                  sigma_min: float = 2.0,
                                  sigma_max: float = 8.0,
                                  alpha: float = 0.5,
                                  beta: float = 0.5,
                                  gamma: float = 5.0,
                                  morph_open_radius: int = 1,
                                  **hessian_func_kwargs) -> ants.core.ANTsImage:
    """
    Computes a vesselness measure image using Hessian-based objectness filtering.

    This function calculates the vesselness measure of a given 3D image using
    multi-scale Hessian filtering with specified parameters. We call the
    :func:`ants.hessian_objectness` after max-normalizing the input image.
    It enhances tubular structures like vessels, making them more pronounced
    in the output image. Optionally, a morphological opening operation can be
    applied to the result to refine the output and remove pepper-like artefacts.

    From the docs of :func:`ants.hessian_objectness`:
    '
    Based on the paper by Westin et al., "Geometrical
    Diffusion Measures for MRI from Tensor Basis Analysis" and Luca Antiga's
    Insight Journal paper http://hdl.handle.net/1926/576.
    '

    Args:
        input_image (ants.core.ANTsImage): Input 3D image for vesselness computation.
        sigma_min (float, optional): Minimum scale for multi-scale Hessian filtering
            (default: 2.0).
        sigma_max (float, optional): Maximum scale for multi-scale Hessian filtering
            (default: 8.0).
        alpha (float, optional): Alpha parameter for vesselness computation
            (default: 0.5).
        beta (float, optional): Beta parameter for vesselness computation
            (default: 0.5).
        gamma (float, optional): Gamma parameter for vesselness computation
            (default: 5.0).
        morph_open_radius (int, optional): Radius for the optional morphological
            opening operation (default: 1). If set to 0, no morphological opening
            will be applied.
        **hessian_func_kwargs: Additional keyword arguments for the Hessian
            objectness function.

    Returns:
        ants.core.ANTsImage: The vesselness-enhanced image.

    Notes:
        - Input image must be 3D; an assertion will fail if a non-3D image is provided.
        - The input image is normalized before processing to ensure robustness.
        - Morphological opening, if applied, uses a grayscale operation to refine
          the tubular structures.
        - The function has defaults for vesselness computation, but can be used to detect
          globular or plate-like structures as well.

    Raises:
        AssertionError: If the input image is not 3D.

    Workflow:
        1. Normalize the input image to have values between 0 and 1.
        2. Apply Hessian-based objectness filtering using the provided parameters
           (`sigma_min`, `sigma_max`, `alpha`, `beta`, `gamma`).
        3. Perform a morphological opening operation with the specified radius
           (`morph_open_radius`), if applicable.
        4. Return the computed vesselness image.
    """
    assert len(input_image.shape) == 3, "Input image must be 3D."

    tmp_img = input_image / input_image.max()
    hess_objectness_img = tmp_img.hessian_objectness(sigma_min=sigma_min,
                                                     sigma_max=sigma_max,
                                                     gamma=gamma,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     **hessian_func_kwargs)
    if morph_open_radius > 0:
        hess_objectness_img = hess_objectness_img.morphology(operation='open',
                                                             radius=morph_open_radius,
                                                             mtype='grayscale')
    return hess_objectness_img


def calc_vesselness_mask_from_quantiled_vesselness(input_image: ants.core.ANTsImage,
                                                   min_quantile: float = 0.99,
                                                   morph_dil_radius: int = 0,
                                                   z_crop: int = 3) -> ants.core.ANTsImage:
    """
    Generates a binary vesselness mask from a given vesselness image using quantile-based thresholding.

    This function creates a binary mask by thresholding a vesselness image at a specified
    quantile of non-zero voxel values. Additionally, it allows for optional z-axis cropping
    and morphological dilation to refine the mask.

    Args:
        input_image (ants.core.ANTsImage): Input vesselness image.
        min_quantile (float, optional): Minimum quantile value for voxel thresholding
            (default: 0.99). Must be in the range [0, 1).
        morph_dil_radius (int, optional): Radius for morphological dilation to refine the
            mask (default: 0). No dilation is applied if set to 0.
        z_crop (int, optional): Number of slices to crop from the z-axis from the bottom
            (default: 3).

    Returns:
        ants.core.ANTsImage: Binary vesselness mask.

    Notes:
        - The input image must be an ANTs image containing vesselness measures.
        - The quantile value (`min_quartile`) determines the threshold value based on
          non-zero voxel intensities.
        - If `z_crop` is greater than 0, the z-axis is cropped from the top and bottom.
        - Morphological dilation is applied to the binary mask with the specified radius,
          if provided.
          
    Raises:
        - AssertionError: If the input image is not 3D.
        - AssertionError: If the provided quantile is not in the range [0, 1].
        - AssertionError: If the provided z-crop is larger than the number of z-slices in the
          input image.

    """
    assert 1 >= min_quantile >= 0, "Minimal quantile must be greater than 0 and less than 1."
    assert len(input_image.shape) == 3, "Input image must be 3D."
    assert z_crop < input_image.shape[2], "Z-crop must be less than input image's Z-dimension."

    vess_vals_arr = input_image.numpy()
    vess_vals_arr = vess_vals_arr[vess_vals_arr != 0].flatten()
    thresh_val = np.quantile(vess_vals_arr, q=min_quantile)
    vess_mask_img = input_image.threshold_image(low_thresh=thresh_val, high_thresh=None)

    vess_mask_img[:, :, :z_crop] = 0

    if morph_dil_radius > 0:
        vess_mask_img = vess_mask_img.morphology(operation='dilate', radius=morph_dil_radius)
    return vess_mask_img

step_calc_vesselness_measure_image = ANTsImageToANTsImage(calc_vesselness_measure_image)
step_calc_vesselness_mask_from_quantiled_vesselness = ANTsImageToANTsImage(calc_vesselness_mask_from_quantiled_vesselness)