"""
The 'image_operations_4d' module provides several functions used to do preprocessing
on 4D PET imaging series. These functions typically take one or more paths to imaging
data in NIfTI format, and save modified data to a NIfTI file, and may return the
modified imaging array as output.

Class :class:`ImageOps4D`` is also included in this module, and provides specific
implementations of the functions presented herein.

TODOs:
    * (weighted_series_sum) Refactor the DecayFactor key extraction into its own function
    * (weighted_series_sum) Refactor verbose reporting into the class as it is unrelated to
      computation
    * (write_tacs) Shift to accepting color-key dictionaries rather than a file path.
    * (extract_tac_from_4dnifty_using_mask) Write the number of voxels in the mask, or the
      volume of the mask. This is necessary for certain analyses with the resulting tacs,
      such as finding the average uptake encompassing two regions.
    * Methods that create new images should copy over a previous metadata file, if one exists,
      and create a new one if it does not.

"""
import os
import re
import tempfile
from typing import Union
import fsl.wrappers
from scipy.interpolate import interp1d
import ants
import nibabel
from nibabel import processing
import numpy as np
from . import image_io
from . import math_lib
from . import qc_plots


def weighted_series_sum(input_image_4d_path: str,
                        out_image_path: str,
                        half_life: float,
                        verbose: bool,
                        start_time: float=0,
                        end_time: float=-1) -> np.ndarray:
    r"""
    Sum a 4D image series weighted based on time and re-corrected for decay correction.

    First, a scaled image is produced by multiplying each frame by its length in seconds,
    and dividing by the decay correction applied:

    .. math::
    
        f_i'=f_i\times \frac{t_i}{d_i}

    Where :math:`f_i,t_i,d_i`` are the i-th frame, frame duration, and decay correction factor of
    the PET series. This scaled image is summed over the time axis. Then, to get the output, we
    multiply by a factor called ``total decay`` and divide by the full length of the image:

    .. math::

        d_{S} = \frac{\lambda*t_{S}}{(1-\exp(-\lambda*t_{S}))(\exp(\lambda*t_{0}))}

    .. math::
    
        S(f) = \sum(f_i') * d_{S} / t_{S}

    where :math:`\lambda=\log(2)/T_{1/2}` is the decay constant of the radio isotope,
    :math:`t_0` is the start time of the first frame in the PET series, the subscript :math:`S`
    indicates the total quantity computed over all frames, and :math:`S(f)` is the final weighted
    sum image.

    
    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image on which the weighted sum is calculated. Assume a metadata
            file exists with the same path and file name, but with extension .json,
            and follows BIDS standard.
        out_image_path (str): Path to a .nii or .nii.gz file to which the weighted
            sum is written.
        half_life (float): Half life of the PET radioisotope in seconds.
        verbose (bool): Set to ``True`` to output processing information.
        start_time (float): Time, relative to scan start in seconds, at which
            calculation begins. Must be used with ``end_time``. Default value 0.
        end_time (float): Time, relative to scan start in seconds, at which
            calculation ends. Use value ``-1`` to use all frames in image series.
            If equal to ``start_time`, one frame at start_time is used. Default value -1.

    Returns:
        summed_image (np.ndarray): 3D image array, in the same space as the input,
            with the weighted sum calculation applied.

    Raises:
        ValueError: If ``half_life`` is zero or negative.
    """
    if half_life <= 0:
        raise ValueError('(ImageOps4d): Radioisotope half life is zero or negative.')
    pet_meta = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    pet_image = nibabel.load(input_image_4d_path)
    pet_series = pet_image.get_fdata()
    frame_start = pet_meta['FrameTimesStart']
    frame_duration = pet_meta['FrameDuration']

    if 'DecayCorrectionFactor' in pet_meta.keys():
        decay_correction = pet_meta['DecayCorrectionFactor']
    elif 'DecayFactor' in pet_meta.keys():
        decay_correction = pet_meta['DecayFactor']
    else:
        raise ValueError("Neither 'DecayCorrectionFactor' nor 'DecayFactor' exist in meta-data file")

    if 'TracerRadionuclide' in pet_meta.keys():
        tracer_isotope = pet_meta['TracerRadionuclide']
        if verbose:
            print(f"(ImageOps4d): Radio isotope is {tracer_isotope} "
                f"with half life {half_life} s")

    if end_time==-1:
        pet_series_adjusted = pet_series
        frame_start_adjusted = frame_start
        frame_duration_adjusted = frame_duration
        decay_correction_adjusted = decay_correction
    else:
        scan_start = frame_start[0]
        nearest_frame = interp1d(x=frame_start,
                                 y=range(len(frame_start)),
                                 kind='nearest',
                                 bounds_error=False,
                                 fill_value='extrapolate')
        calc_first_frame = int(nearest_frame(start_time+scan_start))
        calc_last_frame = int(nearest_frame(end_time+scan_start))
        if calc_first_frame==calc_last_frame:
            calc_last_frame += 1
        pet_series_adjusted = pet_series[:,:,:,calc_first_frame:calc_last_frame]
        frame_start_adjusted = frame_start[calc_first_frame:calc_last_frame]
        frame_duration_adjusted = frame_duration[calc_first_frame:calc_last_frame]
        decay_correction_adjusted = decay_correction[calc_first_frame:calc_last_frame]

    image_weighted_sum = math_lib.weighted_sum_computation(frame_duration=frame_duration_adjusted,
                                                           half_life=half_life,
                                                           pet_series=pet_series_adjusted,
                                                           frame_start=frame_start_adjusted,
                                                           decay_correction=decay_correction_adjusted)

    pet_sum_image = nibabel.nifti1.Nifti1Image(dataobj=image_weighted_sum,
                                               affine=pet_image.affine,
                                               header=pet_image.header)
    nibabel.save(pet_sum_image, out_image_path)
    if verbose:
        print(f"(ImageOps4d): weighted sum image saved to {out_image_path}")
    return pet_sum_image


def determine_motion_target(motion_target_option: Union[str,tuple],
                            input_image_4d_path: str=None,
                            half_life: float=None):
    """
    Produce a motion target given the ``motion_target_option`` from a method
    running registrations on PET, i.e. :meth:`motion_correction` or
    :meth:`register_pet`.

    The motion target option can be a string or a tuple. If it is a string,
    then if this string is a file, use the file as the motion target.

    If it is the option ``weighted_series_sum``, then run
    :meth:`weighted_series_sum` and return the output path.

    If it is a tuple, run a weighted sum on the PET series on a range of 
    frames. The elements of the tuple are treated as times in seconds, counted
    from the time of the first frame, i.e. (0,300) would average all frames 
    from the first to the frame 300 seconds later. If the two elements are the
    same, returns the frame closest to the entered time.

    Args:
        motion_target_option (str | tuple): Determines how the method behaves,
            according to the above description. Can be a file, a method
            ('weighted_series_sum'), or a tuple range e.g. (0,300).
        input_image_4d_path (str): Path to the PET image. This is intended to
            be supplied by the parent method employing this function. Default
            value None.
        half_life (float): Half life of the radiotracer used in the image
            located at ``input_image_4d_path``. Only used if a calculation is
            performed.
    
    Returns:
        out_image_file (str): File to use as a target to compute
            transformations on.
    """
    if isinstance(motion_target_option,str):
        if os.path.exists(motion_target_option):
            return motion_target_option
        elif motion_target_option=='weighted_series_sum':
            out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
            weighted_series_sum(input_image_4d_path=input_image_4d_path,
                                out_image_path=out_image_file,
                                half_life=half_life,
                                verbose=False)
            return out_image_file
    elif isinstance(motion_target_option,tuple):
        start_time = motion_target_option[0]
        end_time = motion_target_option[1]
        try:
            float(start_time)
            float(end_time)
        except:
            raise TypeError('Start time and end time of calculation must be '
                            'able to be cast into float! Provided values are '
                            f"{start_time} and {end_time}.")
        out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
        weighted_series_sum(input_image_4d_path=input_image_4d_path,
                            out_image_path=out_image_file,
                            half_life=half_life,
                            verbose=False,
                            start_time=start_time,
                            end_time=end_time)
        return out_image_file


def motion_correction(input_image_4d_path: str,
                      motion_target_option: Union[str,tuple],
                      out_image_path: str,
                      verbose: bool,
                      type_of_transform: str='DenseRigid',
                      half_life: float=None,
                      **kwargs) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Correct PET image series for inter-frame motion. Runs rigid motion
    correction module from Advanced Normalisation Tools (ANTs) with default
    inputs.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be motion corrected.
        motion_target_option (str | tuple): Target image for computing
            transformation. See :meth:`determine_motion_target`.
        reference_image_path (str): Path to a .nii or .nii.gz file containing a
            3D reference image in the same space as the input PET image. Can be
            a weighted series sum, first or last frame, an average over a
            subset of frames, or another option depending on the needs of the
            data.
        out_image_path (str): Path to a .nii or .nii.gz file to which the
            motion corrected PET series is written.
        verbose (bool): Set to ``True`` to output processing information.
        type_of_transform (str): Type of transform to perform on the PET image,
            must be one of antspy's transformation types, i.e. 'DenseRigid' or
            'Translation'. Any transformation type that uses >6 degrees of
            freedom is not recommended, use with caution. See 
            :py:func:`ants.registration`.
        half_life (float): Half life of the PET radioisotope in seconds.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`ants.motion_correction`.

    Returns:
        pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy
            array.
        pet_moco_params (list[str]): List of ANTS registration files applied to
            each frame.
        pet_moco_fd (list[float]): List of framewise displacement measure
            corresponding to each frame transform.
    """
    pet_ants = ants.image_read(input_image_4d_path)

    motion_target_image_path = determine_motion_target(motion_target_option=motion_target_option,
                                                       input_image_4d_path=input_image_4d_path,
                                                       half_life=half_life)

    motion_target_image = ants.image_read(motion_target_image_path)
    motion_target_image_ants = ants.from_nibabel(motion_target_image)
    pet_moco_ants_dict = ants.motion_correction(image=pet_ants,
                                                fixed=motion_target_image_ants,
                                                type_of_transform=type_of_transform,
                                                **kwargs)
    if verbose:
        print('(ImageOps4D): motion correction finished.')

    pet_moco_ants = pet_moco_ants_dict['motion_corrected']
    pet_moco_params = pet_moco_ants_dict['motion_parameters']
    pet_moco_fd = pet_moco_ants_dict['FD']
    pet_moco_np = pet_moco_ants.numpy()
    pet_moco_nibabel = ants.to_nibabel(pet_moco_ants)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(
        input_image_4d_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)

    nibabel.save(pet_moco_nibabel, out_image_path)
    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")
    return pet_moco_np, pet_moco_params, pet_moco_fd


def register_pet(input_reg_image_path: str,
                 reference_image_path: str,
                 motion_target_option: Union[str,tuple],
                 out_image_path: str,
                 verbose: bool,
                 type_of_transform: str='DenseRigid',
                 half_life: str=None,
                 **kwargs):
    """
    Computes and runs rigid registration of 4D PET image series to 3D anatomical image, typically
    a T1 MRI. Runs rigid registration module from Advanced Normalisation Tools (ANTs) with  default
    inputs. Will upsample PET image to the resolution of anatomical imaging.

    Args:
        input_reg_image_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be registered to anatomical space.
        reference_image_path (str): Path to a .nii or .nii.gz file containing a 3D
            anatomical image to which PET image is registered.
        motion_target_option (str | tuple): Target image for computing
            transformation. See :meth:`determine_motion_target`.
        type_of_transform (str): Type of transform to perform on the PET image, must be one of antspy's
            transformation types, i.e. 'DenseRigid' or 'Translation'. Any transformation type that uses
            >6 degrees of freedom is not recommended, use with caution. See :py:func:`ants.registration`.
        out_image_path (str): Path to a .nii or .nii.gz file to which the registered PET series
            is written.
        verbose (bool): Set to ``True`` to output processing information.
        kwargs (keyword arguments): Additional arguments passed to :py:func:`ants.registration`.
    """
    motion_target = determine_motion_target(motion_target_option=motion_target_option,
                                                input_image_4d_path=input_reg_image_path,
                                                half_life=half_life)
    motion_target_image = ants.image_read(motion_target)
    mri_image = ants.image_read(reference_image_path)
    pet_moco = ants.image_read(input_reg_image_path)
    xfm_output = ants.registration(moving=motion_target_image,
                                   fixed=mri_image,
                                   type_of_transform=type_of_transform,
                                   write_composite_transform=True,
                                   **kwargs)
    if verbose:
        print(f'Registration computed transforming image {motion_target} to '
              f'{reference_image_path} space')

    xfm_apply = ants.apply_transforms(moving=pet_moco,
                                      fixed=mri_image,
                                      transformlist=xfm_output['fwdtransforms'],
                                      imagetype=3)
    if verbose:
        print(f'Registration applied to {input_reg_image_path}')

    ants.image_write(xfm_apply, out_image_path)
    if verbose:
        print(f'Transformed image saved to {out_image_path}')

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_reg_image_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)


def warp_pet_atlas(input_image_path: str,
                   anat_image_path: str,
                   atlas_image_path: str,
                   out_image_path: str,
                   verbose: bool,
                   type_of_transform: str='SyN',
                   **kwargs):
    """
    Compute and apply a warp on a 3D or 4D image in anatomical space
    to atlas space using ANTs.

    Args:
        input_image_path (str): Image to be registered to atlas. Must be in
            anatomical space. May be 3D or 4D.
        anat_image_path (str): Image used to compute registration to atlas space.
        atlas_image_path (str): Atlas to which input image is warped.
        out_image_path (str): Path to which warped image is saved.
        type_of_transform (str): Type of non-linear transform applied to input 
            image using :py:func:`ants.registration`.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`ants.registration`.
    
    Returns:
        xfm_to_apply (list[str]): The computed transforms, saved to a temp dir.
    """
    pet_image_ants = ants.image_read(input_image_path)
    anat_image_ants = ants.image_read(anat_image_path)
    atlas_image_ants = ants.image_read(atlas_image_path)

    anat_atlas_xfm = ants.registration(fixed=atlas_image_ants,
                                       moving=anat_image_ants,
                                       type_of_transform=type_of_transform,
                                       write_composite_transform=True,
                                       **kwargs)
    xfm_to_apply = anat_atlas_xfm['fwdtransforms']
    if verbose:
        print(f'Xfms located at: {xfm_to_apply}')

    dim = pet_image_ants.dimension
    pet_atlas_xfm = ants.apply_transforms(fixed=atlas_image_ants,
                                          moving=pet_image_ants,
                                          transformlist=xfm_to_apply,
                                          imagetype=dim-1)

    ants.image_write(pet_atlas_xfm,out_image_path)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)

    return xfm_to_apply

def apply_xfm_ants(input_image_path: str,
                   ref_image_path: str,
                   out_image_path: str,
                   xfm_paths: list[str]):
    """
    Applies existing transforms in ANTs or ITK format to an input image, onto
    a reference image. This is useful for applying the same transform on
    different images to atlas space, for example.

    Args:
        input_image_path (str): Path to image on which transform is applied.
        ref_image_path (str): Path to image to which transform is applied.
        out_image_path (str): Path to which the transformed image is saved.
        xfm_paths (list[str]): List of transforms to apply to image. Must be in
            ANTs or ITK format, and can be affine matrix or warp coefficients.
    """
    pet_image_ants = ants.image_read(input_image_path)
    ref_image_ants = ants.image_read(ref_image_path)

    dim = pet_image_ants.dimension
    xfm_image = ants.apply_transforms(fixed=ref_image_ants,
                                      moving=pet_image_ants,
                                      transformlist=xfm_paths,
                                      imagetype=dim-1)

    ants.image_write(xfm_image,out_image_path)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)


def apply_xfm_fsl(input_image_path: str,
                  ref_image_path: str,
                  out_image_path: str,
                  warp_path: str=None,
                  premat_path: str=None,
                  postmat_path: str=None,
                  **kwargs):
    """
    Applies existing transforms in FSL format to an input image, onto a
    reference image. This is useful for applying the same transform on
    different images to atlas space, for example.

    .. important::
        Requires installation of ``FSL``, and environment variables ``FSLDIR`` and
        ``FSLOUTPUTTYPE`` set appropriately in the shell.

    Args:
        input_image_path (str): Path to image on which transform is applied.
        ref_image_path (str): Path to image to which transform is applied.
        out_image_path (str): Path to which the transformed image is saved.
        warp_path (str): Path to FSL warp file.
        premat_path (str): Path to FSL ``premat`` matrix file.
        postmat_path (str): Path to FSL ``postmat`` matrix file.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`fsl.wrappers.applywarp`.
    """

    fsl.wrappers.applywarp(src=input_image_path,
                           ref=ref_image_path,
                           out=out_image_path,
                           warp=warp_path,
                           premat=premat_path,
                           postmat=postmat_path,
                           **kwargs)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)


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


def extract_tac_from_4dnifty_using_mask(input_image_4d_path: str,
                                        segmentation_image_path: str,
                                        region: int,
                                        verbose: bool) -> np.ndarray:
    """
    Creates a time-activity curve (TAC) by computing the average value within a region, for each 
    frame in a 4D PET image series. Takes as input a PET image, which has been registered to
    anatomical space, a segmentation image, with the same sampling as the PET, and a list of values
    corresponding to regions in the segmentation image that are used to compute the average
    regional values. Currently, only the mean over a single region value is implemented.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image, registered to anatomical space.
        segmentation_image_path (str): Path to a .nii or .nii.gz file containing a 3D segmentation
            image, where integer indices label specific regions. Must have same sampling as PET
            input.
        region (int): Value in the segmentation image corresponding to a region
            over which the TAC is computed.
        verbose (bool): Set to ``True`` to output processing information.

    Returns:
        tac_out (np.ndarray): Mean of PET image within regions for each frame in 4D PET series.

    Raises:
        ValueError: If the segmentation image and PET image have different
            sampling.
    """

    pet_image_4d = nibabel.load(input_image_4d_path).get_fdata()
    num_frames = pet_image_4d.shape[3]
    seg_image = nibabel.load(segmentation_image_path).get_fdata()

    if seg_image.shape!=pet_image_4d.shape[:3]:
        raise ValueError('Mis-match in image shape of segmentation image '
                         f'({seg_image.shape}) and PET image '
                         f'({pet_image_4d.shape[:3]}). Consider resampling '
                         'segmentation to PET or vice versa.')

    tac_out = np.zeros(num_frames, float)
    if verbose:
        print(f'Running TAC for region index {region}')
    masked_voxels = seg_image == region
    masked_image = pet_image_4d[masked_voxels].reshape((-1, num_frames))
    tac_out = np.mean(masked_image, axis=0)
    return tac_out


def suvr(input_image_path: str,
         segmentation_image_path: str,
         ref_region: int,
         out_image_path: str,
         verbose: bool):
    """
    Computes an ``SUVR`` (Standard Uptake Value Ratio) by taking the average of
    an input image within a reference region, and dividing the input image by
    said average value.

    Args:
        input_image_path (str): Path to 3D weighted series sum or other
            parametric image on which we compute SUVR.
        segmentation_image_path (str): Path to segmentation image, which we use
            to compute average uptake value in the reference region.
        ref_region (int): Region number mapping to the reference region in the
            segmentation image.
        out_image_path (str): Path to output image file which is written to.
        verbose (bool): Set to ``True`` to output processing information.
    """
    ref_region_avg = extract_tac_from_4dnifty_using_mask(input_image_4d_path=input_image_path,
                                                         segmentation_image_path=segmentation_image_path,
                                                         region=ref_region,
                                                         verbose=verbose)

    pet_nibabel = nibabel.load(filename=input_image_path)
    pet_image = pet_nibabel.get_fdata()
    suvr_image = pet_image / ref_region_avg,

    out_image = nibabel.nifti1.Nifti1Image(dataobj=suvr_image,
                                           affine=pet_nibabel.affine,
                                           header=pet_nibabel.header)
    nibabel.save(img=out_image,filename=out_image_path)
    return out_image


def gauss_blur(input_image_path: str,
               blur_size_mm: float,
               out_image_path: str,
               verbose: bool):
    """
    Blur an image with a 3D Gaussian kernal of a provided size in mm.

    Args:
        input_image_path (str): Path to 3D or 4D input image to be blurred.
        blur_size_mm (float): Size of the Gaussian kernal in mm.
        out_image_path (str): Path to save the blurred output image.
        verbose (bool): Set to ``True`` to output processing information.
    """
    input_image = nibabel.load(filename=input_image_path)


def write_tacs(input_image_4d_path: str,
               label_map_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               verbose: bool,
               time_frame_keyword: str = 'FrameReferenceTime'):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and label map. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
    regions_abrev = label_map['abbreviations']
    regions_map = label_map['mappings']

    tac_extraction_func = extract_tac_from_4dnifty_using_mask

    for i, _maps in enumerate(label_map['mappings']):
        extracted_tac = tac_extraction_func(input_image_4d_path=input_image_4d_path,
                                            segmentation_image_path=segmentation_image_path,
                                            region=int(regions_map[i]),
                                            verbose=verbose)
        region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
        header_text = f'{time_frame_keyword}\t{regions_abrev[i]}_mean_activity'
        out_tac_path = os.path.join(out_tac_dir, f'tac-{regions_abrev[i]}.tsv')
        np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')


class ImageOps4d():
    """
    :class:`ImageOps4D` to provide basic implementations of the preprocessing functions in module
    ``image_operations_4d``. Uses a properties dictionary ``preproc_props`` to
    determine the inputs and outputs of preprocessing methods.

    Key methods include:
        - :meth:`update_props`: Update properties dictionary ``preproc_props``
          with new properties.
        - :meth:`run_preproc`: Given a method in ``image_operations_4d``, run the
          provided method with inputs and outputs determined by properties
          dictionary ``preproc_props``.

    Attributes:
        -`output_directory`: Directory in which files are written to.
        -`output_filename_prefix`: Prefix appended to beginning of written
         files.
        -`preproc_props`: Properties dictionary used to set parameters for PET
         preprocessing.

    Example:

    .. code-block:: python
        output_directory = '/path/to/processing'
        output_filename_prefix = 'sub-01'
        sub_01 = pet_cli.image_operations_4d.ImageOps4d(output_directory,output_filename_prefix)
        params = {
            'FilePathPET': '/path/to/pet.nii.gz',
            'FilePathAnat': '/path/to/mri.nii.gz',
            'HalfLife': 1220.04,  # C11 half-life in seconds
            'FilePathRegInp': '/path/to/image/to/be/registered.nii.gz',
            'FilePathMocoInp': '/path/to/image/to/be/motion/corrected.nii.gz',
            'MotionTarget': '/path/to/pet/reference/target.nii.gz',
            'FilePathTACInput': '/path/to/registered/pet.nii.gz',
            'FilePathLabelMap': '/path/to/label/map.tsv',
            'FilePathSeg': '/path/to/segmentation.nii.gz',
            'TimeFrameKeyword': 'FrameTimesStart'  # using start time or midpoint reference time
            'Verbose': True,
        }
        sub_01.update_props(params)
        sub_01.run_preproc('weighted_series_sum')
        sub_01.run_preproc('motion_correction')
        sub_01.run_preproc('register_pet')
        sub_01.run_preproc('write_tacs')


    See Also:
        :class:`ImageIO`

    """
    def __init__(self,
                 output_directory: str,
                 output_filename_prefix: str) -> None:
        self.output_directory = os.path.abspath(output_directory)
        self.output_filename_prefix = output_filename_prefix
        self.preproc_props = self._init_preproc_props()


    @staticmethod
    def _init_preproc_props() -> dict:
        """
        Initializes preproc properties dictionary.

        The available fields in the preproc properties dictionary are described
        as follows:
            * FilePathPET (str): Path to PET file to be analysed.
            * FilePathMocoInp (str): Path to PET file to be motion corrected.
            * FilePathRegInp (str): Path to PET file to be registered to anatomical data.
            * FilePathAnat (str): Path to anatomical image to which ``FilePathRegInp`` is registered.
            * FilePathTACInput (str): Path to PET file with which TACs are computed.
            * FilePathSeg (str): Path to a segmentation image in anatomical space.
            * FilePathLabelMap (str): Path to a label map file, indexing segmentation values to ROIs.
            * MotionTarget (str | tuple): Target for transformation methods. See :meth:`determine_motion_target`.
            * MocoPars (keyword arguments): Keyword arguments fed into the method call :meth:`ants.motion_correction`.
            * RegPars (keyword arguments): Keyword arguments fed into the method call :meth:`ants.registration`.
            * HalfLife (float): Half life of radioisotope.
            * RegionExtract (int): Region index in the segmentation image to extract TAC from, if running TAC on a single ROI.
            * TimeFrameKeyword (str): Keyword in metadata file corresponding to frame timing array to be used in analysis.
            * Verbose (bool): Set to ``True`` to output processing information.

        """
        preproc_props = {'FilePathPET': None,
                 'FilePathMocoInp': None,
                 'FilePathRegInp': None,
                 'FilePathAnat': None,
                 'FilePathTACInput': None,
                 'FilePathSeg': None,
                 'FilePathLabelMap': None,
                 'MotionTarget': None,
                 'MocoPars': None,
                 'RegPars': None,
                 'HalfLife': None,
                 'RegionExtract': None,
                 'TimeFrameKeyword': None,
                 'Verbose': False}
        return preproc_props
    

    def update_props(self,new_preproc_props: dict) -> dict:
        """
        Update the processing properties with items from a new dictionary.

        Args:
            new_preproc_props (dict): Dictionary with updated properties 
                values. Can have any or all fields filled from the available
                list of fields.

        Returns:
            updated_props (dict): The updated ``preproc_props`` dictionary.


        """
        preproc_props = self.preproc_props
        valid_keys = [*preproc_props]
        updated_props = preproc_props.copy()
        keys_to_update = [*new_preproc_props]

        for key in keys_to_update:

            if key not in valid_keys:
                raise ValueError("Invalid preproc property! Expected one of:\n"
                                 f"{valid_keys}.\n Got {key}.")

            updated_props[key] = new_preproc_props[key]

        self.preproc_props = updated_props
        return updated_props


    def _check_method_props_exist(self,
                                 method_name: str) -> None:
        """
        Check if all necessary properties exist in the ``props`` dictionary to
        run the given method.

        Args:
            method_name (str): Name of method to be checked. Must be name of 
                a method in this module.
        """
        preproc_props = self.preproc_props
        existing_keys = [*preproc_props]

        if method_name=='weighted_series_sum':
            required_keys = ['FilePathPET','HalfLife','Verbose']
        elif method_name=='motion_correction':
            required_keys = ['FilePathMocoInp','MotionTarget','Verbose']
        elif method_name=='register_pet':
            required_keys = ['MotionTarget','FilePathRegInp','FilePathAnat','Verbose']
        elif method_name=='resample_segmentation':
            required_keys = ['FilePathTACInput','FilePathSeg','Verbose']
        elif method_name=='extract_tac_from_4dnifty_using_mask':
            required_keys = ['FilePathTACInput','FilePathSeg','RegionExtract','Verbose']
        elif method_name=='write_tacs':
            required_keys = ['FilePathTACInput','FilePathLabelMap','FilePathSeg','Verbose','TimeFrameKeyword']
        else:
            raise ValueError("Invalid method_name! Must be either"
                             "'weighted_series_sum', 'motion_correction', "
                             "'register_pet', 'resample_segmentation', "
                             "'extract_tac_from_4dnifty_using_mask', or "
                             f"'write_tacs'. Got {method_name}")
        for key in required_keys:
            if key not in existing_keys:
                raise ValueError(f"Preprocessing method requires property"
                                 f" {key}, however {key} was not found in "
                                 "processing properties. Existing properties "
                                 f"are: {existing_keys}, while needed keys to "
                                 f"run {method_name} are: {required_keys}.")


    def run_preproc(self,
                    method_name: str):
        """
        Run a specific preprocessing step.

        Args:
            method_name (str): Name of method to be run. Must be name of a
                method in this module.
        """
        preproc_props = self.preproc_props
        self._check_method_props_exist(method_name=method_name)
        if method_name=='weighted_series_sum':
            output_file_name = f'{self.output_filename_prefix}_wss.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            weighted_series_sum(input_image_4d_path=preproc_props['FilePathPET'],
                                out_image_path=outfile,
                                half_life=preproc_props['HalfLife'],
                                verbose=preproc_props['Verbose'])
        elif method_name=='motion_correction':
            output_file_name = f'{self.output_filename_prefix}_moco.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            moco_outputs = motion_correction(input_image_4d_path=preproc_props['FilePathMocoInp'],
                                             motion_target_option=preproc_props['MotionTarget'],
                                             out_image_path=outfile,
                                             verbose=preproc_props['Verbose'],
                                             half_life=preproc_props['HalfLife'],
                                             kwargs=preproc_props['MocoPars'])
            motion = moco_outputs[2]
            output_plot = os.path.join(self.output_directory,
                                       f'{self.output_filename_prefix}_motion.png')
            qc_plots.motion_plot(framewise_displacement=motion,
                                 output_plot=output_plot)
            return moco_outputs
        elif method_name=='register_pet':
            output_file_name = f'{self.output_filename_prefix}_reg.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            register_pet(motion_target_option=preproc_props['MotionTarget'],
                         input_reg_image_path=preproc_props['FilePathRegInp'],
                         reference_image_path=preproc_props['FilePathAnat'],
                         out_image_path=outfile,
                         verbose=preproc_props['Verbose'],
                         half_life=preproc_props['HalfLife'],
                         kwargs=preproc_props['RegPars'])
        elif method_name=='resample_segmentation':
            output_file_name = f'{self.output_filename_prefix}_seg-res.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            resample_segmentation(input_image_4d_path=preproc_props['FilePathTACInput'],
                                  segmentation_image_path=preproc_props['FilePathSeg'],
                                  out_seg_path=outfile,
                                  verbose=preproc_props['Verbose'])
            self.update_props({'FilePathSeg': outfile})
        elif method_name=='extract_tac_from_4dnifty_using_mask':
            return extract_tac_from_4dnifty_using_mask(input_image_4d_path=preproc_props['FilePathTACInput'],
                                                segmentation_image_path=preproc_props['FilePathSeg'],
                                                region=preproc_props['RegionExtract'],
                                                verbose=preproc_props['Verbose'])
        elif method_name=='write_tacs':
            outdir = os.path.join(self.output_directory,'tacs')
            os.makedirs(outdir,exist_ok=True)
            write_tacs(input_image_4d_path=preproc_props['FilePathTACInput'],
                       label_map_path=preproc_props['FilePathLabelMap'],
                       segmentation_image_path=preproc_props['FilePathSeg'],
                       out_tac_dir=outdir,
                       verbose=preproc_props['Verbose'],
                       time_frame_keyword=preproc_props['TimeFrameKeyword'])
        else:
            raise ValueError("Invalid method_name! Must be either"
                             "'weighted_series_sum', 'motion_correction', "
                             "'register_pet', 'resample_segmentation', "
                             "'extract_tac_from_4dnifty_using_mask', or "
                             f"'write_tacs'. Got {method_name}")
        return None
