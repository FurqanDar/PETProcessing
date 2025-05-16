r"""
The 'image_operations_4d' module provides several functions used to do preprocessing
on 4D PET imaging series. These functions typically take one or more paths to imaging
data in NIfTI format, and save modified data to a NIfTI file, and may return the
modified imaging array as output.

TODO:
    *   (weighted_series_sum) Refactor the DecayFactor key extraction into its own function
    *   (weighted_series_sum) Refactor verbose reporting into the class as it is unrelated to
        computation
    *   (extract_tac_from_4dnifty_using_mask) Write the number of voxels in the mask, or the
        volume of the mask. This is necessary for certain analyses with the resulting tacs,
        such as finding the average uptake encompassing two regions.
    *   Methods that create new images should copy over a previous metadata file, if one exists,
        and create a new one if it does not.
    *   (stitch_broken_scans) Separate 'add desc entity' section to its own function somewhere.
    *   (stitch_broken_scans) Assumes non-BIDS key 'DecayFactor' instead of BIDS-required 'DecayCorrectionFactor' for
        ease-of-use with NIL data. Should be changed in the future.
    *   (stitch_broken_scans) Currently writes intermediate files even if output_image_path is None.
    *   (suvr) Allow list to be passed as ref_region to use multiple regions together as a reference region (i.e. left
        and right cerebellum gray matter).

"""
import os
import pathlib
import datetime
import tempfile
import ants
import nibabel
import numpy as np
from scipy.ndimage import center_of_mass

from ..utils.useful_functions import weighted_series_sum
from ..utils import image_io, math_lib
from ..preproc.decay_correction import undo_decay_correction, decay_correct

def stitch_broken_scans(input_image_path: str,
                        output_image_path: str,
                        noninitial_image_paths: list[str]) -> ants.ANTsImage:
    """
    'Stitch' together 2 or more images from one session into a single image.

    This function takes multiple images (4D) from a single PET session in which the scan had to pause in the middle (a
    'broken scan'), recomputes decay corrections for all noninitial images using the correct TimeZero (TimeZero for the
    first image), then combines all the data into a single file to write (unless output_image_path is None, in which
    case the function will pass the ANTsImage object.

    Important: All noninitial images must be registered to the first image prior to calling this function.

    Args:
        input_image_path (str): Path to the initial image captured during PET session. 'TimeZero' from this image will be considered
            as the true value to correct the rest of the images to.
        output_image_path (str): Path to which the stitched image will be written. If None, no file will be written.
        noninitial_image_paths (list[str]): Path(s) to 1 or more additional images containing data from broken sections
            of the PET session. Note that all images must be registered to the first (input_image_path).

    Returns:
        ants.ANTsImage: stitched image
    """

    initial_image = ants.image_read(filename=input_image_path)
    initial_image_data = initial_image.numpy()
    initial_image_metadata = image_io.load_metadata_for_nifti_with_same_filename(image_path=input_image_path)
    noninitial_image_metadata_dicts = [image_io.load_metadata_for_nifti_with_same_filename(image_path=path)
                                       for path in noninitial_image_paths]

    try:
        noninitial_time_zeroes = [meta['TimeZero'] for meta in noninitial_image_metadata_dicts]
        actual_time_zero = initial_image_metadata['TimeZero']
    except KeyError:
        raise KeyError(f'.json sidecar for one of your input images does not contain required BIDS key "TimeZero". '
                       f'Aborting...')

    initial_scan_time = datetime.time.fromisoformat(actual_time_zero)
    placeholder_date = datetime.date.today()
    initial_scan_datetime = datetime.datetime.combine(date=placeholder_date,
                                                      time=initial_scan_time)
    noninitial_scan_times = [datetime.time.fromisoformat(t) for t in noninitial_time_zeroes]
    noninitial_scan_datetimes = [datetime.datetime.combine(date=placeholder_date, time=scan_time)
                                 for scan_time in noninitial_scan_times]

    times_since_timezero = [t - initial_scan_datetime for t in noninitial_scan_datetimes]

    for t_d, additional_image_metadata in zip(times_since_timezero,noninitial_image_metadata_dicts):
        original_frame_times_start = additional_image_metadata['FrameTimesStart']
        additional_image_metadata['FrameTimesStart'] = [t+t_d.total_seconds() for t in original_frame_times_start]
        additional_image_metadata['TimeZero'] = actual_time_zero

    corrected_arrays = [initial_image_data]
    new_metadata = initial_image_metadata

    for additional_image_path, metadata in zip(noninitial_image_paths,noninitial_image_metadata_dicts):

        original_path = pathlib.Path(additional_image_path)
        original_stem = original_path.stem
        split_stem = original_stem.split("_")
        split_stem.insert(-1, "desc-nodecaycorrect")
        new_stem = "_".join(split_stem)
        new_path = str(original_path).replace(original_stem, new_stem)

        undo_decay_correction(input_image_path=additional_image_path,
                              output_image_path=new_path,
                              metadata_dict=metadata)

        corrected_image_path = new_path.replace("desc-nodecaycorrect", "desc-decayredone")
        corrected_image = decay_correct(input_image_path=new_path,
                                        output_image_path=corrected_image_path)

        corrected_arrays.append(corrected_image.numpy())
        updated_metadata = image_io.load_metadata_for_nifti_with_same_filename(image_path=corrected_image_path)
        new_metadata['FrameTimesStart'].extend(updated_metadata['FrameTimesStart'])
        new_metadata['FrameReferenceTime'].extend(updated_metadata['FrameReferenceTime'])
        new_metadata['FrameDuration'].extend(updated_metadata['FrameDuration'])
        new_metadata['DecayFactor'].extend(updated_metadata['DecayFactor'])
        new_metadata['ImageDecayCorrected'] = updated_metadata['ImageDecayCorrected']
        new_metadata['ImageDecayCorrectionTime'] = updated_metadata['ImageDecayCorrectionTime']

    stitched_image_array = np.concatenate(corrected_arrays, axis=3)

    stitched_image = ants.from_numpy(data=stitched_image_array,
                                     origin=initial_image.origin,
                                     spacing=initial_image.spacing,
                                     direction=initial_image.direction)

    if output_image_path is not None:
        ants.image_write(image=stitched_image,
                         filename=output_image_path)
        image_io.write_dict_to_json(meta_data_dict=new_metadata,
                                    out_path=image_io.gen_meta_data_filepath_for_nifti(nifty_path=output_image_path))

    return stitched_image

def crop_image(input_image_path: str,
               out_image_path: str,
               x_dim: int=256,
               y_dim: int=256):
    """
    Crops an image in the X and Y axes to exclude voxels outside of the head. This is done to
    reduce the size of the image for faster processing, while preserving scientifically
    valuable information. Preserves dimension along Z and time axes.

    The returned, cropped image is centered on the "center of mass" computed using 
    :py:func:`scipy.ndimage.center_of_mass`. If the image is 3D, the center of mass is computed
    directly on the image array. If the image is 4D, the image is first averaged over the time axis
    before computing the center of mass.

    Args:
        input_image_path (str): Path to input image to be cropped.
        out_image_path (str): Path to which cropped image is saved.
        x_dim (int): Size of the X axis of the returned image. Default value 256.
        y_dim (int): Size of the Y axis of the returned image. Default value 256.
    
    Returns:
        cropped_image (nibabel.nifti1.Nifti1Image): The cropped image.
    """
    image = nibabel.load(input_image_path)
    image_np = image.get_fdata()

    if len(image_np.shape)<4:
        center = center_of_mass(image_np)
    else:
        image_mean = np.mean(image_np,axis=-1)
        center = center_of_mass(image_mean)

    center = np.round(center).astype('int')
    x_half = x_dim // 2
    y_half = y_dim // 2

    cropped_image = image.slicer[center[0]-x_half:center[0]+x_half,
                                 center[1]-y_half:center[1]+y_half]
    nibabel.save(cropped_image,out_image_path)
    image_io.safe_copy_meta(input_image_path=input_image_path,
                            out_image_path=out_image_path)
    return cropped_image


def rescale_image(input_image: ants.core.ANTsImage, rescale_constant: float, op: str = '/') -> ants.core.ANTsImage:
    r"""Rescales a 3D or 4D ANTsImage intensity values by performing division or
    multiplication with a given constant.

    This function supports two operations: dividing the input image by a
    rescale constant or multiplying it by the constant. Division is only
    allowed with a positive rescale constant to avoid invalid operations.
    The operation is applied element-wise across the image data.

    Args:
        input_image (ants.core.ANTsImage): Input image, given as an ANTsImage object.
        rescale_constant (float): The constant to rescale the image intensities. For division (`op="/"`),
            this value must be greater than zero.
        op (str, optional): Operation to perform, either `'/'` for division or `'*'` for
            multiplication. Default is `'/'`.

    Returns:
        ants.core.ANTsImage: The rescaled ANTsImage with updated intensity values.

    Raises:
        AssertionError: If `op` is not one of `'/'` or `'*'`.
        AssertionError: If division (`op="/"`) is requested, but `rescale_constant` is not greater than zero.

    Example:
        .. code-block:: python

            import ants
            from petpal.preproc.image_operations_4d import rescale_image

            # Load a sample ANTsImage
            input_img = ants.image_read('example_image.nii')

            # Rescale intensities by division with a constant (e.g., 2.0)
            rescaled_img = rescale_image(input_image=input_img, rescale_constant=2.0, op='/')

            # Rescale intensities by multiplication with a constant (e.g., 1.5)
            rescaled_img = rescale_image(input_image=input_img, rescale_constant=1.5, op='*')
    """
    assert op in ('/', '*'), "Operations supported by this function are `/` (division) or `*` (multiplication)."
    if op == '/':
        assert rescale_constant > 0, "Rescaling constant must be greater than zero."
        return input_image / rescale_constant
    else:
        return input_image * rescale_constant


def determine_motion_target(motion_target_option: str | tuple | list,
                            input_image_4d_path: str = None,
                            half_life: float = None) -> str:
    """
    Produce a motion target given the ``motion_target_option`` from a method
    running registrations on PET, i.e. :meth:`motion_correction` or
    :meth:`register_pet`.

    The motion target option can be a string or a tuple. If it is a string,
    then if this string is a file, use the file as the motion target.

    If it is the option ``weighted_series_sum``, then run
    :meth:`weighted_series_sum` and return the output path.

    If it is the option ``mean_image``, then compute the time-average of the
    4D-PET image.

    If it is a tuple, run a weighted sum on the PET series on a range of
    frames. The elements of the tuple are treated as times in seconds, counted
    from the time of the first frame, i.e. (0,300) would average all frames
    from the first to the frame 300 seconds later. If the two elements are the
    same, returns the one frame closest to the entered time.

    Args:
        motion_target_option (str | tuple | list): Determines how the method behaves,
            according to the above description. Can be a file, a method
            ('weighted_series_sum' or 'mean_image'), or a tuple range e.g. (0,600).
        input_image_4d_path (str): Path to the PET image. This is intended to
            be supplied by the parent method employing this function. Default
            value None.
        half_life (float): Half life of the radiotracer used in the image
            located at ``input_image_4d_path``. Only used if a calculation is
            performed.

    Returns:
        out_image_file (str): File to use as a target to compute
            transformations on.

    Raises:
        ValueError: If ``motion_target_option`` does not match an acceptable option, or if 
        ``half_life`` is not specifiedwhen ``motion_target_option`` is not 'mean_image'
        TypeError: If start and end time are incompatible with ``float`` type.
    """
    if motion_target_option != 'mean_image' and half_life is None:
        raise ValueError('half_life must be specified if not using "mean_image" for motion_target_option')

    if isinstance(motion_target_option, str):
        if os.path.exists(motion_target_option):
            return motion_target_option

        if motion_target_option == 'weighted_series_sum':
            out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
            weighted_series_sum(input_image_4d_path=input_image_4d_path,
                                out_image_path=out_image_file,
                                half_life=half_life,
                                verbose=False)
            return out_image_file

        if motion_target_option == 'mean_image':
            out_image_file = tempfile.mkstemp(suffix='_mean.nii.gz')[1]
            input_img = ants.image_read(input_image_4d_path)
            mean_img = get_average_of_timeseries(input_image=input_img)
            ants.image_write(image=mean_img,filename=out_image_file)
            return out_image_file

        raise ValueError("motion_target_option did not match a file or 'weighted_series_sum'")

    if isinstance(motion_target_option, (list, tuple)):

        start_time = motion_target_option[0]
        end_time = motion_target_option[1]

        try:
            float(start_time)
            float(end_time)
        except Exception as exc:
            raise TypeError('Start time and end time of calculation must be '
                            'able to be cast into float! Provided values are '
                            f"{start_time} and {end_time}.") from exc

        out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
        weighted_series_sum(input_image_4d_path=input_image_4d_path,
                            out_image_path=out_image_file,
                            half_life=half_life,
                            verbose=False,
                            start_time=float(start_time),
                            end_time=float(end_time))

        return out_image_file

    raise ValueError('motion_target_option did not match str or tuple type.')


def brain_mask(input_image_4d_path: str,
               out_image_path: str,
               atlas_image_path: str,
               atlas_mask_path: str,
               motion_target_option='mean_image',
               half_life: float=None):
    """
    Create a brain mask for a PET image. Create target PET image, which is then warped to a
    provided anatomical atlas. The transformation to atlas space is then applied to transform a
    provided mask in atlas space into PET space. This mask can then by used in various operations.

    Args:
        input_image_4d_path (str): Path to input 4D PET image.
        out_image_path (str): Path to which brain mask in PET space is written.
        atlas_image_path (str): Path to anatomical atlas image.
        atlas_mask_path (str): Path to brain mask in atlas space.
        motion_target_option: Used to determine 3D target in PET space. Default 'mean_image'.
    
    Note:
        Requires access to an anatomical atlas or scan with a corresponding brain mask on said
        anatomical data. FSL users can use the MNI152 atlas and mask available at 
        $FSLDIR/data/standard/.
    """
    atlas = ants.image_read(atlas_image_path)
    atlas_mask = ants.image_read(atlas_mask_path)
    pet_ref = ants.image_read(determine_motion_target(
        motion_target_option=motion_target_option,
        input_image_4d_path=input_image_4d_path,
        half_life=half_life
    ))
    xfm = ants.registration(
        fixed=atlas,
        moving=pet_ref,
        type_of_transform='SyN'
    )
    mask_on_pet = ants.apply_transforms(
        fixed=pet_ref,
        moving=atlas_mask,
        transformlist=xfm['invtransforms'],
        interpolator='nearestNeighbor'
    )
    mask = mask_on_pet.get_mask()
    ants.image_write(image=mask,filename=out_image_path)

def extract_mean_roi_tac_from_nifti_using_segmentation(input_image_4d_numpy: np.ndarray,
                                                       segmentation_image_numpy: np.ndarray,
                                                       region: int,
                                                       verbose: bool,
                                                       with_std: bool=False) -> np.ndarray:
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
        with_std (bool): If True, returns mean and standard deviation as a tuple. If False,
            returns mean alone. Default False.

    Returns:
        tac_out (np.ndarray): Mean of PET image within regions for each frame in 4D PET series.

    Raises:
        ValueError: If the segmentation image and PET image have different
            sampling.
    """

    pet_image_4d = input_image_4d_numpy
    if len(pet_image_4d.shape)==4:
        num_frames = pet_image_4d.shape[3]
    else:
        num_frames = 1
    seg_image = segmentation_image_numpy

    if seg_image.shape[:3]!=pet_image_4d.shape[:3]:
        raise ValueError('Mis-match in image shape of segmentation image '
                         f'({seg_image.shape}) and PET image '
                         f'({pet_image_4d.shape[:3]}). Consider resampling '
                         'segmentation to PET or vice versa.')

    if verbose:
        print(f'Running TAC for region index {region}')
    masked_voxels = (seg_image > region - 0.1) & (seg_image < region + 0.1)
    masked_image = pet_image_4d[masked_voxels].reshape((-1, num_frames))
    tac_out = np.mean(masked_image, axis=0)

    if with_std:
        tac_std = np.std(masked_image, axis=0)
        return tac_out, tac_std

    return tac_out


def threshold(input_image_numpy: np.ndarray,
              lower_bound: float=-np.inf,
              upper_bound: float=np.inf):
    """
    Threshold an image above and/or below a pair of values.
    """
    bounded_image = np.zeros(input_image_numpy.shape)
    bounded_image_where = (input_image_numpy > lower_bound) & (input_image_numpy < upper_bound)
    bounded_image[bounded_image_where] = input_image_numpy[bounded_image_where]
    return bounded_image


def binarize_image_with_threshold(input_image_numpy: np.ndarray,
                                  lower_bound: float=-np.inf,
                                  upper_bound: float=np.inf):
    """
    Threshold an image above and/or below a pair of values, and return a binary mask.

    Args:
        input_image_numpy (np.ndarray): Input image data to binarize with threshold (upper and/or lower).
        lower_bound (float): Lower bound of the threshold.
        upper_bound (float): Upper bound of the threshold.

    Returns:
        bounded_image (np.ndarray): Binary mask of original image where voxels within threshold are 1, and 0 elsewhere.
    """
    bounded_image = np.zeros(input_image_numpy.shape)
    bounded_image_where = (input_image_numpy > lower_bound) & (input_image_numpy < upper_bound)
    bounded_image[bounded_image_where] = 1
    return bounded_image


def get_average_of_timeseries(input_image: ants.ANTsImage):
    """
    Get average of a 4D ANTsImage and return as a 3D ANTsImage.
    """
    assert len(input_image.shape) == 4, "Input image must be 4D"
    mean_array = input_image.mean(axis=-1)
    mean_image = ants.from_numpy(data=mean_array,
                                 origin=input_image.origin[:-1],
                                 spacing=input_image.spacing[:-1],
                                 direction=input_image.direction[:-1,:-1])
    return mean_image


def suvr(input_image_path: str,
         out_image_path: str | None,
         segmentation_image_path: str,
         ref_region: int,
         verbose: bool=False) -> ants.ANTsImage:
    """
    Computes an ``SUVR`` (Standard Uptake Value Ratio) by taking the average of
    an input image within a reference region, and dividing the input image by
    said average value.

    Args:
        input_image_path (str): Path to 3D weighted series sum or other
            parametric image on which we compute SUVR.
        out_image_path (str): Path to output image file which is written to. If None, no output is written.
        segmentation_image_path (str): Path to segmentation image, which we use
            to compute average uptake value in the reference region.
        ref_region (int): Region number mapping to the reference region in the
            segmentation image.
        verbose (bool): Set to ``True`` to output processing information. Default is False.

    Returns:
        ants.ANTsImage: SUVR parametric image
    """
    pet_img = ants.image_read(filename=input_image_path)
    pet_arr = pet_img.numpy()
    segmentation_img = ants.image_read(filename=segmentation_image_path,
                                        pixeltype='unsigned int')
    segmentation_arr = segmentation_img.numpy()

    if len(pet_arr.shape)!=3:
        raise ValueError("SUVR input image is not 3D. If your image is dynamic, try running 'weighted_series_sum'"
                         " first.")

    ref_region_avg = extract_mean_roi_tac_from_nifti_using_segmentation(input_image_4d_numpy=pet_arr,
                                                                        segmentation_image_numpy=segmentation_arr,
                                                                        region=ref_region,
                                                                        verbose=verbose)
    
    suvr_arr = pet_arr / ref_region_avg[0]

    out_img = ants.from_numpy_like(data=suvr_arr,
                                     image=pet_img)

    if out_image_path is not None:
        ants.image_write(image=out_img,
                         filename=out_image_path)
        image_io.safe_copy_meta(input_image_path=input_image_path,
                                out_image_path=out_image_path)

    return out_img


def gauss_blur(input_image_path: str,
               blur_size_mm: float,
               out_image_path: str,
               verbose: bool,
               use_fwhm: bool=True):
    """
    Blur an image with a 3D Gaussian kernal of a provided size in mm. Extracts
    Gaussian sigma from provided blur size, and voxel sizes in the image
    header. :py:func:`scipy.ndimage.gaussian_filter` is used to apply blurring.
    Uses wrapper around :meth:`gauss_blur_computation`.
    
    Args:
        input_image_path (str): Path to 3D or 4D input image to be blurred.
        blur_size_mm (float): Sigma of the Gaussian kernal in mm.
        out_image_path (str): Path to save the blurred output image.
        verbose (bool): Set to ``True`` to output processing information.
        use_FWHM (bool): If ``True``, ``blur_size_mm`` is interpreted as the
            FWHM of the Gaussian kernal, rather than the standard deviation.

    Returns:
        out_image (nibabel.nifti1.Nifti1Image): Blurred image in nibabel format.
    """
    input_nibabel = nibabel.load(filename=input_image_path)
    input_image = input_nibabel.get_fdata()
    input_zooms = input_nibabel.header.get_zooms()

    blur_image = math_lib.gauss_blur_computation(input_image=input_image,
                                                 blur_size_mm=blur_size_mm,
                                                 input_zooms=input_zooms,
                                                 use_fwhm=use_fwhm)

    out_image = nibabel.nifti1.Nifti1Image(dataobj=blur_image,
                                           affine=input_nibabel.affine,
                                           header=input_nibabel.header)
    nibabel.save(img=out_image,filename=out_image_path)

    image_io.safe_copy_meta(input_image_path=input_image_path,out_image_path=out_image_path)

    if verbose:
        print(f'Blurred image saved to {out_image_path}.')

    return out_image


def roi_tac(input_image_4d_path: str,
            roi_image_path: str,
            region: int,
            out_tac_path: str,
            verbose: bool,
            time_frame_keyword: str = 'FrameReferenceTime'):
    """
    Function to write Tissue Activity Curves for a single region, given a mask,
    4D PET image, and region mapping. Computes the average of the PET image 
    within each region. Writes a tsv table with region name, frame start time,
    and mean value within region.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.load_metadata_for_nifti_with_same_filename(input_image_4d_path)
    tac_extraction_func = extract_mean_roi_tac_from_nifti_using_segmentation
    pet_numpy = nibabel.load(input_image_4d_path).get_fdata()
    seg_numpy = nibabel.load(roi_image_path).get_fdata()


    extracted_tac = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                        segmentation_image_numpy=seg_numpy,
                                        region=region,
                                        verbose=verbose)
    region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
    header_text = 'mean_activity'
    np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')


class SimpleAutoImageCropper(object):
    r"""
    Class for automatically cropping 3D or 4D medical images based on pixel intensity thresholds.

    This class provides functionality to load a medical image, determine the meaningful regions
    by thresholding, and crop the image to remove regions outside these boundaries.
    It also supports copying metadata from the original image.

    Attributes:
        input_image_path (str): The file path to the input image.
        out_image_path (str): The file path to save the cropped image.
        thresh (float): The threshold value used to determine the boundaries.
        verbose (bool): If True, prints information about image shapes.
        input_img_obj (nibabel.Nifti1Image): The loaded input image object.
        crop_img_obj (nibabel.Nifti1Image): The cropped image object.

    Example:
        
        .. code-block:: python
        
            from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
            cropper = SimpleAutoImageCropper(
                input_image_path='path/to/input_image_path.nii',
                out_image_path='path/to/output_image.nii',
                thresh_val=0.01,
                verbose=True,
                copy_metadata=True
            )

    See Also:
        - :meth:`get_cropped_image`
        - :meth:`get_index_pairs_for_all_dims`
        - :meth:`get_left_and_right_boundary_indices_for_threshold`
        - :meth:`gen_line_profile`
        
        
    """
    def __init__(self,
                 input_image_path: str,
                 out_image_path: str,
                 thresh_val: float = 1.0e-2,
                 verbose: bool = True,
                 copy_metadata: bool = True
                 ):
        r"""
        Initializes the SimpleAutoImageCropper with input image path, output image path, and other
        parameters.

        Loads the input image, generates the cropped image using the specified threshold, and saves
        it to the output path.

        Args:
            input_image_path (str): The file path to the input image.
            out_image_path (str): The file path to save the cropped image.
            thresh_val (float, optional): The threshold value used to determine the boundaries.
                Must be less than 0.5. Defaults to 1e-2.
            verbose (bool, optional): If True, prints information about image shapes. Defaults to
                True.
            copy_metadata (bool, optional): If True, copies metadata from the original image to the
                cropped image. Defaults to True.

        Raises:
            AssertionError: If the `thresh_val` is not less than 0.5.

        Example:
            
            .. code-block:: python

                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
                cropper = SimpleAutoImageCropper(
                    input_image_path='path/to/input_image_path.nii',
                    out_image_path='path/to/output_image.nii',
                    thresh_val=0.01,
                    verbose=True,
                    copy_metadata=True
                )

        """
        self.input_image_path = input_image_path
        self.out_image_path = out_image_path
        self.thresh = thresh_val
        self.verbose = verbose
        self.input_img_obj = nibabel.load(self.input_image_path)
        self.crop_img_obj = self.get_cropped_image(img_obj=self.input_img_obj, thresh=self.thresh)

        nibabel.save(filename=self.out_image_path, img=self.crop_img_obj)
        if copy_metadata:
            image_io.safe_copy_meta(input_image_path=self.input_image_path,
                                    out_image_path=self.out_image_path)

        if verbose:
            print(f"(info): Input image has shape:  {self.input_img_obj.shape}")
            print(f"(info): Output image has shape: {self.crop_img_obj.shape}")


    @staticmethod
    def gen_line_profile(img_arr: np.ndarray, dim: str = 'x'):
        r"""
        Generates a line profile by averaging the pixel intensities along specified dimensions.

        This function computes the mean pixel intensities along a specified dimension (x, y, or z)
        of a 3D or 4D image array.

        Args:
            img_arr (np.ndarray): The input image array.
            dim (str, optional): The dimension along which to compute the line profile.
                                 Must be one of 'x', 'y', or 'z'. Case-insensitive. Defaults to 'x'.

        Returns:
            np.ndarray: The computed line profile as a 1D array.

        Raises:
            AssertionError: If `dim` is not one of 'x', 'y', or 'z'.

        Example:
            
            .. code-block:: python
            
                import numpy as np
                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
                img_arr = np.random.rand(100, 100, 100)  # Example 3D array
                x_profile = SimpleAutoImageCropper.gen_line_profile(img_arr=img_arr, dim='x')
                print(x_profile)
        
        """
        tmp_dim = dim.lower()
        assert tmp_dim in ['x', 'y', 'z']
        if tmp_dim == 'x':
            return np.mean(img_arr, axis=(1, 2))
        if tmp_dim == 'y':
            return np.mean(img_arr, axis=(0, 2))
        if tmp_dim == 'z':
            return np.mean(img_arr, axis=(0, 1))

    @staticmethod
    def get_left_and_right_boundary_indices_for_threshold(line_prof: np.ndarray,
                                                          thresh: float = 1e-2):
        r"""
        Determines the left and right boundary indices above a threshold in a line profile.

        This function identifies the indices where the normalized line profile crosses the 
        specified threshold value, indicating the boundaries of the region of interest.

        Args:
            line_prof (np.ndarray): The input line profile as a 1D array.
            thresh (float, optional): The threshold value for determining boundaries. Must be less
                than 0.5. Defaults to 1e-2.

        Returns:
            tuple: A tuple containing the left and right boundary indices (left_index, right_index).

        Raises:
            AssertionError: If the `thresh` value is not less than 0.5.

        Example:
            
            .. code-block:: python
            
                import numpy as np
                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper as Crop

                line_prof = np.random.rand(100)  # Example normalized line profile
                boundaries = Crop.get_left_and_right_boundary_indices_for_threshold
                left_index, right_index = boundaries(line_prof=line_prof, thresh=0.01)
                print(left_index, right_index)
        
        """
        assert thresh < 0.5
        norm_prof = line_prof / np.max(line_prof)
        l_ind, r_ind = np.argwhere(norm_prof > thresh).T[0][[0, -1]]
        return l_ind, r_ind

    @staticmethod
    def get_index_pairs_for_all_dims(img_obj: nibabel.Nifti1Image, thresh: float = 1e-2):
        r"""
        Gets the boundary indices for each dimension of the input image based on a threshold value.

        This function computes the left and right boundary indices for all dimensions (x, y, z)
        by generating line profiles and applying a threshold to identify meaningful regions.

        Args:
            img_obj (nibabel.Nifti1Image): The input NIfTI image object.
            thresh (float, optional): The threshold value used to determine the boundaries.
                                      Must be less than 0.5. Defaults to 1e-2.

        Returns:
            tuple: A tuple of boundary index pairs for each dimension, formatted as
                   ((x_left, x_right), (y_left, y_right), (z_left, z_right)).

        Raises:
            AssertionError: If the `thresh` value is not less than 0.5.
            
        See Also:
            - :meth:`get_index_pairs_for_all_dims`

        Example:
            
            .. code-block:: python
            
                import nibabel as nib
                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
                input_image_path = 'path/to/input_image_path.nii'
                img_obj = nib.load(input_image_path)
    
                boundaries = SimpleAutoImageCropper.get_index_pairs_for_all_dims(img_obj=img_obj,
                                                                                 thresh=0.01)
                print(boundaries)
        
        """
        if len(img_obj.shape) > 3:
            tmp_data = np.mean(img_obj.get_fdata(), axis=-1)
        else:
            tmp_data = img_obj.get_fdata()

        prof_func = SimpleAutoImageCropper.gen_line_profile
        index_func = SimpleAutoImageCropper.get_left_and_right_boundary_indices_for_threshold

        x_line_prof = prof_func(img_arr=tmp_data, dim='x')
        x_left, x_right = index_func(line_prof=x_line_prof, thresh=thresh)

        y_line_prof = prof_func(img_arr=tmp_data, dim='y')
        y_left, y_right = index_func(line_prof=y_line_prof, thresh=thresh)

        z_line_prof = prof_func(img_arr=tmp_data, dim='z')
        z_left, z_right = index_func(line_prof=z_line_prof, thresh=thresh)

        return (x_left, x_right), (y_left, y_right), (z_left, z_right)

    @staticmethod
    def get_cropped_image(img_obj: nibabel.Nifti1Image, thresh: float = 1e-2):
        r"""
        Crops the input medical image based on a threshold value.

        This function determines the boundaries of the meaningful regions in the input image
        by thresholding and then crops the image to remove regions outside these boundaries.

        Args:
            img_obj (nibabel.Nifti1Image): The input NIfTI image object to be cropped.
            thresh (float, optional): The threshold value used to determine the boundaries.
                                      Must be less than 0.5. Defaults to 1e-2.

        Returns:
            nibabel.Nifti1Image: The cropped NIfTI image object.

        Raises:
            AssertionError: If the `thresh` value is not less than 0.5.
            
        See Also:
            - :meth:`get_index_pairs_for_all_dims`
            - :meth:`get_left_and_right_boundary_indices_for_threshold`
            - :meth:`gen_line_profile`

        Example:
            
            .. code-block:: python
            
                import nibabel as nib
                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
                input_image_path = 'path/to/input_image_path.nii'
                img_obj = nib.load(input_image_path)
    
                cropped_img = SimpleAutoImageCropper.get_cropped_image(img_obj=img_obj, thresh=0.01)
                nib.save(cropped_img, 'path/to/output_image.nii')
        
        """
        (x_l, x_r), (y_l, y_r), (z_l, z_r) = SimpleAutoImageCropper.get_index_pairs_for_all_dims(img_obj=img_obj,
                                                                                                 thresh=thresh)

        return img_obj.slicer[x_l:x_r, y_l:y_r, z_l:z_r, ...]
