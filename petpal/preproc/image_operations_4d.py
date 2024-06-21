"""
The 'image_operations_4d' module provides several functions used to do preprocessing
on 4D PET imaging series. These functions typically take one or more paths to imaging
data in NIfTI format, and save modified data to a NIfTI file, and may return the
modified imaging array as output.

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
from scipy.interpolate import interp1d
import nibabel
import numpy as np
from ..utils import image_io, math_lib

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

    Where :math:`f_i,t_i,d_i` are the i-th frame, frame duration, and decay correction factor of
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
            If equal to ``start_time``, one frame at start_time is used. Default value -1.

    Returns:
        np.ndarray: 3D image array, in the same space as the input, with the weighted sum calculation applied.

    Raises:
        ValueError: If ``half_life`` is zero or negative.
    """
    if half_life <= 0:
        raise ValueError('(ImageOps4d): Radioisotope half life is zero or negative.')
    pet_meta = image_io.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
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

    wsc = math_lib.weighted_sum_computation
    image_weighted_sum = wsc(frame_duration=frame_duration_adjusted,
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

    image_io.safe_copy_meta(input_image_path=input_image_4d_path,
                            out_image_path=out_image_path)

    return pet_sum_image


def extract_tac_from_nifty_using_mask(input_image_4d_numpy: np.ndarray,
                                      segmentation_image_numpy: np.ndarray,
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

    pet_image_4d = input_image_4d_numpy
    if len(pet_image_4d.shape)==4:
        num_frames = pet_image_4d.shape[3]
    else:
        num_frames = 1
    seg_image = segmentation_image_numpy

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
    pet_nibabel = nibabel.load(filename=input_image_path)
    pet_image = pet_nibabel.get_fdata()
    seg_nibabel = nibabel.load(filename=segmentation_image_path)
    seg_image = seg_nibabel.get_fdata()

    if len(pet_image.shape)!=3:
        raise ValueError("SUVR input image is not 3D. If your image is dynamic"
                         ", try running 'weighted_series_sum' first.")

    ref_region_avg = extract_tac_from_nifty_using_mask(input_image_4d_numpy=pet_image,
                                                       segmentation_image_numpy=seg_image,
                                                       region=ref_region,
                                                       verbose=verbose)

    suvr_image = pet_image / ref_region_avg[0]

    out_image = nibabel.nifti1.Nifti1Image(dataobj=suvr_image,
                                           affine=pet_nibabel.affine,
                                           header=pet_nibabel.header)
    nibabel.save(img=out_image,filename=out_image_path)

    image_io.safe_copy_meta(input_image_path=input_image_path,
                            out_image_path=out_image_path)

    return out_image


def gauss_blur(input_image_path: str,
               blur_size_mm: float,
               out_image_path: str,
               verbose: bool,
               use_FWHM: bool=True):
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
                                                 use_FWHM=use_FWHM)

    out_image = nibabel.nifti1.Nifti1Image(dataobj=blur_image,
                                           affine=input_nibabel.affine,
                                           header=input_nibabel.header)
    nibabel.save(img=out_image,filename=out_image_path)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.load_metadata_for_nifty_with_same_filename(input_image_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)

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

    pet_meta = image_io.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    tac_extraction_func = extract_tac_from_nifty_using_mask
    pet_numpy = nibabel.load(input_image_4d_path).get_fdata()
    seg_numpy = nibabel.load(roi_image_path).get_fdata()


    extracted_tac = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                        segmentation_image_numpy=seg_numpy,
                                        region=region,
                                        verbose=verbose)
    region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
    header_text = 'mean_activity'
    np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')


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

    pet_meta = image_io.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
    regions_abrev = label_map['abbreviation']
    regions_map = label_map['mapping']

    tac_extraction_func = extract_tac_from_nifty_using_mask
    pet_numpy = nibabel.load(input_image_4d_path).get_fdata()
    seg_numpy = nibabel.load(segmentation_image_path).get_fdata()

    for i, _maps in enumerate(label_map['mapping']):
        extracted_tac = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                            segmentation_image_numpy=seg_numpy,
                                            region=int(regions_map[i]),
                                            verbose=verbose)
        region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
        header_text = f'{time_frame_keyword}\t{regions_abrev[i]}_mean_activity'
        out_tac_path = os.path.join(out_tac_dir, f'tac-{regions_abrev[i]}.tsv')
        np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')
