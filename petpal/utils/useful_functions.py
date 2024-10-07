"""
Module to handle abstracted functionalities
"""
import os

from petpal.utils import image_io, math_lib

import nibabel
import numpy as np
from scipy.interpolate import interp1d

FULL_NAME = [
    'Background',
    'CorticalGrayMatter',
    'SubcorticalGrayMatter',
    'GrayMatter',
    'gm',
    'WhiteMatter',
    'wm',
    'CerebrospinalFluid',
    'Bone',
    'SoftTissue',
    'Nonbrain',
    'Lesion',
    'Brainstem',
    'Cerebellum'
]
SHORT_NAME = [
    'BG',
    'CGM',
    'SGM',
    'GM',
    'GM',
    'WM',
    'WM',
    'CSF',
    'B',
    'ST',
    'NB',
    'L',
    'BS',
    'CBM'
]


def make_path(paths: list[str]):
    """
    Creates a new path in local system by joining paths, and making any new directories, if
    necessary.

    Args:
        paths (list[str]): A list containing strings to be joined as a path in the system
            directory.

    Note:
        If the final string provided includes a period '.' (a proxy for checking if the path is a 
        file name) this method will result in creating the folder above the last provided string in
        the list.
    """
    end_dir = paths[-1]
    if end_dir.find('.') == -1:
        out_path = os.path.join(paths)
    else:
        out_path = os.path.join(paths[:-1])
    os.makedirs(out_path,exist_ok=True)


def abbreviate_region(region_name: str):
    """
    Converts long region names to their associated abbreviations.
    """
    name_out = region_name.replace('-','').replace('_','')
    for i,_d in enumerate(FULL_NAME):
        full_name = FULL_NAME[i]
        short_name = SHORT_NAME[i]
        name_out = name_out.replace(full_name,short_name)
    return name_out


def build_label_map(region_names: list[str]):
    """
    Builds a BIDS compliant label map. Loop through CTAB and convert names to
    abbreviations using :meth:`abbreviate_region`
    """
    abbreviated_names = list(map(abbreviate_region,region_names))
    return abbreviated_names


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
        np.ndarray: 3D image array, in the same space as the input, with the weighted sum
            calculation applied.

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
        raise ValueError("Neither 'DecayCorrectionFactor' nor 'DecayFactor' exist in meta-data "
                         "file")

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
