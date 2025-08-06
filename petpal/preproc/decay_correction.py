"""
Provides functions for undo-ing decay correction and recalculating it.

"""

import math

import ants
import numpy as np

from ..utils import image_io
from ..utils.scan_timing import ScanTimingInfo


def undo_decay_correction(input_image_path: str,
                          output_image_path: str,
                          metadata_dict: dict = None) -> ants.ANTsImage:
    """Uses decay factors from the metadata for an image to remove decay correction for each frame.

    This function expects to find decay factors in the .json sidecar file, or the metadata_dict, if given. If there are
    no decay factors (either under the key 'DecayFactor' or the BIDS-required 'DecayCorrectionFactor') listed, it may
    result in unexpected behavior. In addition to returning an ANTsImage containing the "decay uncorrected" data, the
    function writes an image to output_image_path, unless it is passed as 'None'.

    Args:
        input_image_path (str): Path to input (.nii.gz or .nii) image. A .json sidecar file should exist in the same
             directory as the input image.
        output_image_path (str): Path to output (.nii.gz or .nii) output image. If None, no image will be written.
        metadata_dict (dict, optional): Metadata dictionary to use instead of corresponding .json sidecar. If not
            specified (default behavior), function will try to use sidecar .json in the same directory as
            input_image_path.

    Returns:
        ants.ANTsImage: ANTsImage with decay correction reversed."""

    decay_corrected_image = ants.image_read(filename=input_image_path)

    if metadata_dict is not None:
        json_data = metadata_dict
    else:
        json_data = image_io.load_metadata_for_nifti_with_same_filename(image_path=input_image_path)

    frame_info = ScanTimingInfo.from_nifti(image_path=input_image_path)
    decay_factors = frame_info.decay

    uncorrected_image_numpy = decay_corrected_image.numpy()

    for frame_num, decay_factor in enumerate(decay_factors):
        uncorrected_image_numpy[..., frame_num] /= decay_factor

    uncorrected_image = ants.from_numpy_like(data=uncorrected_image_numpy,
                                             image=decay_corrected_image)

    if output_image_path is not None:
        ants.image_write(image=uncorrected_image,
                         filename=output_image_path)

        json_data['DecayFactor'] = list(np.ones_like(decay_factors))
        json_data['ImageDecayCorrected'] = "false"
        output_json_path = image_io.gen_meta_data_filepath_for_nifti(nifty_path=output_image_path)
        image_io.write_dict_to_json(meta_data_dict=json_data,
                                    out_path=output_json_path)

    return uncorrected_image


def decay_correct(input_image_path: str,
                  output_image_path: str) -> ants.ANTsImage:
    r"""Recalculate decay_correction for nifti image based on frame reference times.

    This function will compute frame reference times based on frame time starts and frame durations (both of which
    are required by BIDS. These reference times are used in the following equation to determine the decay factor for
    each frame. For more information, refer to Turku Pet Centre's materials at
    https://www.turkupetcentre.net/petanalysis/decay.html

    .. math::
        decay\_factor = \exp(\lambda*t)

    where :math:`\lambda=\log(2)/T_{1/2}` is the decay constant of the radio isotope and depends on its half-life and
    `t` is the frame's reference time with respect to TimeZero (ideally, injection time).

    Note: BIDS 'DecayCorrectionTime' is set to 0 (seconds w.r.t. TimeZero) for the image. If this assumption doesn't
        hold, be wary of downstream effects.

    Args:
        input_image_path (str): Path to input (.nii.gz or .nii) image. A .json sidecar file should exist in the same
             directory as the input image.
        output_image_path (str): Path to output (.nii.gz or .nii) output image.

    Returns:
        ants.ANTsImage: Decay-Corrected Image

    """
    half_life = image_io.get_half_life_from_nifti(image_path=input_image_path)

    json_data = image_io.load_metadata_for_nifti_with_same_filename(image_path=input_image_path)
    uncorrected_image = ants.image_read(filename=input_image_path)

    frame_info = ScanTimingInfo.from_nifti(image_path=input_image_path)
    frame_reference_times = np.asarray(frame_info.start + frame_info.duration / 2.0, float).tolist()

    original_decay_factors = frame_info.decay
    if np.any(original_decay_factors != 1):
        raise ValueError(f'Decay Factors other than 1 found in metadata for {input_image_path}. This likely means the '
                         f'image has not had its previous decay correction undone. Try running undo_decay_correction '
                         f'before running this function to avoid decay correcting an image more than once.')

    corrected_data = uncorrected_image.numpy()
    new_decay_factors = []
    for frame_num, frame_reference_time in enumerate(frame_reference_times):
        new_decay_factor = math.exp(((math.log(2) / half_life) * frame_reference_time))
        corrected_data[..., frame_num] *= new_decay_factor
        new_decay_factors.append(new_decay_factor)

    corrected_image = ants.from_numpy_like(data=corrected_data,
                                           image=uncorrected_image)

    if output_image_path is not None:
        ants.image_write(image=corrected_image,
                         filename=output_image_path)
        output_json_path = image_io.gen_meta_data_filepath_for_nifti(nifty_path=output_image_path)
        json_data['DecayFactor'] = new_decay_factors
        json_data['ImageDecayCorrected'] = "true"
        json_data['ImageDecayCorrectionTime'] = 0
        json_data['FrameReferenceTime'] = frame_reference_times
        image_io.write_dict_to_json(meta_data_dict=json_data,
                                    out_path=output_json_path)

    return corrected_image


def calculate_frame_decay_factor(frame_reference_time: float,
                                 half_life: float) -> float:
    frame_decay_factor = math.exp((math.log(2)/half_life)*frame_reference_time)
    return frame_decay_factor