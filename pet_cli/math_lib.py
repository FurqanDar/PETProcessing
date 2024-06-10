"""
Library for math functions for use elsewhere.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def weighted_sum_computation(frame_duration: np.ndarray,
                             half_life: float,
                             pet_series: np.ndarray,
                             frame_start: np.ndarray,
                             decay_correction: np.ndarray):
    """
    Weighted sum of a PET image based on time and re-corrected for decay correction.

    Args:
        image_frame_duration (np.ndarray): Duration of each frame in pet series
        half_life (float): Half life of tracer radioisotope in seconds.
        pet_series (np.ndarray): 4D PET image series, as a data array.
        image_frame_start (np.ndarray): Start time of each frame in pet series,
            measured with respect to scan TimeZero.
        image_decay_correction (np.ndarray): Decay correction factor that scales
            each frame in the pet series. 

    Returns:
        image_weighted_sum (np.ndarray): 3D PET image computed by reversing decay correction
            on the PET image series, scaling each frame by the frame duration, then re-applying
            decay correction and scaling the image to the full duration.

    See Also:
        * :meth:`pet_cli.image_operations_4d.weighted_series_sum`: Function where this is implemented.

    """
    decay_constant = np.log(2.0) / half_life
    image_total_duration = np.sum(frame_duration)
    total_decay = decay_constant * image_total_duration
    total_decay /= 1.0 - np.exp(-1.0 * decay_constant * image_total_duration)
    total_decay /= np.exp(-1 * decay_constant * frame_start[0])
    
    pet_series_scaled = pet_series[:, :, :] * frame_duration / decay_correction
    pet_series_sum_scaled = np.sum(pet_series_scaled, axis=3)
    image_weighted_sum = pet_series_sum_scaled * total_decay / image_total_duration
    return image_weighted_sum


def gauss_blur_computation(input_image: np.ndarray,
                           blur_size_mm: float,
                           input_zooms: list,
                           use_FWHM: bool):
    """
    Applies a Gaussian blur to an array image. Function intended to be a
    wrapper to be applied by other methods.
    """
    if use_FWHM:
        blur_size = blur_size_mm / (2*np.sqrt(2*np.log(2)))
    else:
        blur_size = blur_size_mm

    sigma_x = blur_size / input_zooms[0]
    sigma_y = blur_size / input_zooms[1]
    sigma_z = blur_size / input_zooms[2]

    blur_image = gaussian_filter(input=input_image,
                                 sigma=(sigma_x,sigma_y,sigma_z),
                                 axes=(0,1,2))
    return blur_image
