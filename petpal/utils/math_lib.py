"""
Library for math functions for use elsewhere.
"""
import numpy as np


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
        * :meth:`petpal.image_operations_4d.weighted_series_sum`: Function where this is implemented.

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
