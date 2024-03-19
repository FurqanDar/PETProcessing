"""
Library for math functions for use elsewhere.
"""
import numpy as np


def weighted_sum_computation(
        image_frame_duration: np.ndarray,
        half_life: float,
        pet_series: np.ndarray,
        image_frame_start: np.ndarray,
        image_decay_correction: np.ndarray
        ):
    """
    Sum a 4D image series weighted based on time and re-corrected for decay correction.
    Credit to Avi Snyder who wrote the original version of this code in C.

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
    """
    decay_constant = np.log(2) / half_life
    image_total_duration = np.sum(image_frame_duration)
    total_decay    = decay_constant * image_total_duration / \
        (1-np.exp(-1*decay_constant*image_total_duration)) / \
            np.exp(-1*decay_constant*image_frame_start[0])

    pet_series_scaled = pet_series[:,:,:] \
        * image_frame_duration \
        / image_decay_correction
    pet_series_sum_scaled = np.sum(pet_series_scaled,axis=3)
    image_weighted_sum = pet_series_sum_scaled * total_decay / image_total_duration
    return image_weighted_sum
