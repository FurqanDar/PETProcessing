"""
This module contains a collection of functions to calculate an image-derived input function (IDIF) given a "necktangle".
"Necktangles" are a 3D rectangular region of interest over the neck to help identify the carotid PET signal.
This method was developed and provided for use in this software package by `Dr. Karl Friedrichsen <https://orcid.org/0000-0002-9233-1418>`_

Requires:
    The module relies on the :doc:`numpy <numpy:index>` module.

"""

import numpy as np


def single_threshold_idif_from_4d_pet_with_necktangle(pet_4d_data: np.ndarray,
                                                      carotid_necktangle_mask_3d_data: np.ndarray,
                                                      percentile: float,
                                                      bolus_start_frame: int = 3,
                                                      bolus_end_frame: int = 7) -> np.ndarray:
    """
    Applies the single bolus percentile IDIF method to calculate the time-activity curve (TAC) from PET data.

    Args:
        pet_4d_data (np.ndarray): 4-dimensional array representing the PET data.
        carotid_necktangle_mask_3d_data (np.ndarray): 3-dimensional array representing the carotid mask data.
        percentile (float): The percentile used as the threshold value for masking the data.
        bolus_start_frame (int, optional): The starting frame index for the bolus mean calculation. Defaults to 3.
        bolus_end_frame (int, optional): The ending frame index for the bolus mean calculation. Defaults to 7.

    Returns:
        tac (np.ndarray): The time-activity curve (TAC) calculated using the Karl simple percentile IDIF method.
    """
    bolus_mean_3d = average_across_4d_frames(pet_4d_data=pet_4d_data,
                                             start_frame=bolus_start_frame,
                                             end_frame=bolus_end_frame)
    if bolus_mean_3d.shape != carotid_necktangle_mask_3d_data.shape:
        raise ValueError("array1 and array2 must have the same shape.")

    carotid_masked_bolus_mean_3d = np.where(carotid_necktangle_mask_3d_data == 1, bolus_mean_3d, np.nan)

    threshold_value = np.nanpercentile(carotid_masked_bolus_mean_3d, percentile)
    bolus_mean_threshold_masked_3d = np.where(carotid_masked_bolus_mean_3d > threshold_value,
                                              carotid_masked_bolus_mean_3d, np.nan)
    bolus_mean_threshold_binary_mask_3d = np.where(bolus_mean_threshold_masked_3d >= 1000, 1, 0)

    pet_data_4d_masked = np.zeros(pet_4d_data.shape)
    for frame in range(pet_data_4d_masked.shape[3]):
        pet_data_4d_masked[:, :, :, frame] = np.where(bolus_mean_threshold_binary_mask_3d == 1,
                                                      pet_4d_data[:, :, :, frame], np.nan)

    tac = np.nanmean(pet_data_4d_masked, axis=(0, 1, 2))

    return tac


def average_across_4d_frames(pet_4d_data: np.ndarray,
                             start_frame: int,
                             end_frame: int) -> np.ndarray:
    """
    Calculates the mean frame across the specified range of frames in the 4-dimensional PET data.

    Args:
        pet_4d_data (np.ndarray): 4-dimensional array representing the PET data.
        start_frame (int): The starting frame index for the mean calculation.
        end_frame (int): The ending frame index for the mean calculation.

    Returns:
        mean_frame (np.ndarray): The mean frame calculated from the specified range of frames.
    """
    if start_frame < 0 or end_frame >= pet_4d_data.shape[3]:
        raise ValueError("Frame indices are out of bounds.")

    mean_frame = np.nanmean(pet_4d_data[:, :, :, start_frame:end_frame + 1], axis=3)

    return mean_frame


def get_frame_time_midpoints(frame_start_times: np.ndarray,
                             frame_duration_times: np.ndarray) -> np.ndarray:
    """
        Calculates the midpoint times of each frame based on the start times and duration times.

        Args:
            frame_start_times (np.ndarray): An array of frame start times.
            frame_duration_times (np.ndarray): An array of frame duration times.

        Returns:
            frame_midpoint_times (np.ndarray): An array of frame midpoint times.

        Raises:
            None.
        """
    frame_midpoint_times = (frame_start_times + (frame_duration_times / 2)).astype(float)
    return frame_midpoint_times


def load_fslmeants_to_numpy_3d(fslmeants_filepath: str) -> np.ndarray:
    """
    Loads `fslmeants <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PPIHowToRun?highlight=%28fslmeants%29>`_ (--show-all) data from a file and converts it into a 3D numpy array.

    Args:
        fslmeants_filepath (str): The filepath of the fslmeants data file.

    Returns:
        numpy_3d_array (np.ndarray): A 3D numpy array representing the fslmeants data.

    Raises:
        None.
    """
    data = np.loadtxt(fslmeants_filepath)
    x_coord_min = min(data[0].astype(int))
    y_coord_min = min(data[1].astype(int))
    z_coord_min = min(data[2].astype(int))
    x_dim = (max(data[0].astype(int)) - x_coord_min) + 1
    y_dim = (max(data[1].astype(int)) - y_coord_min) + 1
    z_dim = (max(data[2].astype(int)) - z_coord_min) + 1
    t_dim = data.shape[0] - 3
    numpy_3d_array = np.zeros((x_dim, y_dim, z_dim, t_dim), dtype=float)
    for location in range(data.shape[1]):
        x_coord = data[0, location].astype(int) - x_coord_min
        y_coord = data[1, location].astype(int) - y_coord_min
        z_coord = data[2, location].astype(int) - z_coord_min
        numpy_3d_array[x_coord, y_coord, z_coord, :] = data[3:, location]

    return numpy_3d_array


def double_threshold_idif_from_4d_pet_necktangle(necktangle_matrix: np.ndarray,
                                                 percentile: float,
                                                 frame_midpoint_times: np.ndarray) -> np.ndarray:
    """
    Computes the IDIF from a 4D PET necktangle matrix given a percentile for thresholding.
    This function finds the highest mean frame from the first 10 frames of the 4D PET, creates a mean 3D image of that frame, the one before it, and the one after it.
    Then, this function applies an automatic percentile thresholding of 90% to that mean image to generate a carotid mask.
    Finally, that carotid mask is applied to the 4D PET image, and the resulting 4D image undergoes percentile thresholding of the "percentile" value frame by frame to get the TAC.

    Args:
        necktangle_matrix (np.ndarray): A 4D numpy array representing the PET necktangle matrix.
        percentile (float): The percentile value to calculate the manual threshold.
        frame_midpoint_times (np.ndarray): An array of frame midpoint times.

    Returns:
        tac (np.ndarray): A 2D numpy array representing the time-activity curve (TAC) with frame midpoint times and manual thresholds.

    Raises:
        None.
    """
    first_ten_frames = necktangle_matrix[:, :, :, :10]
    frame_averages = np.nanmean(first_ten_frames, axis=(0, 1, 2))
    bolus_index = np.argmax(frame_averages)
    bolus_window_4d = necktangle_matrix[:, :, :, bolus_index - 1:bolus_index + 2]
    bolus_window_average_3d = np.nanmean(bolus_window_4d, axis=3)
    automatic_threshold_value = np.nanpercentile(bolus_window_average_3d, 90)
    automatic_threshold_mask_3d = np.where(bolus_window_average_3d > automatic_threshold_value, 1, np.nan)
    tac = np.zeros((2, necktangle_matrix.shape[3]))
    tac[0, :] = frame_midpoint_times
    for frame in range(tac.shape[1]):
        current_frame = necktangle_matrix[:, :, :, frame]
        automatic_masked_frame = np.where(automatic_threshold_mask_3d == 1, current_frame, np.nan)
        manual_threshold_value = np.nanpercentile(automatic_masked_frame, percentile)
        tac[1, frame] = manual_threshold_value

    return tac
