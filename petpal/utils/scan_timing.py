"""
Module to handle timing information of PET scans.
"""
import math
from dataclasses import dataclass
import numpy as np

from .image_io import load_metadata_for_nifti_with_same_filename

def calculate_frame_reference_time(frame_duration: float,
                                 frame_start: float,
                                 half_life: float): 
    decay_constant = math.log(2)/half_life
    frame_reference_time = frame_start + math.log((decay_constant*frame_duration)/(1-math.exp(-decay_constant*frame_duration)))/decay_constant
    return frame_reference_time


@dataclass
class ScanTimingInfo:
    """
    A data structure to represent and streamline access to timing information for image scans.

    This class encapsulates details about a scan's timing, including:
    - Start and end times of each scan frame.
    - Duration and center times of the frames.
    - Decay values (if applicable).

    Additionally, the class provides properties for easy conversion of timing values to minutes
    if the times are given in seconds and exceed a threshold (assumed to be 200.0 seconds).

    Attributes:
        duration (np.ndarray[float]): Array of frame durations.
        end (np.ndarray[float]): Array of frame end times.
        start (np.ndarray[float]): Array of frame start times.
        center (np.ndarray[float]): Array of frame center times (midpoints).
        decay (np.ndarray[float]): Array of decay coefficients for the scan frames.

    Properties:
        duration_in_mins (np.ndarray[float]):
            Returns the frame durations converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original durations.

        end_in_mins (np.ndarray[float]):
            Returns the frame end times converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original end times.

        start_in_mins (np.ndarray[float]):
            Returns the frame start times converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original start times.

        center_in_mins (np.ndarray[float]):
            Returns the frame center times converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original center times.

    Examples:

        .. code-block:: python

            import numpy as np
            from petpal.utils.scan_timing import ScanTimingInfo

            # Explicitly setting the attributes
            ## Define scan timing information
            duration = np.array([60.0, 120.0, 180.0])  # seconds
            start = np.array([0.0, 60.0, 180.0])
            end = np.array([60.0, 180.0, 360.0])
            center = (start + end) / 2.0  # Calculate the midpoints
            decay = np.array([1.0, 0.9, 0.8])  # Example decay values

            ## Create an instance of ScanTimingInfo
            scan_timing_info = ScanTimingInfo(duration=duration, end=end, start=start, center=center, decay=decay)

            ## Access original timing information
            print(scan_timing_info.duration)  # [ 60. 120. 180.]
            print(scan_timing_info.center)    # [30.  120. 270.]

            ## Access timing as minutes (when times exceed 200.0 seconds)
            print(scan_timing_info.duration_in_mins)  # [ 60. 120. 180.] (Unchanged)
            print(scan_timing_info.center_in_mins)    # [30. 120. 270.] (Unchanged)

            ## Example when `end` is greater than 200.0:
            scan_timing_info.end = np.array([300.0, 400.0, 500.0])  # Update end times
            print(scan_timing_info.end_in_mins)  # [5. 6.66666667 8.33333333] (Converted to minutes)
            print(scan_timing_info.start_in_mins)  # [0. 1. 3.] (Converted to minutes)

            # Getting the object directly from a nifti image file
            # assuming the metadata shares the name
            scan_timing_info = ScanTimingInfo.from_nifti("/path/to/image.nii.gz")

    """
    duration: np.ndarray[float]
    end: np.ndarray[float]
    start: np.ndarray[float]
    center: np.ndarray[float]
    decay: np.ndarray[float]

    @property
    def duration_in_mins(self) -> np.ndarray[float]:
        """
        Returns the frame durations in minutes. Validates values by checking if the final frame
        value is greater than 200: if so, then assumes values are in seconds and divides by 60.
        """
        if self.end[-1] >= 200.0:
            return self.duration / 60.0
        return self.duration

    @property
    def end_in_mins(self) -> np.ndarray[float]:
        """
        Returns the frame time ends in minutes. Validates values by checking if the final frame
        value is greater than 200: if so, then assumes values are in seconds and divides by 60.
        """
        if self.end[-1] >= 200.0:
            return self.end / 60.0
        return self.end

    @property
    def start_in_mins(self) -> np.ndarray[float]:
        """
        Returns the frame time starts in minutes. Validates values by checking if the final frame
        value is greater than 200: if so, then assumes values are in seconds and divides by 60.
        """
        if self.end[-1] >= 200.0:
            return self.start / 60.0
        return self.start

    @property
    def center_in_mins(self) -> np.ndarray[float]:
        """
        Returns the frame reference times in minutes. Validates values by checking if the final
        frame value is greater than 200: if so, then assumes values are in seconds and divides by
        60.
        """
        if self.end[-1] >= 200.0:
            return self.center / 60.0
        return self.center


    @classmethod
    def from_metadata(cls, metadata_dict: dict):
        r"""
        Extracts frame timing information and decay factors from a json metadata.
        Expects that the JSON metadata has ``FrameDuration`` and ``DecayFactor`` or
        ``DecayCorrectionFactor`` keys.

        .. important::
            This function tries to infer `FrameTimesEnd` and `FrameTimesStart` from the frame
            durations if those keys are not present in the metadata file. If the scan is broken,
            this might generate incorrect results.


        Args:
            metadata_dict (dict): The metadata dictionary, loaded into memory.

        Returns:
            :class:`ScanTimingInfo`: Frame timing information with the following elements:
                - duration (np.ndarray): Frame durations in seconds.
                - start (np.ndarray): Frame start times in seconds.
                - end (np.ndarray): Frame end times in seconds.
                - center (np.ndarray): Frame center times in seconds.
                - decay (np.ndarray): Decay factors for each frame.
        """
        frm_dur = np.asarray(metadata_dict['FrameDuration'], float)
        try:
            frm_ends = np.asarray(metadata_dict['FrameTimesEnd'], float)
        except KeyError:
            frm_ends = np.cumsum(frm_dur)
        try:
            frm_starts = np.asarray(metadata_dict['FrameTimesStart'], float)
        except KeyError:
            frm_starts = np.diff(frm_ends)
        try:
            decay = np.asarray(metadata_dict['DecayCorrectionFactor'], float)
        except KeyError:
            decay = np.asarray(metadata_dict['DecayFactor'], float)
        try:
            frm_centers = np.asarray(metadata_dict['FrameReferenceTime'], float)
        except KeyError:
            frm_centers = np.asarray(frm_starts + frm_dur / 2.0, float)

        return cls(duration=frm_dur,
                   start=frm_starts,
                   end=frm_ends,
                   center=frm_centers,
                   decay=decay)

    @classmethod
    def from_nifti(cls, image_path: str):
        r"""
        Extracts frame timing information and decay factors from a NIfTI image metadata.
        Expects that the JSON metadata file has ``FrameDuration`` and ``DecayFactor`` or
        ``DecayCorrectionFactor`` keys.

        .. important::
            This function tries to infer `FrameTimesEnd` and `FrameTimesStart` from the frame 
            durations if those keys are not present in the metadata file. If the scan is broken,
            this might generate incorrect results.


        Args:
            image_path (str): Path to the NIfTI image file.

        Returns:
           :class:`ScanTimingInfo`: Frame timing information with the following elements:
                - duration (np.ndarray): Frame durations in seconds.
                - start (np.ndarray): Frame start times in seconds.
                - end (np.ndarray): Frame end times in seconds.
                - center (np.ndarray): Frame center times in seconds.
                - decay (np.ndarray): Decay factors for each frame.
        """
        _meta_data = load_metadata_for_nifti_with_same_filename(image_path=image_path)
        return cls.from_metadata(metadata_dict=_meta_data)


def get_window_index_pairs_from_durations(frame_durations: np.ndarray, w_size: float):
    r"""
    Computes start and end index pairs for windows of a given size based on frame durations.

    Args:
        frame_durations (np.ndarray): Array of frame durations in seconds.
        w_size (float): Window size in seconds.

    Returns:
        np.ndarray: Array of shape (2, N), where the first row contains start indices,
            and the second row contains end indices for each window.

    Raises:
        ValueError: If `w_size` is less than or equal to 0.
        ValueError: If `w_size` is greater than the total duration of all frames.
    """
    if w_size <= 0:
        raise ValueError("Window size has to be > 0")
    if w_size > np.sum(frame_durations):
        raise ValueError("Window size is larger than the whole scan.")
    _tmp_w_ids = [0]
    _w_dur_sum = 0
    for frm_id, frm_dur in enumerate(frame_durations):
        _w_dur_sum += frm_dur
        if _w_dur_sum >= w_size:
            _tmp_w_ids.append(frm_id + 1)
            _w_dur_sum = 0
    w_start_ids = np.asarray(_tmp_w_ids[:-1])
    w_end_ids = np.asarray(_tmp_w_ids[1:])
    id_pairs = np.vstack((w_start_ids, w_end_ids))
    return id_pairs


def get_window_index_pairs_for_image(image_path: str, w_size: float):
    """
    Computes start and end index pairs for windows of a given size
    based on the frame durations of a NIfTI image.

    Args:
        image_path (str): Path to the NIfTI image file.
        w_size (float): Window size in seconds.

    Returns:
        np.ndarray: Array of shape (2, N), where the first row contains start indices,
            and the second row contains end indices for each window.

    Raises:
        ValueError: If `w_size` is less than or equal to 0.
        ValueError: If `w_size` is greater than the total duration of all frames.

    See Also:
        :func:`get_window_index_pairs_from_durations`
    """
    image_frame_info = ScanTimingInfo.from_nifti(image_path=image_path)
    return get_window_index_pairs_from_durations(frame_durations=image_frame_info.duration,
                                                 w_size=w_size)
