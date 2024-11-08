"""
Class to handle data related to time activity curves (TACs).

TODO:
    * Add more unit handling functionality
    * Cover exception handling
    * Refactor safe_load_tac to this module as a public method

"""
from dataclasses import dataclass
import numpy as np
from .image_io import safe_load_tac

@dataclass
class TimeActivityCurve:
    """Class to store time activity curve (TAC) data.
    
    Attributes:
        tac_times_in_minutes (np.ndarray): Frame times for the TAC stored in an array.
        tac_vals (np.ndarray): Activity values at each frame time stored in an array."""
    tac_times_in_minutes: np.ndarray
    tac_vals: np.ndarray


class TimeActivityCurveFromFile:
    """
    Class to handle data related to time activity curves (TACs).

    Attributes:
        tac_path (str): Path to the original time activity curve file.
        tac_times_in_minutes (np.ndarray): Frame times for the TAC stored in an array.
        tac_vals (np.ndarray): Activity values at each frame time stored in an array.
    """
    def __init__(self,
                 tac_path: str):
        """
        Initialize TimeActivityCurve class

        Args:
            tac_path (str): Path to the TAC that will be analyzed.
        """
        self.tac_path = tac_path
        self.tac_times_in_minutes, self.tac_vals = self.get_tac_data()

    def get_tac_data(self):
        """
        Retrieves data from the TAC file. Uses :meth:`petpal.utils.image_io.safe_load_tac`.

        See also:
            * :meth:`petpal.utils.image_io.safe_load_tac`

        """
        return safe_load_tac(self.tac_path)

    def get_frame_durations(self) -> np.ndarray:
        """
        Get array containing the duration of each frame in minutes.

        For a set of N frames, the first N-1 frame durations are estimated as the difference
        between each frame time and the next frame time. Frame N is then inferred as being the same
        duration as frame N-1.

        The frame durations in the originating metadata is preferable to computing it here. 
        However, if the frame durations are not present in the metadata this function is useful
        to recover them.

        Returns:
            tac_durations_in_minutes (np.ndarray): The estimated duration of each frame in minutes.
        """
        tac_times_in_minutes = self.tac_times_in_minutes
        tac_durations_in_minutes = np.zeros((len(tac_times_in_minutes)))

        tac_durations_in_minutes[:-1] = tac_times_in_minutes[1:]-tac_times_in_minutes[:-1]
        tac_durations_in_minutes[-1] = tac_durations_in_minutes[-2]

        return tac_durations_in_minutes
