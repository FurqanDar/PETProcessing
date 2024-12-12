"""
Class to handle data related to time activity curves (TACs).

TODO:
    * Add more unit handling functionality
    * Cover exception handling
    * Refactor safe_load_tac to this module as a public method

"""
import os, glob, pathlib
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


class MultiTACAnalysisMixin:
    """
    A mixin class providing utilities for handling multiple analysis of Time Activity Curves (TACs)
    in a directory.

    Attributes:
        input_tac_path (str): Path to the input TAC file.
        tacs_dir (str): Directory containing TAC files.
        tacs_files_list (list[str]): List of TAC file paths.
        num_of_tacs (int): Number of TACs found in the directory.
        inferred_seg_labels (list[str]): List of inferred segmentation labels for TACs.
    """
    def __init__(self, input_tac_path: str, tacs_dir: str, ):
        """
        Initializes the MultiTACAnalysisMixin with paths and initializes analysis properties.

        Args:
            input_tac_path (str): Path to the input TAC file.
            tacs_dir (str): Directory containing TAC files.
        """
        self._input_tac_path = input_tac_path
        self._tacs_dir = tacs_dir
        self.input_tac_path = input_tac_path
        self.tacs_dir = tacs_dir
        self.tacs_files_list = self.get_tacs_list_from_dir(self.tacs_dir)
        self.num_of_tacs = len(self.tacs_files_list)
        self.inferred_seg_labels = self.infer_segmentation_labels_for_tacs()
    
    @property
    def input_tac_path(self):
        """Gets the input TAC file path."""
        return self._input_tac_path
    
    @input_tac_path.setter
    def input_tac_path(self, input_tac_path):
        """Sets the input TAC file path."""
        self._input_tac_path = input_tac_path
    
    @property
    def reference_tac_path(self):
        """Gets the reference TAC file path."""
        return self.input_tac_path
    
    @reference_tac_path.setter
    def reference_tac_path(self, reference_tac_path):
        """Sets the reference TAC file path."""
        self.input_tac_path = reference_tac_path
    
    @property
    def tacs_dir(self):
        """Gets the TAC directory path."""
        return self._tacs_dir
    
    @tacs_dir.setter
    def tacs_dir(self, tacs_dir):
        """
        Sets the TAC directory path and validates its contents.

        Raises:
            FileNotFoundError: If the directory does not contain any TAC files.
        """
        if self.is_valid_tacs_dir(tacs_dir):
            self._tacs_dir = tacs_dir
        else:
            raise FileNotFoundError("`tacs_dir` must contain at least one TAC file. Check the"
                                    f" contents of the directory: {self.tacs_dir}.")
    
    def is_valid_tacs_dir(self, tacs_dir: str):
        """
        Validates the TAC directory by checking for TAC files.

        Args:
            tacs_dir (str): Directory to validate.

        Returns:
            bool: True if valid, otherwise False.
        """
        tacs_files_list = self.get_tacs_list_from_dir(tacs_dir)
        if tacs_files_list:
            return True
        else:
            return False
    
    @staticmethod
    def get_tacs_list_from_dir(tacs_dir: str) -> list[str]:
        """
        Retrieves a sorted list of TAC file paths from a directory.

        Args:
            tacs_dir (str): Directory from which to retrieve TAC files.

        Returns:
            list[str]: Sorted list of TAC file paths.
        """
        assert os.path.isdir(tacs_dir), f"`tacs_dir` must be a valid directory: {os.path.abspath(tacs_dir)}"
        glob_path = os.path.join(tacs_dir, "*_tac.tsv")
        tacs_files_list = sorted(glob.glob(glob_path))
        
        return tacs_files_list
    
    @staticmethod
    def get_tacs_objects_list_from_files_list(tacs_files_list: list[str]):
        """
        Creates a list of TAC objects from a list of file paths.

        Args:
            tacs_files_list (list[str]): List of TAC file paths.

        Returns:
            list[TimeActivityCurveFromFile]: List of TAC objects.
        """
        tacs_list = [TimeActivityCurveFromFile(tac_path=tac_file) for tac_file in tacs_files_list]
        return tacs_list
    
    @staticmethod
    def get_tacs_vals_from_objs_list(tacs_objects_list: list[TimeActivityCurveFromFile]):
        """
        Extracts TAC values from a list of TAC objects.

        Args:
            tacs_objects_list (list[TimeActivityCurveFromFile]): List of TAC objects.

        Returns:
            list: List of TAC values.
        """
        tacs_vals = [tac.tac_vals for tac in tacs_objects_list]
        return tacs_vals
    
    def get_tacs_vals_from_dir(self, tacs_dir: str):
        """
        Retrieves TAC values from files in a specified directory.

        Args:
            tacs_dir (str): Directory containing TAC files.

        Returns:
            list: List of TAC values.
        """
        tacs_files_list = self.get_tacs_list_from_dir(tacs_dir)
        tacs_objects_list = self.get_tacs_objects_list_from_files_list(tacs_files_list)
        tacs_vals = self.get_tacs_vals_from_objs_list(tacs_objects_list)
        return tacs_vals
    
    @staticmethod
    def infer_segmentation_label_from_tac_path(tac_path: str, tac_id:int):
        """
        Infers a segmentation label from a TAC file path by analyzing the filename.

        This method extracts a segment label from the filename of a TAC file. It checks the presence
        of a `seg-` marker in the filename, which is followed by the segment name. This segment name
        is then formatted with each part capitalized. If no segment label is found,
        a default unknown label is generated using the TAC's ID.

        Args:
            tac_path (str): Path of the TAC file.
            tac_id (int): ID of the TAC.

        Returns:
            str: Inferred segmentation label.
        """
        path = pathlib.Path(tac_path)
        assert path.suffix == '.tsv', '`tac_path` must point to a TSV file (*.tsv)'
        filename = path.name
        fileparts = filename.split("_")
        segname = 'XXXX'
        for part in fileparts:
            if 'seg-' in part:
                segname = part.split('seg-')[-1]
                break
        if segname == 'XXXX':
            return f'UNK{tac_id:03}'
        else:
            segparts = segname.split("-")
            segparts_capped = [a_part.capitalize() for a_part in segparts]
            segname = ''.join(segparts_capped)
            return segname
        
    def infer_segmentation_labels_for_tacs(self):
        """
        Infers segmentation labels for TACs.

        Returns:
            list[str]: List of inferred segmentation labels.
            
        See Also:
            :meth:`infer_segmentation_label_from_tac_path`
        """
        seg_labels = []
        for tac_id, tac_file in enumerate(self.tacs_files_list):
            tmp_seg = self.infer_segmentation_label_from_tac_path(tac_path=tac_file, tac_id=tac_id)
            seg_labels.append(tmp_seg)
            
        return seg_labels