"""
Class to handle data related to time activity curves (TACs).

TODO:
    * Add more unit handling functionality
    * Cover exception handling
    * Refactor safe_load_tac to this module as a public method

"""
import os
import glob
import pathlib
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TimeActivityCurve:
    """Class to store time activity curve (TAC) data.
    
    Attributes:
        times (np.ndarray): Frame times for the TAC stored in an array.
        activity (np.ndarray): Activity values at each frame time stored in an array.
        uncertainty (np.ndarray): Uncertainty in the measurement of activity values stored in an
            array.


    Example:

        .. code-block:: python

            from petpal.utils.time_activity_curve import TimeActivityCurve

            my_tac = TimeActivityCurve.from_tsv('/path/to/tac.tsv')
            print(f"Frame times: {my_tac.times}")
            print(f"Activity: {my_tac.activity}")
            print(f"Uncertainty: {my_tac.uncertainty}")

            my_tac.times = my_tac.times / 60  # convert time units to hours
            my_tac.to_tsv(filename='/path/to/new_tac.tsv')  # save as new file
    """
    times: np.ndarray
    activity: np.ndarray
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if self.uncertainty.size == 0:
            self.uncertainty = np.empty_like(self.times)
            self.uncertainty[:] = np.nan
        assert np.shape(self.uncertainty) == np.shape(self.times) == np.shape(self.activity), (
            f"TAC fields must have the same shapes.\ntimes:{self.times.shape}"
            "activity:{self.activity.shape} uncertainty:{self.uncertainty.shape}")

    @classmethod
    def from_tsv(cls, filename: str):
        """
        Load an instance of TimeActivityCurve object from a TSV TAC file.

        Args:
            filename (str): Path to the TSV TAC file.
        
        Returns:
            (TimeActivityCurve): A TimeActivityCurve object loaded from a TSV TAC file.
        """
        return cls(*safe_load_tac(filename=filename, with_uncertainty=True))

    @property
    def tac(self) -> np.ndarray:
        """
        Get the TAC array, not including uncertainties.

        Returns:
            (np.ndarray): The TAC as a contiguous array, with the first index being time and the
                second index being activity.
        """
        return np.ascontiguousarray([self.times, self.activity])

    @property
    def tac_werr(self) -> np.ndarray:
        """
        Get the TAC array, including uncertainties.

        Returns:
            (np.ndarray): The TAC as a contiguous array, with the first index being time and the
                second index being activity, and the third index being uncertainty.
        """
        return np.ascontiguousarray([self.times, self.activity, self.uncertainty])

    @property
    def times_in_mins(self) -> np.ndarray[float]:
        """
        Returns the TAC measured times in minutes. Validates values by checking if the final
        frame value is greater than 200: if so, then assumes values are in seconds and divides by
        60.
        """
        if self.times[-1] >= 200.0:
            return self.times / 60.0
        return self.times


    def to_tsv(self, filename: str, col_names: list[str]=None):
        """
        Writes the TAC object to file, including measurement times, activity, and uncertainty.

        Args:
            filename (str): Path to the file that will be written to.
            col_names (list[str]): Custom names for time, activity, and uncertainty columns
                respectively. See :meth:`safe_write_tac`. Default None.
        """
        safe_write_tac(filename=filename,tac_data=self.tac_werr,col_names=col_names)


def safe_load_tac(filename: str,
                  with_uncertainty: bool = False,
                  **kwargs) -> np.ndarray:
    """
    Loads time-activity curves (TAC) from a file.
    Tries to read a TAC from specified file and raises an exception if unable to do so. We assume
    that the file has two columns, the first corresponding to time and second corresponding to
    activity.
    Args:
        filename (str): The name of the file to be loaded.
        with_uncertainty (bool): Load uncertainty of measured activity along with timing and 
            activity.
        **kwargs (dict): keyword arguments to pass to :func:`np.loadtxt`.
    Returns:
        np.ndarray: A numpy array containing the loaded TAC. The first index corresponds to the
            times, and the second corresponds to the activity. If with_uncertainty is True, the
            third index corresponds to the uncertainty.
    Raises:
        Exception: An error occurred loading the TAC.

        
    Example:

        .. code-block:: python

            import numpy as np
            from petpal.utils.time_activity_curve import safe_load_tac, safe_write_tac

            my_tac = safe_load_tac(filename='/path/to/tac.tsv')
            my_tac_modified = np.zeros_like(my_tac)
            my_tac_modified[0] = my_tac[0] / 60 # convert time units to hours
            my_tac_modified[1] = my_tac[1] / 37000 # convert activity units to mCi
            my_tac_modified[2] = my_tac[2] / 37000 # convert uncertainties like activity
            safe_write_tac(filename='/path/to/new_tac.tsv',
                           tac_data=my_tac_modified)
    """
    try:
        tac_data = np.asarray(np.loadtxt(filename, **kwargs).T, dtype=float, order='C')
    except ValueError:
        tac_data = np.asarray(np.loadtxt(filename, skiprows=1, **kwargs).T, dtype=float, order='C')
    except Exception as e:
        print(f"Couldn't read file {filename}. Error: {e}")
        raise e

    if np.max(tac_data[0]) >= 300:
        tac_data[0] /= 60.0

    if with_uncertainty:
        return tac_data[:3]
    else:
        return tac_data[:2]


def safe_write_tac(filename: str,
                   tac_data: np.ndarray,
                   col_names: list[str]=None):
    """
    Writes the data in a time-activity curve (TAC) to a file. Assumes the TAC data consists of a
    contiguous numpy array with two or three columns: time, activity, and (optionally) uncertainty.

    Args:
        filename (str): Path to the file the data will be saved as.
        tac_data (np.ndarray): Numpy array containing the data for the TAC. Assumes the TAC data
            consists of a contiguous numpy array with two or three columns: time, activity, and 
            (optionally) uncertainty.
        col_names (list[str]): List of column names assigned to the time, activity, and uncertainty
            columns in the TAC data, respectively. Must match number of columns in `tac_data`.
    
    Raises:
        ValueError: If the number of columns in tac_data is not two or three, or the number of
            columns in tac_data does not match the number of columns in col_names.



    Example:

        .. code-block:: python

            import numpy as np
            from petpal.utils.time_activity_curve import safe_load_tac, safe_write_tac

            my_tac = safe_load_tac(filename='/path/to/tac.tsv')
            my_tac_modified = np.zeros_like(my_tac)
            my_tac_modified[0] = my_tac[0] / 60 # convert time units to hours
            my_tac_modified[1] = my_tac[1] / 37000 # convert activity units to mCi
            my_tac_modified[2] = my_tac[2] / 37000 # convert uncertainties like activity
            safe_write_tac(filename='/path/to/new_tac.tsv',
                           tac_data=my_tac_modified)

    """
    num_cols = len(tac_data)
    if num_cols not in (2, 3):
        raise ValueError(f"Expected two or three columns in tac_data. Got {num_cols}.")

    if col_names is None:
        if num_cols==2:
            col_names = ['FrameReferenceTime', 'MeanActivityConcentration']
        elif num_cols==3:
            col_names = ['FrameReferenceTime', 'MeanActivityConcentration','Uncertainty']

    if num_cols!=len(col_names):
        raise ValueError("Expected the same number of columns in tac_data and col_names. Got "
                         f"{num_cols} in tac_data and {len(col_names)} in col_names.")

    file_header = "\t".join(col_names)
    np.savetxt(fname=filename, X=tac_data.T, header=file_header, comments='')


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
    def __init__(self, input_tac_path: str, tacs_dir: str):
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
        if not os.path.isdir(tacs_dir):
            raise AssertionError("`tacs_dir` must be a valid directory: "
                                 f"got {os.path.abspath(tacs_dir)}")
        glob_path = os.path.join(tacs_dir, "*_tac.tsv")
        tacs_files_list = sorted(glob.glob(glob_path))

        return tacs_files_list


    def get_tacs_objects_dict_from_files_list(self, tacs_files_list: list[str]):
        """
        Creates a dict of TAC objects from a list of file paths.

        Args:
            tacs_files_list (list[str]): List of TAC file paths.

        Returns:
            dict: Dictionary of region name-TAC object pairs.
        """
        tacs_dict = {}
        for tac_file in tacs_files_list:
            region = self.infer_segmentation_label_from_tac_path(tac_path=tac_file)
            tacs_dict[region] = TimeActivityCurve.from_tsv(filename=tac_file)
        return tacs_dict


    def get_tacs_objects_dict_from_dir(self, tacs_dir: str) -> dict:
        """
        Creates a dict of TAC objects from a directory of TAC files.

        Args:
            tacs_dir (str): A directory of TAC files.

        Returns:
            dict: Dictionary of region name-TAC object pairs.
        """
        tacs_files_list = self.get_tacs_list_from_dir(tacs_dir=tacs_dir)
        tacs_dict = self.get_tacs_objects_dict_from_files_list(tacs_files_list=tacs_files_list)
        return tacs_dict

    @staticmethod
    def get_tacs_objects_list_from_files_list(tacs_files_list: list[str]):
        """
        Creates a list of TAC objects from a list of file paths.

        Args:
            tacs_files_list (list[str]): List of TAC file paths.

        Returns:
            list[TimeActivityCurve]: List of TAC objects.
        """
        tacs_list = [TimeActivityCurve.from_tsv(filename=tac_file) for tac_file in tacs_files_list]
        return tacs_list

    @staticmethod
    def get_tacs_vals_from_objs_list(tacs_objects_list: list[TimeActivityCurve]):
        """
        Extracts TAC values from a list of TAC objects.

        Args:
            tacs_objects_list (list[TimeActivityCurve]): List of TAC objects.

        Returns:
            list: List of TAC values.
        """
        tacs_vals = [tac.activity for tac in tacs_objects_list]
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
    def infer_segmentation_label_from_tac_path(tac_path: str, tac_id: int=0):
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
            segname = ''.join(segparts)
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
