"""
This module provides functions and a key class, :class:`GraphicalAnalysisParametricImage`, for 
graphical analysis and creation of parametric images of 4D-PET scan data. It heavily utilizes 
:mod:`numpy` for data manipulation and assumes the input as 4D PET images along with other required
inputs.

The :class:`GraphicalAnalysisParametricImage` class encapsulates the main functionality of the 
module, and encompasses methods for initializing data, running and saving analysis, calculating
various properties, and handling parametric image data.
"""

import os
import warnings
import json
from typing import Tuple, Callable
import nibabel
import numpy as np
import numba
from nibabel import Nifti1Image
from . import graphical_analysis
from ..utils.image_io import safe_load_tac, safe_copy_meta



@numba.njit()
def apply_linearized_analysis_to_all_voxels(pTAC_times: np.ndarray,
                                            pTAC_vals: np.ndarray,
                                            tTAC_img: np.ndarray,
                                            t_thresh_in_mins: float,
                                            analysis_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates parametric images for 4D-PET data using the provided analysis method.

    This function iterates over each voxel in the given `tTAC_img` and applies the provided `analysis_func`
    to compute analysis values. The `analysis_func` should be a numba.jit function for optimization and
    should be following a signature compatible with either of the following: patlak_analysis, logan_analysis,
    or alt_logan_analysis.

    Args:
        pTAC_times (np.ndarray): A 1D array representing the input TAC times in minutes.
        
        pTAC_vals (np.ndarray): A 1D array representing the input TAC values. This array should
                                be of the same length as `pTAC_times`.

        tTAC_img (np.ndarray): A 4D array representing the 3D PET image over time.
                               The shape of this array should be (x, y, z, time).

        t_thresh_in_mins (float): A float representing the threshold time in minutes.
                                  It is applied when calling the `analysis_func`.

        analysis_func (Callable): A numba.jit function to apply to each voxel for given PET data.
                                  It should take the following arguments:

                                    - input_tac_values: 1D numpy array for input TAC values
                                    - region_tac_values: 1D numpy array for regional TAC values
                                    - tac_times_in_minutes: 1D numpy array for TAC times in minutes
                                    - t_thresh_in_minutes: a float for threshold time in minutes

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two 3D numpy arrays representing the calculated slope image
                                       and the intercept image, each of the same spatial dimensions as
                                       `tTAC_img`.
    """
    img_dims = tTAC_img.shape

    slope_img = np.zeros((img_dims[0], img_dims[1], img_dims[2]), float)
    intercept_img = np.zeros_like(slope_img)

    for i in range(0, img_dims[0], 1):
        for j in range(0, img_dims[1], 1):
            for k in range(0, img_dims[2], 1):
                analysis_vals = analysis_func(input_tac_values=pTAC_vals,
                                              region_tac_values=tTAC_img[i, j, k, :],
                                              tac_times_in_minutes=pTAC_times,
                                              t_thresh_in_minutes=t_thresh_in_mins)
                slope_img[i, j, k] = analysis_vals[0]
                intercept_img[i, j, k] = analysis_vals[1]

    return slope_img, intercept_img


def generate_parametric_images_with_graphical_method(pTAC_times: np.ndarray,
                                                     pTAC_vals: np.ndarray,
                                                     tTAC_img: np.ndarray,
                                                     t_thresh_in_mins: float,
                                                     method_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates parametric images for 4D-PET data using a specified graphical analysis method.

    This function maps one of the predefined method names to the corresponding analysis function,
    and then generates parametric images by applying it to the given 4D-PET data using the
    `apply_linearized_analysis_to_all_voxels` function.

    Args:
        pTAC_times (np.ndarray): A 1D array representing the input TAC times in minutes.

        pTAC_vals (np.ndarray): A 1D array representing the input TAC values. This array should
                                be of the same length as `pTAC_times`.

        tTAC_img (np.ndarray): A 4D array representing the 3D PET image over time.
                               The shape of this array should be (x, y, z, time).

        t_thresh_in_mins (float): A float representing the threshold time in minutes.

        method_name (str): The analysis method's name to apply. Must be one of: 'patlak', 'logan', or 'alt_logan'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two 3D numpy arrays representing the calculated slope image
                                       and the intercept image, each of the same spatial dimensions as
                                       `tTAC_img`.

    Raises:
       ValueError: If the `method_name` is not one of the following: 'patlak', 'logan', 'alt_logan'.
    """
    
    analysis_func = graphical_analysis.get_graphical_analysis_method(method_name=method_name)
    slope_img, intercept_img = apply_linearized_analysis_to_all_voxels(pTAC_times=pTAC_times,
                                                                       pTAC_vals=pTAC_vals,
                                                                       tTAC_img=tTAC_img,
                                                                       t_thresh_in_mins=t_thresh_in_mins,
                                                                       analysis_func=analysis_func)

    return slope_img, intercept_img

def _safe_load_4dpet_nifty(filename: str) -> Nifti1Image:
    """
    Safely load a 4D PET NIfTI file.

    This function checks if the given file has a '.nii' or '.nii.gz' extension, then tries to load
    it as a NIfTI file using the nibabel library. If the file cannot be loaded, it raises an
    exception.

    Args:
        filename (str): The path of the NIfTI file to be loaded.

    Returns:
        Nifti1Image: The loaded NIfTI 4D PET image.

    Raises:
        ValueError: If the file does not have a '.nii' or '.nii.gz' extension.
        Exception:  If an error occurred while loading the NIfTI file.
    """
    if not filename.endswith(('.nii', '.nii.gz')):
        raise ValueError("Invalid file extension. Only '.nii' and '.nii.gz' are supported.")

    try:
        return nibabel.load(filename=filename)
    except Exception as e:
        print(f"Couldn't read file {filename}. Error: {e}")
        raise e


class GraphicalAnalysisParametricImage:
    """
    Class for generating parametric images of 4D-PET images using graphical analyses. It provides methods to run
    graphical analysis, calculate properties of the resulting images, and save the results using file paths.

    Attributes:
        input_tac_path (str): Absolute path to the input Time-Activity Curve (TAC) file.
        pet4D_img_path (str): Absolute path to the 4D PET image file.
        output_directory (str): Absolute path to the output directory.
        output_filename_prefix (str): Prefix of the output file names.
        analysis_props (dict): Dictionary of properties of the graphical analysis.
        slope_image (np.ndarray): The slope image resulting from the graphical analysis, initialized to None.
        intercept_image (np.ndarray): The intercept image resulting from the graphical analysis, initialized to None.

    """
    
    def __init__(self,
                 input_tac_path: str,
                 pet4D_img_path: str,
                 output_directory: str,
                 output_filename_prefix: str) -> None:
        """
        Initializes the GraphicalAnalysisParametricImage with the specified parameters.

        This method initializes necessary attributes for the GraphicalAnalysisParametricImage object with the provided
        arguments. It sets the absolute file paths for the input TAC, 4D PET image, and output directory, and
        initializes the analysis properties. Further, it initializes variables for the slope and intercept images.

        Args:
            input_tac_path (str): Path to the input Time-Activity Curve (TAC) file.
            pet4D_img_path (str): Path to the 4D PET image file.
            output_directory (str): Path to the destination directory where output files will be saved.
            output_filename_prefix (str): Prefix to use for the names of the output files.

        Returns:
            None
        """
        self.input_tac_path = os.path.abspath(input_tac_path)
        self.pet4D_img_path = os.path.abspath(pet4D_img_path)
        self.output_directory = os.path.abspath(output_directory)
        self.output_filename_prefix = output_filename_prefix
        self.analysis_props = self.init_analysis_props()
        self.slope_image: np.ndarray = None
        self.intercept_image: np.ndarray = None
    
    def init_analysis_props(self):
        """
        Initializes the analysis properties dictionary.

        The analysis properties dictionary contains properties derived from the analysis.
        It begins with certain known values, such as file paths, but most values are initialized
        to None and filled in later as the analysis is performed.

        Properties include:
            * ``FilePathPTAC`` (str): The path to the input Time-Activity Curve (TAC) file.
            * ``FilePathTTAC`` (str): The path to the 4D PET image file.
            * ``MethodName`` (str): The name of the graphical analysis method used, to be filled in later.
            * ``ImageDimensions`` (tuple): The dimensions of the images resulting from the analysis, to be filled in later.
            * ``StartFrameTime`` (float): The start time of the frame used in the analysis, filled in after the analysis.
            * ``EndFrameTime`` (float): The end time of the frame used in the analysis, filled in after the analysis.
            * ``ThresholdTime`` (float): The time threshold used in the analysis, filled in after the analysis.
            * ``NumberOfPointsFit`` (int): The number of points fitted in the analysis, filled in after the analysis.
            * ``SlopeMaximum`` (float): The maximum slope found in the analysis, filled in after the analysis.
            * ``SlopeMinimum`` (float): The minimum slope found in the analysis, filled in after the analysis.
            * ``SlopeMean`` (float): The mean of the slopes found in the analysis, filled in after the analysis.
            * ``SlopeVariance`` (float): The variance of the slopes found in the analysis, filled in after the analysis.
            * ``InterceptMaximum`` (float): The maximum intercept found in the analysis, filled in after the analysis.
            * ``InterceptMinimum`` (float): The minimum intercept found in the analysis, filled in after the analysis.
            * ``InterceptMean`` (float): The mean of the intercepts found in the analysis, filled in after the analysis.
            * ``InterceptVariance`` (float): The variance of the intercepts found in the analysis, filled in after the analysis.

        Returns:
            props (dict): The initialized properties dictionary.
        """
        props = {
            'FilePathPTAC'     : self.input_tac_path, 'FilePathTTAC': self.pet4D_img_path, 'MethodName': None,
            'ImageDimensions'  : None, 'StartFrameTime': None, 'EndFrameTime': None, 'ThresholdTime': None,
            'NumberOfPointsFit': None, 'SlopeMaximum': None, 'SlopeMinimum': None, 'SlopeMean': None,
            'SlopeVariance'    : None, 'InterceptMaximum': None, 'InterceptMinimum': None, 'InterceptMean': None,
            'InterceptVariance': None,
            }
        return props

    def run_analysis(self, method_name: str, t_thresh_in_mins: float):
        """
        Executes the complete analysis procedure.

        This method orchestrates the full analysis by orchestrating the calculation of parametric images, as well as
        compiling the properties related to the analysis. Both are determined based on the provided method name and the
        threshold time.

        Parameters:
            method_name (str): The name of the methodology adopted for the process.
            t_thresh_in_mins (float): The threshold time used through the analysis (in minutes).

        See Also:
            * :func:`calculate_parametric_images`
            * :func:`calculate_analysis_properties`

        Returns:
            None

        """
        self.calculate_parametric_images(method_name=method_name, t_thresh_in_mins=t_thresh_in_mins)
        self.calculate_analysis_properties(method_name=method_name, t_thresh_in_mins=t_thresh_in_mins)

    def save_analysis(self):
        """
        Stores the results from an analysis routine.

        This method executes the storage of parametric images, as well as the properties related to the analysis. It
        assumes that the method 'run_analysis' is called before this method.

        Raises:
            RuntimeError: If the method 'run_analysis' is not called before this method.

        See Also:
            * :func:`save_parametric_images`
            * :func:`save_analysis_properties`

        Returns:
            None

        """
        if self.slope_image is None:
            raise RuntimeError("'run_analysis' method must be called before 'save_analysis'.")
        self.save_parametric_images()
        self.save_analysis_properties()

    def calculate_analysis_properties(self, method_name: str, t_thresh_in_mins: float):
        """
        Performs a set of calculations to collate various analysis properties.

        This method orchestrates the calculation of properties related to both the parametric images and the fitting
        process. It does this by calling :meth:`calculate_parametric_images_properties` and
        :meth:`calculate_fit_properties` respectively.

        Parameters:
            method_name (str): The name of the method used for the fitting process.
            t_thresh_in_mins (float): The threshold time (in minutes) used for the fitting process.

        See Also:
            * :meth:`calculate_parametric_images_properties`
            * :meth:`calculate_fit_properties`

        Returns:
            None. The results are stored within the instance's ``analysis_props`` variable.
        """
        self.calculate_parametric_images_properties()
        self.calculate_fit_properties(method_name=method_name, t_thresh_in_mins=t_thresh_in_mins)

    def calculate_fit_properties(self, method_name: str, t_thresh_in_mins: float):
        """
        Calculates and stores the properties related to the fitting process.

        This method calculates several properties related to the fitting process, including the threshold time, the name
        of the method used, the start and end frame time, and the number of points used in the fit. These values are
        stored in the instance's `analysis_props` variable.

        Parameters:
            method_name (str): The name of the methodology adopted for the fitting process.
            t_thresh_in_mins (float): The threshold time (in minutes) used in the fitting process.

        Note:
            This method relies on the :func:`safe_load_tac` function to load time-activity curve (TAC) data from the
            file at ``self.input_tac_path``, and the :func:`petpal.graphical_analysis.get_index_from_threshold`
            function to get the index from the threshold time.

        See also:
            * :func:`safe_load_tac`: Function to safely load TAC data from a file.
            * :func:`petpal.graphical_analysis.get_index_from_threshold`: Function to get the index from the threshold time.

        Returns:
            None. The results are stored within the instance's ``analysis_props`` variable.
        """
        self.analysis_props['ThresholdTime'] = t_thresh_in_mins
        self.analysis_props['MethodName'] = method_name

        p_tac_times, _ = safe_load_tac(filename=self.input_tac_path)
        t_thresh_index = graphical_analysis.get_index_from_threshold(times_in_minutes=p_tac_times,
                                                                     t_thresh_in_minutes=t_thresh_in_mins)
        self.analysis_props['StartFrameTime'] = p_tac_times[t_thresh_index]
        self.analysis_props['EndFrameTime'] = p_tac_times[-1]
        self.analysis_props['NumberOfPointsFit'] = len(p_tac_times[t_thresh_index:])

    def calculate_parametric_images_properties(self):
        """
        Initiates the calculation of properties for parametric images.

        This method triggers the calculation of statistical properties for slope and intercept images.
        Additionally, it captures the shape of the slope image as the 'ImageDimensions' and stores it in `analysis_props`.

        Note:
            You should ensure the `slope_image` attribute has been correctly set before calling this method. This means
            that `run_analysis` has already been called.

        See Also:
            calculate_slope_image_properties: Method to calculate various statistics for slope image.
            calculate_intercept_image_properties: Method to calculate various statistics for intercept image.

        Returns:
            None. The results are stored within the instance's `analysis_props` variable.
        """
        self.analysis_props['ImageDimensions'] = self.slope_image.shape
        self.calculate_slope_image_properties()
        self.calculate_intercept_image_properties()

    def calculate_slope_image_properties(self):
        """
        Calculates and stores statistical properties of the slope image.

        This method calculates the maximum, minimum, mean, and variance of
        the `slope_image` attribute, and stores these values in the `analysis_props` dictionary.

        The keys in `analysis_props` for these values are: `SlopeMaximum`, `SlopeMinimum`,
        `SlopeMean`, and `SlopeVariance`, respectively.

        Note:
            You should ensure the `slope_image` attribute has been correctly set before calling this method.

        No explicit return value. The results are stored within the instance's `analysis_props` variable.
        """
        self.analysis_props['SlopeMaximum'] = np.max(self.slope_image)
        self.analysis_props['SlopeMinimum'] = np.min(self.slope_image)
        self.analysis_props['SlopeMean'] = np.mean(self.slope_image)
        self.analysis_props['SlopeVariance'] = np.var(self.slope_image)

    def calculate_intercept_image_properties(self):
        """
        Calculates and stores statistical properties of the intercept image.

        This method calculates the maximum, minimum, mean, and variance of
        the `intercept_image` attribute, and stores these values in the `analysis_props` dictionary.

        The keys in `analysis_props` for these values are: `InterceptMaximum`, `InterceptMinimum`, `InterceptMean`,
        and `InterceptVariance`, respectively.

        Note:
            You should ensure the `intercept_image` attribute has been correctly set before calling this method.

        No explicit return value. The results are stored within the instance's `analysis_props` variable.
        """
        self.analysis_props['InterceptMaximum'] = np.max(self.intercept_image)
        self.analysis_props['InterceptMinimum'] = np.min(self.intercept_image)
        self.analysis_props['InterceptMean'] = np.mean(self.intercept_image)
        self.analysis_props['InterceptVariance'] = np.var(self.intercept_image)

    # TODO: Come up with a smarter way to get the PET data in the correct units. I would prefer that 4DPET is saved in the right units already.
    def calculate_parametric_images(self, method_name: str, t_thresh_in_mins: float):
        """
        Performs graphical analysis of PET parametric images and generates/updates the slope and intercept images.
        
        Warning:
            This method divides the PET image values by 37000 for unit conversion to Bq/cc. Be aware of this if you are
            using this function with PET images that are in other units!

        This method uses the given graphical analysis method and threshold to perform the analysis given the input Time
        Activity Curve (TAC) and 4D PET image, and updates the slope and intercept images accordingly. PET images are
        loaded from the specified path and divided by 37000 to convert the values to Bq/cc. Then, the parametric images
        are calculated using the specified graphical method and threshold time by explicitly analyzing each voxel in
        the 4D PET image.

        Args:
            method_name (str): The name of the graphical analysis method to be used.
            t_thresh_in_mins (float): The threshold time in minutes.

        Returns:
            None

        Raises:
            Exception: An error occurred during the graphical analysis. This could be due to an invalid method name or
            incorrect inputs to the method.
            
        See Also:
            * :func:`generate_parametric_images_with_graphical_method`
            * :func:`petpal.graphical_analysis.patlak_analysis`
            * :func:`petpal.graphical_analysis.logan_analysis`
            * :func:`petpal.graphical_analysis.alternative_logan_analysis`
            
        Notes:
            The conversion to Bq/cc is hard-coded, and could be changed in later versions of the module.
        """
        p_tac_times, p_tac_vals = safe_load_tac(self.input_tac_path)
        nifty_pet4d_img = _safe_load_4dpet_nifty(filename=self.pet4D_img_path)
        warnings.warn("PET image values are being divided by 37000 for unit conversion to Bq/cc.", UserWarning)
        self.slope_image, self.intercept_image = generate_parametric_images_with_graphical_method(
                pTAC_times=p_tac_times, pTAC_vals=p_tac_vals, tTAC_img=nifty_pet4d_img.get_fdata() / 37000.,
                t_thresh_in_mins=t_thresh_in_mins, method_name=method_name)

    def save_parametric_images(self):
        """
        Saves the slope and intercept images as NIfTI files in the specified output directory.

        This method generates and saves two NIfTI files: one for the slope image and one for the intercept image.
        It uses the output directory and filename prefix provided during instantiation of the class,
        along with the analysis method name, to generate a filename prefix for both images.
        The filenames follow the patterns `{output_filename_prefix}-parametric-{method}-slope.nii.gz` and
        `{output_filename_prefix}-parametric-{method}-intercept.nii.gz` respectively.
        The affine transformation matrix for the new NIfTI images is derived from the original 4D PET image.

        Args:
            None

        Returns:
            None

        Raises:
            IOError: An error occurred accessing the output_directory or while writing to the NIfTI file.

        """
        file_name_prefix = os.path.join(self.output_directory,
                                        f"{self.output_filename_prefix}_desc-{self.analysis_props['MethodName']}")
        nifty_img_affine = _safe_load_4dpet_nifty(filename=self.pet4D_img_path).affine
        try:
            tmp_slope_img = nibabel.Nifti1Image(dataobj=self.slope_image, affine=nifty_img_affine)
            nibabel.save(tmp_slope_img, f"{file_name_prefix}-slope.nii.gz")

            tmp_intercept_img = nibabel.Nifti1Image(dataobj=self.intercept_image, affine=nifty_img_affine)
            nibabel.save(tmp_intercept_img, f"{file_name_prefix}_intercept.nii.gz")
            
            safe_copy_meta(input_image_path=self.pet4D_img_path,
                                    out_image_path=f"{file_name_prefix}_slope.nii.gz")
            safe_copy_meta(input_image_path=self.pet4D_img_path,
                                    out_image_path=f"{file_name_prefix}_intercept.nii.gz")
        except IOError as e:
            print("An IOError occurred while attempting to write the NIfTI image files.")
            raise e from None

    def save_analysis_properties(self):
        """
        Saves the analysis properties to a JSON file in the output directory.

        This method involves saving a dictionary of analysis properties, which include file paths, analysis method,
        start and end frame times, threshold time, number of points fitted, and various properties like the maximum,
        minimum, mean, and variance of slopes and intercepts found in the analysis. These analysis properties are
        written to a JSON file in the output directory with the name following the pattern
        `{output_filename_prefix}-analysis-props.json`.

        Args:
            None

        Returns:
            None

        Raises:
            IOError: An error occurred accessing the output_directory or while writing to the JSON file.
            
        See Also:
            * :func:`save_analysis_properties`
        """
        analysis_props_file = os.path.join(self.output_directory,
                                           f"{self.output_filename_prefix}_desc-{self.analysis_props['MethodName']}_props.json")
        with open(analysis_props_file, 'w',encoding='utf-8') as f:
            json.dump(obj=self.analysis_props, fp=f, indent=4)
