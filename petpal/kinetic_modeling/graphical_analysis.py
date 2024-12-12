"""
This module provides functions and a key class, :class:`GraphicalAnalysis`, for performing graphical analysis on Time
Activity Curve (TAC) data. It heavily utilizes Numpy and supports various analysis methods like Patlak, Logan, and
alternative Logan analysis.

The :class:`GraphicalAnalysis` class encapsulates the main functionality of the module. It provides an organized way to
perform graphical analysis where it initializes with paths to input data and output details, runs an analysis using a
specific method, calculates the best fit parameters, computes properties related to the fitting process, and stores the
analysis results. All these properties and results are stored within an instance's 'analysis_props' dictionary,
providing an organized storage of result of an analysis task.

TODO:
    * Check if it makes more sense to lift out the more mathy methods out into a separate module.
    * Add references for the TCMs and Patlak. Could maybe rely on Turku PET Center.
    * Handle cases when tac_vals = 0.0. Might be able to use t_thresh so that we are past the 0-values.
    * Add more detailed documentation to patlak analysis.
    
"""

from typing import Callable, Tuple
import os
import json
import numba
import numpy as np
from ..utils.time_activity_curve import MultiTACAnalysisMixin
from ..utils.image_io import safe_load_tac


@numba.njit()
def _line_fitting_make_rhs_matrix_from_xdata(xdata: np.ndarray) -> np.ndarray:
    """Generates the RHS matrix for linear least squares fitting

    Args:
        xdata (np.ndarray): array of independent variable values

    Returns:
        np.ndarray: 2D matrix where first column is `xdata` and the second column is 1s.
        
    """
    out_matrix = np.ones((len(xdata), 2), float)
    out_matrix[:, 0] = xdata
    return out_matrix


@numba.njit()
def fit_line_to_data_using_lls(xdata: np.ndarray, ydata: np.ndarray) -> np.ndarray:
    """Find the linear least squares solution given the x and y variables.
    
    Performs a linear least squares fit to the provided data. Explicitly calls numpy's ``linalg.lstsq`` method by
    constructing the matrix equations. We assume that ``xdata`` and ``ydata`` have the same number of elements.
    
    Args:
        xdata: Array of independent variable values
        ydata: Array of dependent variable values

    Returns:
       Linear least squares solution. (m, b) values
       
    """
    make_2d_matrix = _line_fitting_make_rhs_matrix_from_xdata
    matrix = make_2d_matrix(xdata)
    fit_ans = np.linalg.lstsq(matrix, ydata)[0]
    return fit_ans


@numba.njit()
def fit_line_to_data_using_lls_with_rsquared(xdata: np.ndarray, ydata: np.ndarray) -> Tuple[float, float, float]:
    """Fits a line to the data using least squares and explicitly computes the r-squared value.

    Args:
        xdata (np.ndarray): X-coordinates of the data.
        ydata (np.ndarray): Y-coordinates of the data.

    Returns:
        tuple: A tuple containing three float values: the intercept of the fitted line, the slope of the fitted line,
        and the r-squared value.
    """
    make_2d_matrix = _line_fitting_make_rhs_matrix_from_xdata
    matrix = make_2d_matrix(xdata)
    fit_ans = np.linalg.lstsq(matrix, ydata)
    
    ss_res = fit_ans[1][0]
    ss_tot = np.sum((np.mean(ydata) - ydata) ** 2.)
    r_squared = 1.0 - ss_res / ss_tot
    return fit_ans[0][0], fit_ans[0][1], r_squared


@numba.njit()
def cumulative_trapezoidal_integral(xdata: np.ndarray, ydata: np.ndarray, initial: float = 0.0) -> np.ndarray:
    """Calculates the cumulative integral of `ydata` over `xdata` using the trapezoidal rule.
    
    This function is based `heavily` on the :py:func:`scipy.integrate.cumulative_trapezoid` implementation
    (`source <https://github.com/scipy/scipy/blob/v1.12.0/scipy/integrate/_quadrature.py#L432-L536>`_).
    This implementation only works for 1D arrays and was implemented to work with :mod:`numba`.
    
    Args:
        xdata (np.ndarray): Array for the integration coordinate.
        ydata (np.ndarray): Array for the values to integrate

    Returns:
        (np.ndarray): Cumulative integral of ``ydata`` over ``xdata`` using the trapezoidal rule.
    """
    dx = np.diff(xdata)
    cum_int = np.zeros(len(xdata))
    cum_int[0] = initial
    cum_int[1:] = np.cumsum(dx * (ydata[1:] + ydata[:-1]) / 2.0)
    
    return cum_int


@numba.njit()
def calculate_patlak_x(tac_times: np.ndarray, tac_vals: np.ndarray) -> np.ndarray:
    r"""Calculates the x-variable in Patlak analysis :math:`\left(\frac{\int_{0}^{T}f(t)\mathrm{d}t}{f(T)}\right)`.
    
    Patlak-Gjedde analysis is a linearization of the 2-TCM with irreversible uptake in the second compartment.
    Therefore, we essentially have to fit a line to some data :math:`y = mx+b`. This function calculates the :math:`x`
    variable for Patlak analysis where,
    
    .. math::
    
        x = \frac{\int_{0}^{T} C_\mathrm{P}(t) \mathrm{d}t}{C_\mathrm{P}(t)},
    
    where further :math:`C_\mathrm{P}` is usually the plasma TAC.
    
    Args:
        tac_times (np.ndarray): Array containing the sampled times.
        tac_vals (np.ndarray): Array for activity values at the sampled times.

    Returns:
        (np.ndarray): Patlak x-variable. Cumulative integral of activity divided by activity.
        
    """
    cumulative_integral = cumulative_trapezoidal_integral(xdata=tac_times, ydata=tac_vals, initial=0.0)
    
    return cumulative_integral / tac_vals


@numba.njit()
def get_index_from_threshold(times_in_minutes: np.ndarray, t_thresh_in_minutes: float) -> int:
    """Get the index after which all times are greater than the threshold.

    Args:
        times_in_minutes (np.ndarray): Array containing times in minutes.
        t_thresh_in_minutes (float): Threshold time in minutes.

    Returns:
        int: Index for first time greater than or equal to the threshold time.
        
    Notes:
        If the threshold value is larger than the array, we return -1.
        
    """
    if t_thresh_in_minutes > np.max(times_in_minutes):
        return -1
    else:
        return np.argwhere(times_in_minutes >= t_thresh_in_minutes)[0, 0]


@numba.njit()
def patlak_analysis(tac_times_in_minutes: np.ndarray,
                    input_tac_values: np.ndarray,
                    region_tac_values: np.ndarray,
                    t_thresh_in_minutes: float) -> np.ndarray:
    """
    Performs Patlak analysis given the input TAC, region TAC, times and threshold.
    
    Args:
        tac_times_in_minutes (np.ndarray): Array of times in minutes.
        input_tac_values (np.ndarray): Array of input TAC values
        region_tac_values (np.ndarray): Array of ROI TAC values
        t_thresh_in_minutes (np.ndarray): Threshold time in minutes. Line is fit for all values after the threshold.

    Returns:
        (np.ndarray): Array containing :math:`(K_{i}, V_{0})` values.
    
    .. important::
        * We assume that the input TAC and ROI TAC values are sampled at the same times.
    
    """
    non_zero_indices = np.argwhere(input_tac_values != 0.).T[0]

    if len(non_zero_indices) <= 2:
        return np.asarray([np.nan, np.nan])

    t_thresh = get_index_from_threshold(times_in_minutes=tac_times_in_minutes[non_zero_indices],
                                        t_thresh_in_minutes=t_thresh_in_minutes)

    if len(tac_times_in_minutes[non_zero_indices][t_thresh:]) <= 2:
        return np.asarray([np.nan, np.nan])

    patlak_x = calculate_patlak_x(tac_times=tac_times_in_minutes[non_zero_indices],
                                  tac_vals=input_tac_values[non_zero_indices])
    patlak_y = region_tac_values[non_zero_indices] / input_tac_values[non_zero_indices]

    patlak_values = fit_line_to_data_using_lls(xdata=patlak_x[t_thresh:], ydata=patlak_y[t_thresh:])

    return patlak_values


@numba.njit()
def patlak_analysis_with_rsquared(tac_times_in_minutes: np.ndarray,
                                  input_tac_values: np.ndarray,
                                  region_tac_values: np.ndarray,
                                  t_thresh_in_minutes: float) -> Tuple[float, float, float]:
    """Performs Patlak analysis given the input TAC, region TAC, times and threshold.

    Args:
        tac_times_in_minutes (np.ndarray): Array of times in minutes.
        input_tac_values (np.ndarray): Array of input TAC values
        region_tac_values (np.ndarray): Array of ROI TAC values
        t_thresh_in_minutes (np.ndarray): Threshold time in minutes. Line is fit for all values after the threshold.

    Returns:
        tuple: (slope, intercept, :math:`R^2`)

    .. important::
        * We assume that the input TAC and ROI TAC values are sampled at the same times.

    """
    non_zero_indices = np.argwhere(input_tac_values != 0.).T[0]

    if len(non_zero_indices) <= 2:
        return np.nan, np.nan, np.nan

    t_thresh = get_index_from_threshold(times_in_minutes=tac_times_in_minutes[non_zero_indices],
                                        t_thresh_in_minutes=t_thresh_in_minutes)

    if len(tac_times_in_minutes[non_zero_indices][t_thresh:]) <= 2:
        return np.nan, np.nan, np.nan

    patlak_x = calculate_patlak_x(tac_times=tac_times_in_minutes[non_zero_indices],
                                  tac_vals=input_tac_values[non_zero_indices])
    patlak_y = region_tac_values[non_zero_indices] / input_tac_values[non_zero_indices]

    patlak_values = fit_line_to_data_using_lls_with_rsquared(xdata=patlak_x[t_thresh:], ydata=patlak_y[t_thresh:])

    return patlak_values


@numba.njit
def logan_analysis(tac_times_in_minutes: np.ndarray,
                   input_tac_values: np.ndarray,
                   region_tac_values: np.ndarray,
                   t_thresh_in_minutes: float) -> np.ndarray:
    """Performs Logan analysis on given input TAC, regional TAC, times and threshold, considering non-zero values.

    This function is similar to :func:`logan_analysis_with_rsquared`, but avoids the issue of division by zero by
    only considering non-zero TAC values for the region TAC since it is in the denominator. If the number of non-zero
    indices is less than or equal to 2, or if the number of time points after the threshold is less than or equal to 2,
    the function returns an array of NaNs.

    Args:
        tac_times_in_minutes (np.ndarray): Array of times in minutes.
        input_tac_values (np.ndarray): Array of input TAC values.
        region_tac_values (np.ndarray): Array of ROI TAC values.
        t_thresh_in_minutes (np.ndarray): Threshold time in minutes. Line is fit for all values after the threshold.

    Returns:
        np.ndarray: Array of two elements - (slope, intercept) of the best-fit line to the given data.

    .. important::
        * The interpretation of the returned values depends on the underlying kinetic model.
        * We assume that the input TAC and ROI TAC values are sampled at the same times.
        
    """
    non_zero_indices = np.argwhere(region_tac_values != 0.).T[0]

    if len(non_zero_indices) <= 2:
        return np.asarray([np.nan, np.nan])

    t_thresh = get_index_from_threshold(times_in_minutes=tac_times_in_minutes[non_zero_indices],
                                        t_thresh_in_minutes=t_thresh_in_minutes)

    if len(tac_times_in_minutes[non_zero_indices][t_thresh:]) <= 2:
        return np.asarray([np.nan, np.nan])

    logan_x = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=input_tac_values)
    logan_y = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=region_tac_values)

    logan_x = logan_x[non_zero_indices][t_thresh:] / region_tac_values[non_zero_indices][t_thresh:]
    logan_y = logan_y[non_zero_indices][t_thresh:] / region_tac_values[non_zero_indices][t_thresh:]

    fit_ans = fit_line_to_data_using_lls(xdata=logan_x, ydata=logan_y)
    return fit_ans


@numba.njit()
def logan_analysis_with_rsquared(tac_times_in_minutes: np.ndarray,
                                 input_tac_values: np.ndarray,
                                 region_tac_values: np.ndarray,
                                 t_thresh_in_minutes: float) -> tuple[float, float, float]:
    """
    Performs Logan analysis on given input TAC, regional TAC, times and threshold.

    Args:
        tac_times_in_minutes (np.ndarray): Array of times in minutes.
        input_tac_values (np.ndarray): Array of input TAC values
        region_tac_values (np.ndarray): Array of ROI TAC values
        t_thresh_in_minutes (np.ndarray): Threshold time in minutes. Line is fit for all values after the threshold.

    Returns:
        tuple: (slope, intercept, :math:`R^2`)

    .. important::
        * The interpretation of the values depends on the underlying kinetic model.
        * We assume that the input TAC and ROI TAC values are sampled at the same times.
        
    """

    non_zero_indices = np.argwhere(region_tac_values != 0.).T[0]

    if len(non_zero_indices) <= 2:
        return np.nan, np.nan, np.nan

    t_thresh = get_index_from_threshold(times_in_minutes=tac_times_in_minutes[non_zero_indices],
                                        t_thresh_in_minutes=t_thresh_in_minutes)

    if len(tac_times_in_minutes[non_zero_indices][t_thresh:]) <= 2:
        return np.nan, np.nan, np.nan

    logan_x = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=input_tac_values)
    logan_y = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=region_tac_values)

    logan_x = logan_x[non_zero_indices][t_thresh:] / region_tac_values[non_zero_indices][t_thresh:]
    logan_y = logan_y[non_zero_indices][t_thresh:] / region_tac_values[non_zero_indices][t_thresh:]

    logan_values = fit_line_to_data_using_lls_with_rsquared(xdata=logan_x, ydata=logan_y)

    return logan_values


@numba.njit
def alternative_logan_analysis(tac_times_in_minutes: np.ndarray,
                               input_tac_values: np.ndarray,
                               region_tac_values: np.ndarray,
                               t_thresh_in_minutes: float) -> np.ndarray:
    """
    Performs Alternative Logan analysis on given input TAC, regional TAC, times and threshold, considering non-zero
    values for regional TAC.

    This function is similar to :func:`alternative_logan_analysis_with_rsquared`, but avoids the issue of division by
    zero by only considering non-zero TAC values for the region TAC since it is in the denominator. If the number of
    indices is less than or equal to 2, or if the number of time points after the threshold is less than or equal to 2,
    the function returns an array of NaNs.

    Args:
        tac_times_in_minutes (np.ndarray): Array of times in minutes.
        input_tac_values (np.ndarray): Array of input TAC values.
        region_tac_values (np.ndarray): Array of ROI TAC values.
        t_thresh_in_minutes (np.ndarray): Threshold time in minutes. Line is fit for all values after the threshold.

    Returns:
        np.ndarray: Array of two elements - (slope, intercept) of the best-fit line to the given data.

    .. important::
        * The interpretation of the returned values depends on the underlying kinetic model.
        * We assume that the input TAC and ROI TAC values are sampled at the same times.
        
    """
    non_zero_indices = np.argwhere(input_tac_values != 0.).T[0]

    if len(non_zero_indices) <= 2:
        return np.asarray([np.nan, np.nan])

    t_thresh = get_index_from_threshold(times_in_minutes=tac_times_in_minutes[non_zero_indices],
                                        t_thresh_in_minutes=t_thresh_in_minutes)

    if len(tac_times_in_minutes[non_zero_indices][t_thresh:]) <= 2:
        return np.asarray([np.nan, np.nan])

    alt_logan_x = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=input_tac_values)
    alt_logan_y = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=region_tac_values)

    alt_logan_x = alt_logan_x[non_zero_indices][t_thresh:] / input_tac_values[non_zero_indices][t_thresh:]
    alt_logan_y = alt_logan_y[non_zero_indices][t_thresh:] / input_tac_values[non_zero_indices][t_thresh:]

    fit_ans = fit_line_to_data_using_lls(xdata=alt_logan_x, ydata=alt_logan_y)
    return fit_ans


@numba.njit()
def alternative_logan_analysis_with_rsquared(tac_times_in_minutes: np.ndarray,
                                             input_tac_values: np.ndarray,
                                             region_tac_values: np.ndarray,
                                             t_thresh_in_minutes: float) -> tuple[float, float, float]:
    """Performs alternative logan analysis on given input TAC, regional TAC, times and threshold.

    Args:
        tac_times_in_minutes (np.ndarray): Array of times in minutes.
        input_tac_values (np.ndarray): Array of input TAC values
        region_tac_values (np.ndarray): Array of ROI TAC values
        t_thresh_in_minutes (np.ndarray): Threshold time in minutes. Line is fit for all values after the threshold.

    Returns:
        tuple: (slope, intercept, :math:`R^2`)

    .. important::
        * The interpretation of the values depends on the underlying kinetic model.
        * We assume that the input TAC and ROI TAC values are sampled at the same times.
        
    """

    non_zero_indices = np.argwhere(input_tac_values != 0.).T[0]

    if len(non_zero_indices) <= 2:
        return np.nan, np.nan, np.nan

    t_thresh = get_index_from_threshold(times_in_minutes=tac_times_in_minutes[non_zero_indices],
                                        t_thresh_in_minutes=t_thresh_in_minutes)

    if len(tac_times_in_minutes[non_zero_indices][t_thresh:]) <= 2:
        return np.nan, np.nan, np.nan

    alt_logan_x = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=input_tac_values)
    alt_logan_y = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=region_tac_values)

    alt_logan_x = alt_logan_x[non_zero_indices][t_thresh:] / input_tac_values[non_zero_indices][t_thresh:]
    alt_logan_y = alt_logan_y[non_zero_indices][t_thresh:] / input_tac_values[non_zero_indices][t_thresh:]

    alt_logan_values = fit_line_to_data_using_lls_with_rsquared(xdata=alt_logan_x,
                                                                ydata=alt_logan_y)

    return alt_logan_values


def get_graphical_analysis_method(method_name: str) -> Callable:
    """
    Function for obtaining the appropriate graphical analysis method.

    This function accepts a string specifying a graphical time-activity curve (TAC) analysis method. It returns a
    reference to the function that performs the selected analysis method.

    Args:
        method_name (str): The name of the graphical method. This should be one of the following strings: 'patlak',
                           'logan', or 'alt_logan'.

    Returns:
        function: A reference to the function that performs the corresponding graphical TAC analysis. The returned
        function will take arguments specific to the analysis method, such as input TAC values, tissue TAC values, TAC
        times in minutes, and threshold time in minutes.
    
    Raises:
        ValueError: If `method_name` is not one of the supported graphical analysis methods, i.e., 'patlak', 'logan', or
                    'alt_logan'.
    
    Example:
        .. code-block:: python
            
            from petpal.graphical_analysis import get_graphical_analysis_method as get_method
            from petpal.utils.image_io import safe_load_tac as load_tac
            
            selected_func = get_method('logan')
            input_tac_values, tac_times_in_minutes = load_tac("PATH/TO/PLASMA/TAC.tsv")
            tissue_tac_values, _ = load_tac("PATH/TO/ROI/TAC.tsv")
            
            results = selected_func(tac_times_in_minutes,
                                    input_tac_values,
                                    tissue_tac_values,
                                    t_thresh_in_minutes)
            print(results)
                                    
    """
    if method_name == "patlak":
        return patlak_analysis
    if method_name == "logan":
        return logan_analysis
    if method_name == "alt_logan":
        return alternative_logan_analysis
    raise ValueError(f"Invalid method_name! Must be either 'patlak', 'logan', or 'alt_logan'. Got {method_name}")


def get_graphical_analysis_method_with_rsquared(method_name: str) -> Callable:
    """
    Function for obtaining the appropriate graphical analysis method which also calculates the r-squared value.

    This function accepts a string specifying a graphical time-activity curve (TAC) analysis method. It returns a
    reference to the function that performs the selected analysis method.

    Args:
        method_name (str): The name of the graphical method. This should be one of the following strings: 'patlak',
                           'logan', or 'alt_logan'.

    Returns:
        function: A reference to the function that performs the corresponding graphical TAC analysis. The returned
        function will take arguments specific to the analysis method, such as input TAC values, tissue TAC values, TAC
        times in minutes, and threshold time in minutes.

    Raises:
        ValueError: If `method_name` is not one of the supported graphical analysis methods, i.e., 'patlak', 'logan', or
                    'alt_logan'.

    Example:
        .. code-block:: python
            
            from petpal.graphical_analysis import get_graphical_analysis_method_with_rsquared as get_method
            from petpal.graphical_analysis import safe_load_tac as load_tac
            
            selected_func = get_method('logan')
            input_tac_values, tac_times_in_minutes = load_tac("PATH/TO/PLASMA/TAC.tsv")
            tissue_tac_values, _ = load_tac("PATH/TO/ROI/TAC.tsv")
            
            results = selected_func(input_tac_values,
                                    tissue_tac_values,
                                    tac_times_in_minutes,
                                    t_thresh_in_minutes)
            print(results)
            
    """
    if method_name == "patlak":
        return patlak_analysis_with_rsquared
    if method_name == "logan":
        return logan_analysis_with_rsquared
    if method_name == "alt_logan":
        return alternative_logan_analysis_with_rsquared
    raise ValueError(f"Invalid method_name! Must be either 'patlak', 'logan', or 'alt_logan'. Got {method_name}")


class GraphicalAnalysis:
    """
    :class:`GraphicalAnalysis` to handle Graphical Analysis for time activity curve (TAC) data.

    The :class:`GraphicalAnalysis` class provides methods for managing TAC data analysis. The class is initialized
    with paths to input TAC data, region of interest (ROI) TAC data files, an output directory, and output filename prefix.

    Analysis is performed by specifying a method name and threshold time in the :func:`run_analysis` method. The results
    of the analysis are stored within the instance's 'analysis_props' dictionary.

    Key methods include:
    - :func:`init_analysis_props`: Initializes a dictionary with keys for analysis properties and default values.
    - :func:`run_analysis`: Runs the graphical analysis on the data using a specific method.
    - :func:`calculate_fit`: Calculates the best fit values for a given graphical analysis method.
    - :func:`calculate_fit_properties`: Calculates and stores the properties related to the fitting process.
    - :func:`save_analysis`: Stores the 'analysis_props' dictionary into a JSON file.

    Raises:
        RuntimeError: If the :func:`run_analysis` method has not been run before :func:`save_analysis` method.

    Attributes:
        input_tac_path (str): The path to input TAC data file.
        roi_tac_path (str): The path to ROI TAC data file.
        output_directory (str): Directory where the output should be saved.
        output_filename_prefix (str): Output filename prefix for saving the analysis.
        analysis_props (dict): Property dictionary used to store results of the analysis.
        method (str): The name of the graphical analysis method to be utilised.
        fit_thresh_in_mins (float): The fitting threshold time in minutes for the analysis method.
        self.analysis_func (Callable): The function used for performing the fitting.
        
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tac_path: str,
                 output_directory: str,
                 output_filename_prefix: str,
                 method: str,
                 fit_thresh_in_mins: float) -> None:
        """
        Initializes GraphicalAnalysis with provided paths and output details.

        Args:
            input_tac_path (str): The path to the file containing input Time Activity Curve (TAC) data.
            roi_tac_path (str): The path to the file containing Region of Interest (ROI) TAC data.
            output_directory (str): The directory where the output of the analysis should be saved.
            output_filename_prefix (str): The prefix for the name of output file(s).
            method (str): The name of the graphical analysis method to be utilised.
            fit_thresh_in_mins (float): The fitting threshold time in minutes for the analysis method.
    
        Returns:
            None
        """
        self.input_tac_path = os.path.abspath(input_tac_path)
        self.roi_tac_path = os.path.abspath(roi_tac_path)
        self.output_directory = os.path.abspath(output_directory)
        self.output_filename_prefix = output_filename_prefix
        self.analysis_props = self.init_analysis_props()
        self.method = method
        self.fit_thresh_in_mins = fit_thresh_in_mins
        self.analysis_func = get_graphical_analysis_method_with_rsquared(method_name=self.method)

    def init_analysis_props(self) -> dict:
        """
        Initializes analysis properties dictionary.

        This method initializes a dictionary with keys for all the analysis properties and default values set to None.
        The paths to the input TAC and ROI TAC files are set from the object's properties.

        Returns:
            dict: A dictionary with keys for each analysis property and default values. The keys include 'FilePathPTAC'
            (the file path to the input TAC file), 'FilePathTTAC' (the file path to the ROI TAC file), 'MethodName'
            (the name of the method used in the analysis), 'ThresholdTime' (the threshold time for the analysis),
            'StartFrameTime' and 'EndFrameTime' (the start and end times for the frame), 'NumberOfPointsFit' (The number
            of points used in the fit), 'Slope', 'Intercept' and 'RSquared' (the slope, intercept and R-squared of the fit).
        """
        props = {'FilePathPTAC': self.input_tac_path,
                 'FilePathTTAC': self.roi_tac_path,
                 'MethodName': None,
                 'ThresholdTime': None,
                 'StartFrameTime': None,
                 'EndFrameTime': None,
                 'NumberOfPointsFit': None,
                 'Slope': None,
                 'Intercept': None,
                 'RSquared': None}
        return props

    def run_analysis(self):
        """
        Runs the graphical analysis on the data using the specified method.

        This method is the main entry point to carry out the analysis. It executes the steps in order – first calculating
        the fit, then calculating the properties of the fit.

        Returns:
            None

        Side Effects:
            Computes and updates the analysis-related properties in the object based on the provided method and threshold.
        """
        self.calculate_fit()
        self.calculate_fit_properties()

    def calculate_fit(self):
        """
        Calculates the best fit parameters for a graphical analysis method.
    
        This method applies the specified graphical analysis method to the Time Activity Curve (TAC) data and stores
        the slope, intercept, and r-squared values of the fit in the analysis properties.
    
        Args:
            method_name (str): The name of the graphical analysis method for which the fit should be calculated.
            t_thresh_in_mins (float): The threshold time in minutes for the analysis method.
    
        Returns:
            None
            
        Side Effect:
            Updates 'Slope', 'Intercept', and 'RSquared' in self.analysis_props dictionary with calculated fit parameters.

        """
        p_tac_times, p_tac_vals = safe_load_tac(self.input_tac_path)
        _t_tac_times, t_tac_vals = safe_load_tac(self.roi_tac_path)
        slope, intercept, rsquared = self.analysis_func(tac_times_in_minutes=p_tac_times,
                                                        input_tac_values=p_tac_vals,
                                                        region_tac_values=t_tac_vals,
                                                        t_thresh_in_minutes=self.fit_thresh_in_mins)
        self.analysis_props['Slope'] = slope
        self.analysis_props['Intercept'] = intercept
        self.analysis_props['RSquared'] = rsquared

    def calculate_fit_properties(self):
        """
        Calculates and stores the properties related to the fitting process.

        This method calculates several properties related to the fitting process, including the threshold time, the name
        of the method used, the start and end frame time, and the number of points used in the fit. These values are
        stored in the instance's `analysis_props` variable.

        Parameters:
            method_name (str): The name of the graphical method being used.
            t_thresh_in_mins (float): The threshold time (in minutes) used in the fitting process.

        Note:
            This method relies on the :func:`safe_load_tac` function to load time-activity curve (TAC) data from the
            file at ``input_tac_path``, and the :func:`get_index <get_index_from_threshold>`
            function to get the index from the threshold time.

        See also:
            * :func:`safe_load_tac`: Function to safely load TAC data from a file.
            * :func:`get_index_from_threshold`: Function to get the index from the threshold time.

        Returns:
            None. The results are stored within the instance's `analysis_props` variable.
        """
        self.analysis_props['ThresholdTime'] = self.fit_thresh_in_mins
        self.analysis_props['MethodName'] = self.method
        p_tac_times, _ = safe_load_tac(filename=self.input_tac_path)
        t_thresh_index = get_index_from_threshold(times_in_minutes=p_tac_times, t_thresh_in_minutes=self.fit_thresh_in_mins)
        self.analysis_props['StartFrameTime'] = p_tac_times[t_thresh_index]
        self.analysis_props['EndFrameTime'] = p_tac_times[-1]
        self.analysis_props['NumberOfPointsFit'] = len(p_tac_times[t_thresh_index:])

    def save_analysis(self):
        """
        Saves the analysis properties to a JSON file.

        This method saves the 'analysis_props' dictionary to a JSON file. The file is saved in the specified output directory
        under a filename constructed from the 'output_filename_prefix' and the method name used in analysis. Before executing,
        the method checks if analysis has been run by verifying if 'RSquared' in 'analysis_props' is not None.
    
        Raises:
            RuntimeError: If the 'run_analysis' method was not called before 'save_analysis' is invoked.
    
        Returns:
            None
        """
        if self.analysis_props['RSquared'] is None:
            raise RuntimeError("'run_analysis' method must be called before 'save_analysis'.")
        file_name_prefix = os.path.join(self.output_directory,
                                        f"{self.output_filename_prefix}_desc-{self.analysis_props['MethodName']}")
        analysis_props_file = f"{file_name_prefix}_fitprops.json"
        with open(analysis_props_file, 'w',encoding='utf-8') as f:
            json.dump(obj=self.analysis_props, fp=f, indent=4)

    def __call__(self):
        """
        Runs :meth:`run_analysis` and :meth:`save_analysis` to run the analysis and save the analysis properties.
        
        """
        self.run_analysis()
        self.save_analysis()


class MultiTACGraphicalAnalysis(GraphicalAnalysis, MultiTACAnalysisMixin):
    """
    A class that performs graphical analysis on multiple tissue TACs (Time Activity Curves).

    Attributes:
        input_tac_path (str): Path to the input TAC file.
        roi_tacs_dir (str): Directory containing region of interest TAC files.
        output_directory (str): Directory for saving analysis results.
        output_filename_prefix (str): Prefix for output filenames.
        method (str): Method used for analysis.
        fit_thresh_in_mins (Optional[float]): Threshold in minutes for fit calculation.
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir:str,
                 output_directory: str,
                 output_filename_prefix: str,
                 method:str,
                 fit_thresh_in_mins=None,):
        """
        Initializes the MultiTACGraphicalAnalysis object with required paths, method, and threshold.

        Args:
            input_tac_path (str): Path to the input TAC file.
            roi_tacs_dir (str): Directory containing region of interest TAC files.
            output_directory (str): Directory for saving analysis results.
            output_filename_prefix (str): Prefix for output filenames.
            method (str): Method used for analysis.
            fit_thresh_in_mins (Optional[float], optional): Threshold in minutes for fit calculation. Defaults to None.
        """
        MultiTACAnalysisMixin.__init__(self,
                                       input_tac_path=input_tac_path,
                                       tacs_dir=roi_tacs_dir)
        GraphicalAnalysis.__init__(self,
                                   input_tac_path=input_tac_path,
                                   roi_tac_path=roi_tacs_dir,
                                   output_directory=output_directory,
                                   output_filename_prefix=output_filename_prefix,
                                   method=method,
                                   fit_thresh_in_mins=fit_thresh_in_mins
                                   )

    def init_analysis_props(self):
        """
        Initializes analysis properties for each tissue TAC. Overrides
        :meth:`GraphicalAnalysis.init_analysis_props`.

        Returns:
            list[dict]: A list of analysis property dictionaries for each TAC.
        """
        num_of_tacs = self.num_of_tacs
        analysis_props = [GraphicalAnalysis.init_analysis_props(self) for a_tac in range(num_of_tacs)]
        for tac_id, a_prop_dict in enumerate(analysis_props):
            a_prop_dict['FilePathTTAC'] = os.path.abspath(self.tacs_files_list[tac_id])
        return analysis_props


    def calculate_fit(self):
        """
        Calculates the fit for each TAC, updating the analysis properties with slope, intercept, and R-squared values.
        Overrides :meth:`GraphicalAnalysis.calculate_fit`.
        """
        p_tac_times, p_tac_vals = safe_load_tac(self.input_tac_path)
        for tac_id, a_tac in enumerate(self.tacs_files_list):
            _, t_tac_vals = safe_load_tac(a_tac)
            try:
                slope, intercept, rsquared = self.analysis_func(tac_times_in_minutes=p_tac_times,
                                                                input_tac_values=p_tac_vals,
                                                                region_tac_values=t_tac_vals,
                                                                t_thresh_in_minutes=self.fit_thresh_in_mins)
            except np.linalg.LinAlgError:
                slope, intercept, rsquared = np.nan, np.nan, np.nan
            self.analysis_props[tac_id]['Slope'] = slope
            self.analysis_props[tac_id]['Intercept'] = intercept
            self.analysis_props[tac_id]['RSquared'] = rsquared

    def calculate_fit_properties(self):
        """
        Calculates additional properties of the fit, such as threshold, method name,
        start and end times, and number of points. Iterates over all the TACs.
        
        Overrides :meth:`GraphicalAnalysis.calculate_fit_properties`
        """
        p_tac_times, _ = safe_load_tac(self.input_tac_path)
        t_thresh_index = get_index_from_threshold(times_in_minutes=p_tac_times,
                                                  t_thresh_in_minutes=self.fit_thresh_in_mins)
        points_fit = len(p_tac_times[t_thresh_index:])
        start_time=p_tac_times[t_thresh_index]
        end_time=p_tac_times[-1]

        for tac_id, a_tac in enumerate(self.tacs_files_list):
            self.analysis_props[tac_id]['ThresholdTime'] = self.fit_thresh_in_mins
            self.analysis_props[tac_id]['MethodName'] = self.method
            self.analysis_props[tac_id]['StartFrameTime'] = start_time
            self.analysis_props[tac_id]['EndFrameTime'] = end_time
            self.analysis_props[tac_id]['NumberOfPointsFit'] = points_fit

    def save_analysis(self):
        """
        Saves the analysis results to a JSON file for each segment.

        Raises:
            RuntimeError: If 'run_analysis' method has not been called before save_analysis.
        """
        if self.analysis_props[0]['RSquared'] is None:
            raise RuntimeError("'run_analysis' method must be called before 'save_analysis'.")

        for seg_name, fit_props in zip(self.inferred_seg_labels, self.analysis_props):
            filename = [self.output_filename_prefix,
                        f'desc-{self.method}',
                        f'seg-{seg_name}',
                        'fitprops.json']
            filename = '_'.join(filename)
            filepath = os.path.join(self.output_directory, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(obj=fit_props, fp=f, indent=4)
