from typing import Tuple

import numpy as np
from pandas import read_csv
from scipy.interpolate import interp1d as sp_interp
from scipy.optimize import curve_fit as sp_fit


def extract_blood_input_function_from_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Given a CSV file which contains blood activity data, we extract the times and the activity
    
    Assumes that the file is comma-separated and has columns: ID, TIME, UNCORRECTED ACTIVITY, CORRECTED ACTIVITY
    
    :param file_path:
    :return: [times, activity]
    """
    blood_data = read_csv(file_path)
    times, activity = blood_data.to_numpy().T[[1, 3]]
    return times, activity


def extract_blood_input_function_times_from_csv(file_path: str) -> np.ndarray:
    """Given a CSV file which contains blood activity data, we extract just the times.
    
    Assumes that the file is comma-separated and has columns: ID, TIME, UNCORRECTED ACTIVITY, CORRECTED ACTIVITY
    
    :param file_path:
    :return: [times, activity]
    """
    blood_data = read_csv(file_path)
    times = blood_data.to_numpy().T[[1]]
    return times


def extract_blood_input_function_activity_from_csv(file_path: str) -> np.ndarray:
    """Given a CSV file which contains blood activity data, we extract just the activities.
    
    Assumes that the file is comma-separated and has columns: ID, TIME, UNCORRECTED ACTIVITY, CORRECTED ACTIVITY
    
    :param file_path:
    :return: activity
    """
    blood_data = read_csv(file_path)
    activity = blood_data.to_numpy().T[[3]]
    return activity


# TODO: Maybe a class that tracks unitful quantities so we don't have to worry about units
class BloodInputFunction(object):
    """A general purpose class to deal with blood input function related data. The primarily functionality is to be able to
    compute the blood input function at any time, given the raw time and activity data.
    
    Using a manual threshold, we split the raw data into two parts:
    - Below the threshold, we have a simple linear interpolation.
    - Above the threshold, we fit a line to the data.
    
    When the object is instantiated, we automatically find the interpolation and the fit. Then, we can simply call the
    `calc_blood_input_function` function to give us blood input function values at any time.
    
    Lastly, note that this class can be used to deal with any type of (t,y) data where we wish to interpolate the first
    half, and fit a line to the second half.
    """
    
    def __init__(self, time: np.ndarray, activity: np.ndarray, thresh_in_mins: float):
        """Given the input time, activity, and threshold, we calculate the interpolating function for the first half (before
        the threshold) and the linear fit for the data in the second half (after the threshold). The threshold corresponds
        to the time, and not the activity.
        
        
        Currently, there must be at least 3 data points beyond the threshold to fit a line; else, we raise an `AssertionError`
        
        :param time: Time, in minutes.
        :param activity: Activity, assumed to be decay-corrected, corresponding to the times.
        :param thresh_in_mins: The threshold time, in minutes, such that before thresh we use an interpolant, and after thresh we use a linear fit.
        """
        assert time.shape == activity.shape, "`time` and `activity` must have the same dimensions."
        assert np.sum(time >= thresh_in_mins) >= 3, "Need at least 3 data-points above `thresh` to fit a line"
        self.thresh = thresh_in_mins
        below_thresh = time < self.thresh
        above_thresh = time >= self.thresh
        
        self._raw_times = time
        self._raw_activity = activity
        
        self.below_func = sp_interp(x=time[below_thresh], y=activity[below_thresh], assume_sorted=True, kind='linear',
                                    fill_value='extrapolate')
        
        self.above_func = BloodInputFunction.linear_fitting_func(x_data=time[above_thresh],
                                                                 y_data=activity[above_thresh])
    
    def calc_blood_input_function(self, t: np.ndarray) -> np.ndarray:
        """
        Given new time data, assumed to be in minutes, we calculate the blood input on those times.
        
        :param t:
        :return: ndarray of blood input values at the provided times
        """
        y = np.zeros_like(t)
        below_thresh = t < self.thresh
        above_thresh = t >= self.thresh
        
        y[below_thresh] = self.below_func(t[below_thresh])
        y[above_thresh] = self.above_func(t[above_thresh])
        
        return y
    
    @staticmethod
    def _linear_function(x: np.ndarray, m: float, b: float) -> np.ndarray:
        """ Simple equation for a line. `y = m * x + b`
        
        :param x: Independent variable
        :param m: Slope of the line
        :param b: Intercept of the line
        :return: m * x + b
        """
        return m * x + b
    
    @staticmethod
    def linear_fitting_func(x_data: np.ndarray, y_data: np.ndarray):
        """ Given x-data and y-data, we return a function corresponding to the linear fit.
        
        :param x_data: Independent variable
        :param y_data: Dependent variable corresponding to the x_data
        :return: A callable function that takes x-data as an input to compute the line values
        """
        
        popt, _ = sp_fit(f=BloodInputFunction._linear_function, xdata=x_data, ydata=y_data, full_output=False)
        
        def fitted_line_function(x):
            return BloodInputFunction._linear_function(x, *popt)
        
        return fitted_line_function