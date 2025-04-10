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
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d as scipy_interpolate
from scipy.signal import convolve as scipy_convolve


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

    def __len__(self) -> int:
        """
        Returns the number of time points in the time-activity curve (TAC).

        This method provides the length of the `times` attribute, representing the
        number of discrete time points associated with the TAC.

        Returns:
            int: The number of time points in the TAC.

        Example:
            .. code-block:: python

                from petpal.utils.time_activity_curve import TimeActivityCurve

                # Create a TimeActivityCurve object
                my_tac = TimeActivityCurve(
                    times=np.array([0, 10, 20, 30]),
                    activity=np.array([1.2, 2.3, 3.4, 4.5])
                )

                # Get the number of time points
                print(len(my_tac))  # Output: 4
        """
        return len(self.times)

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

        Reads a TSV file using :func:`~.safe_load_tac` and instantiates a TimeActivityCurve.

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
                respectively. See :func:`~.safe_write_tac`. Default None.
        """
        safe_write_tac(filename=filename,tac_data=self.tac_werr,col_names=col_names)

    def set_activity_non_negative(self) -> 'TimeActivityCurve':
        r"""
        Ensures that the time-activity curve (TAC) data is physically consistent.

        This method modifies the TAC object in place by setting `uncertainty` to `NaN`
        and `activity` to `0` for time points where the `activity` values are negative.
        Such adjustments help maintain data consistency for downstream analysis.

        Returns:
            TimeActivityCurve: The updated TAC object with sanitized data.

        Note:
            The method returns self to allow for `.`-chaining.

        Example:
            .. code-block:: python

                from petpal.utils.time_activity_curve import TimeActivityCurve

                # Create a TimeActivityCurve object with negative activity
                my_tac = TimeActivityCurve(
                    times=np.array([0, 10, 20, 30]),
                    activity=np.array([1.2, -2.3, 3.4, -4.5]),
                    uncertainty=np.array([0.1, 0.2, 0.3, 0.4])
                )

                # Sanitize the TAC
                my_tac.set_activity_non_negative()

                print(my_tac.activity)  # Output: [1.2, 0.0, 3.4, 0.0]
                print(my_tac.uncertainty)  # Output: [0.1, NaN, 0.3, NaN]
        """
        self.uncertainty[self.activity < 0] = np.nan
        self.activity[self.activity < 0] = 0
        return self

    def evenly_resampled_tac(self, num_samples: int = 4096) -> 'TimeActivityCurve':
        r"""
        Generates a time-activity curve (TAC) resampled at evenly spaced time points.

        This method uses linear interpolation to recreate the TAC with a specified
        number of evenly spaced samples between the initial and final time points.
        The resulting TAC is sanitized to ensure physical consistency. Uses
        :class:`scipy.interpolate.interp1d` for the interpolation
        with ``kind=='linear'`` and ``fill_value='extrapolate'``.

        .. important::
            If the TAC will be used for convolution later, prefer powers of two for
            the number of samples.

        Args:
            num_samples (int, optional): The number of time points in the resampled TAC.
                Must be greater than 2. Defaults to 4096.

        Returns:
            TimeActivityCurve: A new `TimeActivityCurve` instance with evenly spaced
            time points and resampled activity values.

        Raises:
            AssertionError: If `num_samples` is less than or equal to 2.

        Example:
            .. code-block:: python

                from petpal.utils.time_activity_curve import TimeActivityCurve

                # Create a TimeActivityCurve object
                my_tac = TimeActivityCurve(
                    times=np.array([0, 10, 20, 30]),
                    activity=np.array([1.0, 2.0, 3.0, 4.0])
                )

                # Resample the TAC with evenly spaced time points
                resampled_tac = my_tac.evenly_resampled_tac(num_samples=10)

                print(resampled_tac.times)  # Output: Array of 10 evenly spaced times
                print(resampled_tac.activity)  # Output: Interpolated activity values
        """
        assert num_samples > 2, "Number of samples must be larger than 2."
        new_times = np.linspace(0, self.times[-1], num_samples, dtype=float)
        new_activity = scipy_interpolate(*self.tac, kind='linear', fill_value='extrapolate')(new_times)
        new_tac = TimeActivityCurve(new_times, new_activity)
        new_tac.set_activity_non_negative()
        return new_tac

    def evenly_resampled_tac_given_dt(self, dt: float = 0.1/60.0) -> 'TimeActivityCurve':
        """
        Generates a time-activity curve (TAC) resampled at evenly spaced time intervals.

        This method calculates the number of samples required to achieve the specified
        time interval (`dt`) and then resamples the TAC using linear interpolation. The
        resulting TAC is sanitized to ensure physical consistency.

        Args:
            dt (float, optional): The desired time interval between consecutive
                resampled time points (in the same unit as `times`). Must be greater than 0.
                Defaults to 0.1 / 60.0 (approximately 0.00167).

        Returns:
            TimeActivityCurve: A new `TimeActivityCurve` instance with evenly spaced
            time intervals and resampled activity values.

        Raises:
            AssertionError: If `dt` is less than or equal to 0.

        Example:
            .. code-block:: python

                from petpal.utils.time_activity_curve import TimeActivityCurve

                # Create a TimeActivityCurve object
                my_tac = TimeActivityCurve(
                    times=np.array([0, 10, 20, 30]),
                    activity=np.array([1.0, 2.0, 3.0, 4.0])
                )

                # Resample the TAC with a given time interval (dt)
                resampled_tac = my_tac.evenly_resampled_tac_given_dt(dt=0.1)

                print(resampled_tac.times)  # Output: Evenly spaced time points with interval dt
                print(resampled_tac.activity)  # Output: Interpolated activity values
        """
        assert dt > 0, "dt must be larger than 0."
        num_samples = 1+int(self.times[-1]/dt)
        return self.evenly_resampled_tac(num_samples=num_samples)

    def resampled_tac_on_times(self, new_times: np.ndarray) -> 'TimeActivityCurve':
        """
        Resamples the time-activity curve (TAC) on specified time points.

        This method uses linear interpolation to compute activity values at the
        provided time points (`new_times`). The resulting TAC is sanitized to
        ensure physical consistency.

        Args:
            new_times (np.ndarray): An array of time points where the TAC should
                be resampled. Must be a 1D array of monotonically increasing values.

        Returns:
            TimeActivityCurve: A new `TimeActivityCurve` instance with the specified
            time points and interpolated activity values.

        Example:
            .. code-block:: python

                from petpal.utils.time_activity_curve import TimeActivityCurve
                import numpy as np

                # Create a TimeActivityCurve object
                my_tac = TimeActivityCurve(
                    times=np.array([0, 10, 20, 30]),
                    activity=np.array([1.0, 2.0, 3.0, 4.0])
                )

                # New time points for resampling
                new_times = np.array([5, 15, 25])

                # Resample TAC on new time points
                resampled_tac = my_tac.resampled_tac_on_times(new_times=new_times)

                print(resampled_tac.times)  # Output: [5, 15, 25]
                print(resampled_tac.activity)  # Output: Interpolated activity values at [5, 15, 25]
        """
        new_values = scipy_interpolate(self.times, self.activity, kind='linear', fill_value='extrapolate')(new_times)
        out_tac = TimeActivityCurve(new_times, new_values)
        out_tac.set_activity_non_negative()
        return out_tac

    def add_zero_time_and_activity(self):
        """
        Ensures the time-activity curve (TAC) starts at time 0 with zero activity.

        If the first time point in the TAC is not 0.0, this method prepends a time
        point of 0.0 and assigns it an activity value of 0. The associated uncertainty
        for this time point is set to `NaN`. The method modifies the TAC in place
        and returns the updated instance.

        Returns:
            TimeActivityCurve: The updated `TimeActivityCurve` instance with
            0.0 prepended to time, activity, and uncertainty arrays (if needed).

        Example:
            .. code-block:: python

                from petpal.utils.time_activity_curve import TimeActivityCurve
                import numpy as np

                # Create a TimeActivityCurve object
                my_tac = TimeActivityCurve(
                    times=np.array([10, 20, 30]),
                    activity=np.array([2.0, 3.0, 4.0]),
                    uncertainty=np.array([0.1, 0.2, 0.3])
                )

                # Add 0 time and activity if missing
                my_tac = my_tac.add_zero_time_and_activity()

                print(my_tac.times)       # Output: [ 0, 10, 20, 30 ]
                print(my_tac.activity)   # Output: [ 0, 2.0, 3.0, 4.0 ]
                print(my_tac.uncertainty)  # Output: [ NaN, 0.1, 0.2, 0.3 ]
        """
        if self.times[0] != 0.0:
            self.times = np.append(0, self.times)
            self.activity = np.append(0, self.activity)
            self.uncertainty = np.append(np.nan, self.uncertainty)
        return self

    def shifted_tac(self, shift_in_mins: float = 10.0/60.0, dt: float | None = None) -> 'TimeActivityCurve':
        """
        Returns a time-activity curve (TAC) shifted in time by a specified amount.

        If the shift is positive, the TAC is left-shifted; if negative, the TAC is
        right-shifted. The shift can be applied at the existing time points of the
        TAC or at oversampled points, depending on the value of `dt`.

        Args:
            shift_in_mins (float, optional):
                The amount (in minutes) to shift the TAC. Positive values indicate a left shift;
                negative values indicate a right shift. Defaults to 10.0 / 60.0 (10 seconds).
            dt (float | None, optional):
                The time interval for oversampling. If `None`, the TAC is resampled at its original
                time points. If a value is provided, the TAC is resampled at an evenly spaced interval of `dt`.

        Returns:
            TimeActivityCurve: A new `TimeActivityCurve` instance with the shifted activity values.

        Raises:
            AssertionError: If `dt` is equal to 0.

        Example:
            .. code-block:: python

                # Left-shifting the TAC
                shifted_tac = my_tac.shifted_tac(shift_in_mins=5.0 / 60.0)

                # Right-shifting the TAC with oversampling
                shifted_tac = my_tac.shifted_tac(shift_in_mins=-2.0 / 60.0, dt=0.1 / 60.0)

        See Also:
            * :meth:`left_shifted_tac`
            * :meth:`right_shifted_tac`

        """
        assert dt != 0, "dt must be strictly larger than 0."
        if shift_in_mins < 0:
            return TimeActivityCurve.right_shifted_tac(tac=self, shift_in_mins=shift_in_mins, dt=dt)
        else:
            return TimeActivityCurve.left_shifted_tac(tac=self, shift_in_mins=shift_in_mins, dt=dt)

    @staticmethod
    def left_shifted_tac(tac: 'TimeActivityCurve',
                         shift_in_mins: float = 10.0 / 60.0,
                         dt: float | None = 0.1 / 60.0) -> 'TimeActivityCurve':
        """
        Produces a TAC left-shifted by a specified amount of time.

        The left shift moves activity values earlier in time and fills the trailing
        values with interpolated data. The shift can either be applied on the
        original time points or on a time grid determined by `dt`.

        Args:
            tac (TimeActivityCurve): The original TAC to be shifted.
            shift_in_mins (float, optional):
                The amount (in minutes) to left-shift the TAC. Must be larger than 0.
                Defaults to 10.0 / 60.0 (10 seconds).
            dt (float | None, optional):
                The time interval for oversampling. If `None`, the TAC is resampled at its original
                time points. If a value is provided, the TAC is resampled at an evenly spaced interval of `dt`.

        Returns:
            TimeActivityCurve: A new `TimeActivityCurve` instance with the left-shifted activity values.

        Raises:
            AssertionError: If `shift_in_mins` is not greater than 0.

        Example:
            .. code-block:: python

                # Perform a left shift on the TAC
                shifted_tac = TimeActivityCurve.left_shifted_tac(tac=my_tac, shift_in_mins=5.0 / 60.0)

                # Perform a left shift with oversampling
                shifted_tac = TimeActivityCurve.left_shifted_tac(tac=my_tac, shift_in_mins=2.0 / 60.0, dt=0.05 / 60.0)
        """
        assert shift_in_mins > 0, "shift_in_mins must be larger than 0."
        if dt is None:
            even_tac = tac.evenly_resampled_tac()
        else:
            even_tac = tac.evenly_resampled_tac_given_dt(dt=dt)
        delta_t = even_tac.times[1] - even_tac.times[0] if dt is None else dt

        shift_ind = int(shift_in_mins / delta_t)
        shifted_vals = np.zeros_like(even_tac.activity)
        shifted_vals[:-shift_ind] = even_tac.activity[shift_ind:]
        shifted_vals[-shift_ind:] = scipy_interpolate(even_tac.times[:-shift_ind],
                                                      shifted_vals[:-shift_ind],
                                                      kind='linear',
                                                      fill_value='extrapolate')(even_tac.times[-shift_ind:])
        shifted_tac = TimeActivityCurve(even_tac.times, shifted_vals)
        shifted_tac.set_activity_non_negative()
        if dt is None:
            shifted_vals_on_tac_times = scipy_interpolate(*shifted_tac.tac,
                                                        kind='linear',
                                                        fill_value='extrapolate')(tac.times)
            return TimeActivityCurve(tac.times, shifted_vals_on_tac_times)
        else:
            return shifted_tac

    @staticmethod
    def right_shifted_tac(tac: 'TimeActivityCurve',
                          shift_in_mins: float = 10.0 / 60.0,
                          dt: float | None = 0.1 / 60.0) -> 'TimeActivityCurve':
        """
        Produces a TAC right-shifted by a specified amount of time.

        The right shift moves activity values later in time and fills the leading
        values with interpolated data. The shift can either be applied on the
        original time points or on a time grid determined by `dt`.

        Args:
            tac (TimeActivityCurve): The original TAC to be shifted.
            shift_in_mins (float, optional):
                The amount (in minutes) to right-shift the TAC. Must be larger than 0.
                Defaults to 10.0 / 60.0 (10 seconds).
            dt (float | None, optional):
                The time interval for oversampling. If `None`, the TAC is resampled at its original time points.
                If a value is provided, the TAC is resampled at an evenly spaced interval of `dt`.

        Returns:
            TimeActivityCurve: A new `TimeActivityCurve` instance with the right-shifted
            activity values.

        Raises:
            AssertionError: If `shift_in_mins` is not greater than 0.

        Example:
            .. code-block:: python

                # Perform a right shift on the TAC
                shifted_tac = TimeActivityCurve.right_shifted_tac(tac=my_tac, shift_in_mins=5.0 / 60.0)

                # Perform a right shift with oversampling
                shifted_tac = TimeActivityCurve.right_shifted_tac(tac=my_tac, shift_in_mins=2.0 / 60.0, dt=0.05 / 60.0)
        """
        assert shift_in_mins > 0, "shift_in_mins must be larger than 0."
        if dt is None:
            even_tac = tac.evenly_resampled_tac()
        else:
            even_tac = tac.evenly_resampled_tac_given_dt(dt=dt)
        delta_t = even_tac.times[1] - even_tac.times[0] if dt is None else dt

        shift_ind = int(shift_in_mins / delta_t)
        shifted_vals = np.zeros_like(even_tac.activity)
        shifted_vals[shift_ind:] = even_tac.activity[:-shift_ind]
        shifted_vals[:shift_ind] = scipy_interpolate(even_tac.times[shift_ind:],
                                                     shifted_vals[shift_ind:],
                                                     kind='linear',
                                                     fill_value='extrapolate')(even_tac.times[:shift_ind])
        shifted_tac = TimeActivityCurve(even_tac.times, shifted_vals)
        shifted_tac.set_activity_non_negative()
        if dt is None:
            shifted_vals_on_tac_times = scipy_interpolate(*shifted_tac.tac,
                                                          kind='linear',
                                                          fill_value='extrapolate')(tac.times)
            return TimeActivityCurve(tac.times, shifted_vals_on_tac_times)
        else:
            return shifted_tac

    @staticmethod
    def tac_dispersion(tac: 'TimeActivityCurve',
                       disp_func: Callable[[np.ndarray, ...], np.ndarray],
                       disp_kwargs: dict,
                       num_samples: int = 4096):
        r"""
        Applies a dispersion function to a time-activity curve (TAC) and returns the convolved TAC.

        This method evaluates the specified dispersion function `disp_func` at supersampled time points.
        It performs convolution (using :func:`scipy.signal.convolve`)of the supersampled TAC with
        the dispersion function, and the result is sampled back at the original TAC time points
        to form the new convolved TAC.

        .. note::
            We perform the supersampling to ensure that the TACs are sampled evenly before performing
            the convolution. Convolving non-evenly sampled arrays produces nonsense values.

        Args:
            tac (TimeActivityCurve): The original time-activity curve to be convolved.
            disp_func (Callable[[np.ndarray, ...], np.ndarray]):
                The dispersion function to be applied. This function must accept an array of
                times as its first argument, followed by any additional arguments specified
                in `disp_kwargs`.
            disp_kwargs (dict): Additional keyword arguments to pass to `disp_func`.
            num_samples (int, optional):
                The number of evenly spaced samples for supersampling the TAC before convolution.
                Defaults to 4096.

        Returns:
            TimeActivityCurve: A new `TimeActivityCurve` instance with the convolved activity values,
            resampled at the original TAC time points.

        Example:
            .. code-block:: python

                from petpal.utils.time_activity_curve import TimeActivityCurve
                import numpy as np

                # Define a sample dispersion function
                def monoexponential_kernel(t:np.ndarray, tau:float) -> np.ndarray:
                    return (1.0/tau) * np.exp(-t / tau)

                # Original TAC
                my_tac = TimeActivityCurve(
                    times=np.array([0, 10, 20, 30]),
                    activity=np.array([1.0, 2.0, 3.0, 4.0])
                )

                # Apply dispersion
                convolved_tac = TimeActivityCurve.tac_dispersion(
                    tac=my_tac,
                    disp_func=monoexponential_kernel,
                    disp_kwargs={'tau': 5},
                    num_samples=1024
                )

                print(convolved_tac.times)     # Output: Original TAC time points
                print(convolved_tac.activity)  # Output: Activity after applying dispersion

        See Also:
            * :meth:`evenly_resample_tac`

        """

        even_tac = tac.evenly_resampled_tac(num_samples=num_samples)
        dt = even_tac.times[1] - even_tac.times[0]

        disp_func_vals = disp_func(even_tac.times, **disp_kwargs)
        disp_vals_fine = scipy_convolve(in1=even_tac.activity, in2=disp_func_vals, mode='full')
        disp_vals_fine = disp_vals_fine[:num_samples] * dt

        intp_func = scipy_interpolate(x=even_tac.times,
                                      y=disp_vals_fine,
                                      kind='linear',
                                      fill_value='extrapolate')
        disp_vals = intp_func(tac.times)
        disp_tac = TimeActivityCurve(tac.times, disp_vals)

        return disp_tac.set_activity_non_negative()

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



