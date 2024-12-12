"""
This module provides classes for time-activity curve (TAC) interpolation for Positron Emission Tomography (PET)
data. It enables the resampling of data evenly with respect to time, which is particularly useful when performing
convolutions with respect to time. We use :py:class:`scipy.interpolate.interp1d` for the interpolation.

The module comprises two classes: :class:`EvenlyInterpolate` and :class:`EvenlyInterpolateWithMax`.

The :class:`EvenlyInterpolate` class takes in TAC times and values and a specified delta time, :math:`\\Delta t`, to
resample data by interpolating TACs evenly with respect to time.

The :class:`EvenlyInterpolateWithMax` class extends the functionality of :class:`EvenlyInterpolate` by modifying the
calculation of delta time, :math:`\\Delta t`, to explicitly sample the maximum value of the TAC.

Example:
    In the proceeding example, we shall use the two classes to resample a fake TAC. For better visualization, we shall
    plot the resampled TACs shifted in the y-direction.
        
        .. plot::
            :include-source:
            :caption: Evenly interpolating TACs can sometimes miss the maximum value of the provided TAC given the
                time-step. Therefore, we can explicitly include the maximum value by modifying the time-step.
            
            import numpy as np
            from petpal.kinetic_modeling.tac_interpolation import EvenlyInterpolate, EvenlyInterpolateWithMax
            import matplotlib.pyplot as plt
            
            # define some dummy TAC times and values
            tac_times_in_minutes = np.array([0., 1., 2.5, 4.1, 7., 15.0])
            tac_values = np.array([0., 0.8, 2., 1.5, 0.6, 0.0])
        
            # instantiate EvenlyInterpolate object and resample TAC (and add shift for better visualization)
            even_interp = EvenlyInterpolate(tac_times=tac_times_in_minutes, tac_values=tac_values+0.25, delta_time=1.0)
            resampled_tac = even_interp.get_resampled_tac()
        
            # instantiate EvenlyInterpolateWithMax object and resample TAC (and add shift for better visualization)
            even_interp_max = EvenlyInterpolateWithMax(tac_times=tac_times_in_minutes, tac_values=tac_values+0.5, samples_before_max=3)
            resampled_tac_max = even_interp_max.get_resampled_tac()
            
            # plot the TAC and the resampled TACs
            fig, ax = plt.subplots(1,1, constrained_layout=True, figsize=(8,4))
            plt.plot(tac_times_in_minutes, tac_values, 'ko--', label='Raw TAC', zorder=2)
            plt.plot(*resampled_tac, 'ro-', label='Evenly Resampled TAC', zorder=1)
            plt.plot(*resampled_tac_max, 'bo-', label='Evenly Resampled TAC w/ Max', zorder=0)
            ax.text(s='resampled TACS are \\nshifted for visual clarity',
                    x=0.95, y=0.95, ha='right', va='top', transform=ax.transAxes, fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('TAC Value', fontsize=16)
            plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left')
            plt.show()
        

Note:
    This module utilises :py:class:`scipy.interpolate.interp1d` for linear interpolation. Ensure Scipy is installed for
    this package to function.

TODO:
    * Test if the classes work correctly if the max value of the TAC is at the first or last time-point.

"""

import numpy as np
from scipy.interpolate import interp1d as sp_interpolation


class EvenlyInterpolate:
    """A class for basic evenly interpolating TACs with respect to time
    
    When performing convolutions with respect to time, care needs to be taken to account for the time-step between
    samples. One way to circumvent this problem is to resample data evenly with respect to the independent variable,
    or time.

    Uses :py:class:`scipy.interpolate.interp1d` to perform linear interpolation.

    Attributes:
        interp_func (scipy.interpolate.interp1d): Interpolation function given the provided TAC.
        resample_times (np.ndarray): Array containing evenly spaced TAC times.
        resample_vals (np.ndarray): Interpolated activities at the calculated resample times.
    """
    def __init__(self, tac_times: np.ndarray, tac_values: np.ndarray, delta_time: float) -> None:
        r"""Initializes an instance of the EvenlyInterpolate class for TAC interpolation.

        The constructor takes the Time-Activity Curve (TAC) times and values as inputs, along with a delta time
        value. It utilizes the SciPy function `scipy.interpolate.interp1d` to perform a linear interpolation of the
        provided TAC.

        After initializing, it generates an interpolation function based on the TAC times and values. It uses this function
        to calculate interpolated activities for a new array of evenly spaced times. These times start at the
        start time of the original TAC and end at the end time, with steps of delta time.


        Args:
            tac_times (np.ndarray): The time-points at which the original TAC activities were sampled.
                It should be a 1D numpy array with increasing float values.
            tac_values (np.ndarray): The corresponding activity values of the provided TAC sampled at tac_times.
                It should be a 1D numpy array of float values.
            delta_time (float): The time-step to use for the creation of evenly spaced resample times. It should be
                a positive float value.

        """
        self._tac_times = tac_times
        self._tac_values = tac_values
        self.interp_func = sp_interpolation(x=self._tac_times, y=self._tac_values, kind='linear', bounds_error=True)
        self.resample_times = np.arange(tac_times[0], tac_times[-1] + delta_time, delta_time)
        self.resample_vals = self.interp_func(self.resample_times)

    def get_resampled_tac(self) -> np.ndarray:
        """
        Returns the resampled times and values of the Time-Activity Curve (TAC).

        The function combines the resampled times and values into a single numpy array.

        Returns:
            (np.ndarray): An array containing two numpy arrays. The first array corresponds to the
            resampled times and the second array corresponds to the resampled activity values of the TAC.
            
        """
        return np.asarray([self.resample_times, self.resample_vals])
    
    
class EvenlyInterpolateWithMax(EvenlyInterpolate):
    r"""A class, extends :class:`EvenlyInterpolate`, and modifies the :math:`\Delta t` calculation such that the
    maximum value of the provided input TAC is explicitly sampled.
    
    Attributes:
        interp_func (scipy.interpolate.interp1d): Interpolation function given the provided TAC.
        resample_times (np.ndarray): Array containing evenly spaced TAC times.
        resample_vals (np.ndarray): Interpolated activities at the calculated resample times.
        dt (float): The :math:`\Delta t` for the resampled times such that the maximum value is explicitly sampled.
        
    See Also:
        :class:`EvenlyInterpolate`
        
    """
    def __init__(self, tac_times: np.ndarray, tac_values: np.ndarray, samples_before_max: float = 10.0):
        r"""Initializes an instance of the EvenlyInterpolateWithMax class for TAC interpolation.

        This subclass of the :class:EvenlyInterpolate class strives to ensure the maximum value of the Time Activity
        Curve (TAC) is included in the resampled TAC. It achieves this by determining a dynamic delta time value based
        on the input TAC and the user-specified number of samples before the TAC's max value.

        The constructor takes the TAC times and values as input, alongside a parameter that defines the number of
        samples to take before the max value of the TAC. It computes the delta time using a static method of the class,
        feeding that along with the original times and values into the parent class initializer to perform the actual
        interpolation.

        Args:
            tac_times (np.ndarray): The time-points at which the original TAC activities were sampled.
                It should be a 1D numpy array with increasing float values.
            tac_values (np.ndarray): The corresponding activity values of the provided TAC sampled at tac_times.
                It should be a 1D numpy array of float values.
            samples_before_max (float): The number of samples desired before the maximum TAC value is reached.
                It defaults to 10 if not specified and should be a positive float number.

        """
        self.dt = self.calculate_dt_for_even_spacing_with_max_sampled(tac_times, tac_values, samples_before_max)
        super().__init__(tac_times, tac_values, self.dt)
    
    @staticmethod
    def calculate_dt_for_even_spacing_with_max_sampled(tac_times: np.ndarray,
                                                       tac_values: np.ndarray,
                                                       samples_before_max: float) -> float:
        r"""Calculates the time-step or :math:`\Delta t` for evenly sampling the Time-Activity Curve (TAC)
        while preserving the maximum value of the TAC in the sample.

        This method determines the time-step so that when the TAC gets resampled,
        it is guaranteed that the maximum value of the TAC in the input is also sampled in the resample. The
        :math:`\Delta t` is calculated based on the equation:
        
        .. math::
        
            \Delta t = \frac{t_{\text{max}} - t_0}{N}
        
        where :math:`t_{\text{max}}` is the time at which the max value of the TAC is reached,
        :math:`t_0` is the start time, and :math:`N` is the number of samples desired before the max TAC value.

        Args:
            tac_times (np.ndarray): A 1D numpy array of float values, representing times at which TAC activities are
                sampled.
            tac_values (np.ndarray): A 1D numpy array of float values, indicating the activities of the given TAC at
                corresponding times.
            samples_before_max (float): The number of samples that you want before the maximum TAC value is reached.

        Returns:
            (float): The calculated time-step, :math:`\Delta t`, that ensures that the TAC is evenly sampled, and the
            maximum TAC value is included in the samples.
        """
        t_start = tac_times[0]
        t_for_max_val = tac_times[np.argmax(tac_values)]
        dt = (t_for_max_val - t_start) / samples_before_max
        return dt
