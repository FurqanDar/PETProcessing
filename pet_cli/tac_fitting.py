"""
This module provides functionalities for fitting Tissue Compartment Models (TCM) to Time Activity Curves (TAC)
using various methods.

It includes classes that handle different parts of the TAC fitting process:
    - :class:`TACFitter`: The primary class for fitting TCMs to TACs. It provides utility methods to prepare data,
      set up fitting parameters, and perform the curve fitting. This class allows fitting based on various TCM functions
      such as one-tissue compartment model (1TCM), 2TCM, and others.
    - :class:`TACFitterWithoutBloodVolume`: A subclass of TACFitter designed for scenarios when there is no signal
      contribution from blood volume in the TAC. It utilises the functionalities of :class:`TACFitter` and modifies
      certain methods to exclude the blood volume parameter.

Functions and methods in this module use :mod:`numpy` and :mod:`scipy` packages for data manipulation and optimization
of the fitting process.

Please refer to the documentation of each class for more detailed information.

See Also:
    * :mod:`pet_cli.tcms_as_convolutions`
    * :mod:`pet_cli.blood_input`
    
"""
import inspect
import json
from typing import Callable, Union
import numpy as np
from scipy.optimize import curve_fit as sp_cv_fit
from . import tcms_as_convolutions as pet_tcms
from . import blood_input as pet_bld
import os


def _get_fitting_params_for_tcm_func(f: Callable) -> list:
    r"""
    Fetches the parameter names from the function signature of a given Tissue Compartment Model (TCM) function. The
    functions can be one of the following:
    
    * :func:`generate_tac_1tcm_c1_from_tac<pet_cli.tcms_as_convolutions.generate_tac_1tcm_c1_from_tac>`
    * :func:`generate_tac_2tcm_with_k4zero_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_2tcm_with_k4zero_cpet_from_tac>`
    * :func:`generate_tac_serial_2tcm_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_serial_2tcm_cpet_from_tac>`

    Args:
        f (Callable): TCM function.

    Returns:
        list: List of parameter names.
        
    """
    return list(inspect.signature(f).parameters.keys())[2:]


def _get_number_of_fit_params_for_tcm_func(f: Callable) -> int:
    r"""
    Counts the number of fitting parameters for a given Tissue Compartment Model (TCM) function. The
    functions can be one of the following:
    
    * :func:`generate_tac_1tcm_c1_from_tac<pet_cli.tcms_as_convolutions.generate_tac_1tcm_c1_from_tac>`
    * :func:`generate_tac_2tcm_with_k4zero_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_2tcm_with_k4zero_cpet_from_tac>`
    * :func:`generate_tac_serial_2tcm_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_serial_2tcm_cpet_from_tac>`

    Args:
        f (Callable): TCM function.

    Returns:
        int: Number of fitting parameters.
    """
    return len(_get_fitting_params_for_tcm_func(f))


class TACFitter(object):
    r"""
    A class used for fitting Tissue Compartment Models(TCM) to Time Activity Curves (TAC).

    It facilitates and simplifies the curve fitting process of TCM functions to TAC data. The class
    takes in raw TAC data for the plasma and tissue as input, and provides numerous utility methods
    to prepare data, set up fitting parameters, and perform the curve fitting. The resample method ensures
    data is appropriate for curve fitting by interpolating the TAC data over a regular time grid, and includes a
    time=0 data-point to the TACs if necessary.

    The class provides multiple options for setting up weights for the curve fitting residuals and for providing
    initial guesses and setting up bounds for the fitting parameters of the TCM function.

    Allows fitting on the basis of various TCM functions like one-tissue compartment model (1TCM), 2TCM, and others.

    Attributes:
        resample_times (np.ndarray): Times at which TACs are resampled.
        resampled_t_tac (np.ndarray): Tissue TAC values resampled at these times.
        p_tac_vals (np.ndarray): Plasma TAC values used for feeding to TCM function.
        raw_t_tac (np.ndarray): Raw TAC times for tissue, fed at initialization.
        weights (np.ndarray): Weights for handling residuals during the optimization process.
        tgt_tac_vals (np.ndarray): Tissue TAC values to fit TCM model.
        fit_param_number (int): Number of fitting parameters in the TCM function.
        initial_guesses (np.ndarray): Initial guesses for all the parameters for curve fitting.
        bounds_hi (np.ndarray): Upper bounds for all the parameters for curve fitting.
        fit_results (np.optimize.OptimizeResult): The results of the fit, including optimized parameters and covariance
            matrix.
        fit_param_names (List[str]): Names of fitting parameters in the TCM function.
        raw_p_tac (np.ndarray): Raw TAC times for plasma, fed at initialization.
        resampled_p_tac (np.ndarray): Plasma TAC values resampled on these times.
        sanitized_t_tac (np.ndarray): Sanitized version of tissue TAC times.
        bounds_lo (np.ndarray): Lower bounds for all the parameters for curve fitting.
        bounds (np.ndarray): Bounds for each parameter for curve fitting.
        max_func_evals (int): Maximum number of function evaluations (iterations) for the optimization process.
        tcm_func (Callable): The tissue compartment model (TCM) function to fit.
        sanitized_p_tac (np.ndarray): Sanitized version of plasma TAC times.
        delta_t (float): Delta between the newly created time steps in resampled times.
        
    Example:
        In the following quick example, ``tTAC`` represents a tissue TAC (``[times, values]``) and ``pTAC`` represents the
        input function (``[times, values]``). Furthermore, we want to fit the provided ``tTAC`` with a 2TCM.
        
        .. code-block:: python
        
            import pet_cli.tcms_as_convolutions as pet_tcm
            import pet_cli.tac_fitting as pet_fit
            import numpy as np
            
            tcm_func = pet_tcm.generate_tac_serial_2tcm_cpet_from_tac
            fit = pet_fit.TACFitter(pTAC=pTAC, tTAC=tTAC, tcm_func=tcm_func, resample_num=512)
            fit.run_fit()
            fit_params = fit.fit_results[0]
            print(fit_params.round(3))
    
        In the following example, we use an FDG input function from the module-provided data, and simulate a noisy 1TCM
        TAC and fit it -- showing a plot of everything at the end.
    
        .. plot::
            :include-source:
            :caption: Fitting a noisy simulated 1TCM TAC.
            
            import numpy as np
            import pet_cli.tcms_as_convolutions as pet_tcm
            import pet_cli.tac_fitting as pet_fit
            import matplotlib.pyplot as plt
            import pet_cli.testing_utils as pet_tst
            
            tcm_func = pet_tcm.generate_tac_1tcm_c1_from_tac
            pTAC = np.asarray(np.loadtxt("../data/tcm_tacs/fdg_plasma_clamp_evenly_resampled.txt").T)
            tTAC = tcm_func(*pTAC, k1=1.0, k2=0.25, vb=0.05)
            tTAC[1] = pet_tst.add_gaussian_noise_to_tac_based_on_max(tTAC[1])
            
            fitter = pet_fit.TACFitter(pTAC=pTAC, tTAC=tTAC, tcm_func=tcm_func)
            fitter.run_fit()
            fit_params = fitter.fit_results[0]
            fit_tac = pet_tcm.generate_tac_1tcm_c1_from_tac(*pTAC, *fit_params)
            
            plotter = pet_tst.TACPlots()
            plotter.add_tac(*pTAC, label='Input TAC', pl_kwargs={'color':'black', 'ls':'--'})
            plotter.add_tac(*tTAC, label='Tissue TAC', pl_kwargs={'color':'blue', 'ls':'', 'marker':'o', 'mec':'k'})
            plotter.add_tac(*fit_tac, label='Fit TAC', pl_kwargs={'color':'red', 'ls':'-', 'marker':'', 'lw':2.5})
            plt.legend()
            plt.show()
    
    See Also:
        * :class:`TACFitterWithoutBloodVolume` to assume :math:`V_B=0` and only fit the kinetic parameters.
        
    """
    def __init__(self,
                 pTAC: np.ndarray,
                 tTAC: np.ndarray,
                 weights: Union[None, float, np.ndarray] = None,
                 tcm_func: Callable = None,
                 fit_bounds: Union[np.ndarray, None] = None,
                 resample_num: int = 512,
                 aif_fit_thresh_in_mins: float = 30.0,
                 max_iters: int = 2500):
        r"""
        Initialize TACFitter with provided arguments.

        The init function performs several important operations:
            1. It sets the maximum number of function evaluations (iterations) for the optimization process.
            2. It sets the TCM function properties and initial bounds with the provided TCM function and fit bounds.
            3. It loads the raw tissue and plasma TACs and then resamples them evenly over a regular time grid.
            4. It determines the weights to be used for handling residuals during the optimization process.
            5. It sets the plasma TAC values and tissue TAC values to fit the TCM model.

        Args:
            pTAC (np.ndarray): The plasma TAC, with the form ``[times, values]``.
            tTAC (np.ndarray): The tissue TAC to which we will fit a TCM, with the form ``[times, values]``.
            weights (float, np.ndarray or None, optional): Weights for handling residuals during the optimization
                process. If None, all residuals are equally weighted. Defaults to None.
            tcm_func (Callable, optional): The specific TCM function to be used for fitting. Defaults to None.
            fit_bounds (np.ndarray or None, optional): Bounds for each parameter for curve fitting.
                If None, they will be guessed. Defaults to None.
            resample_num (int, optional): The number of time points used when resampling TAC data. Defaults to 512.
            aif_fit_thresh_in_mins (float, optional): The threshold in minutes when resampling. Defaults to 30.0.
            max_iters (int, optional): Maximum number of function evaluations (iterations) for the optimization process.
                Defaults to 2500.
                
        """
        
        self.max_func_evals: int = max_iters
        self.tcm_func: Callable = None
        self.fit_param_number: int = None
        self.fit_param_names: list = None
        
        self.bounds: np.ndarray = None
        self.initial_guesses: np.ndarray = None
        self.bounds_lo: np.ndarray = None
        self.bounds_hi: np.ndarray = None
        
        self.get_tcm_func_properties(tcm_func)
        self.set_bounds_and_initial_guesses(fit_bounds)
        
        self.raw_p_tac: np.ndarray = pTAC.copy()
        self.raw_t_tac: np.ndarray = tTAC.copy()
        self.sanitized_t_tac: np.ndarray = None
        self.sanitized_p_tac: np.ndarray = None
        self.resample_times: np.ndarray = None
        self.delta_t: float = None
        self.resampled_t_tac: np.ndarray = None
        self.resampled_p_tac: np.ndarray = None
        
        self.resample_tacs_evenly(aif_fit_thresh_in_mins, resample_num)
        
        self.weights: np.ndarray = None
        self.set_weights(weights)
        
        self.p_tac_vals: np.ndarray = self.resampled_p_tac[1]
        self.tgt_tac_vals: np.ndarray = self.resampled_t_tac[1]
        self.fit_results = None
    
    def set_bounds_and_initial_guesses(self, fit_bounds: np.ndarray) -> None:
        r"""
        Sets initial guesses for the fitting parameters, along with their lower and upper bounds.

        The function checks if a custom ``fit_bounds`` is provided. If yes, it makes
        sure that its shape is valid (that is, for each fitting parameter it requires
        the tuple: ``(initial, lower, upper)``) and then sets it. But if no custom
        `fit_bounds` is given, it first initializes bounds to all zeros and then sets
        each parameter's bounds to ``(0.1, 1.0e-8, 5.0)`` except for the last parameter,
        which it sets to ``(0.1, 0.0, 1.0)`` because it corresponds to the fraction of blood
        and is physically constrained between 0 and 1. The function separately stores the initial points,
        lower and upper bounds in three different numpy arrays for later use.

        Args:
            fit_bounds (numpy.ndarray): A 2D numpy array containing initial parameter guesses, and their lower and
                upper bounds in the form of ``(initial, lower, upper)``. The shape should be
                (``number_of_fit_params``, 3).

        Raises:
            AssertionError: If `fit_bounds` doesn't have a valid shape.

        Side Effects:
            - bounds (np.ndarray): Either takes custom defined ``fit_bounds``, or sets the default bounds
                                   for each parameter with the last parameter having bounds defined
                                   between 0 and 1.
            - initial_guesses (np.ndarray): Initial guesses for all the parameters.
            - bounds_lo (np.ndarray): Lower bounds for all the parameters.
            - bounds_hi (np.ndarray): Upper bounds for all the parameters.
        
        """
        assert self.tcm_func is not None, "This method should be run after `get_tcm_func_properties`"
        if fit_bounds is not None:
            assert fit_bounds.shape == (self.fit_param_number, 3), ("Fit bounds has the wrong shape. For each potential"
                                                                    " fitting parameter in `tcm_func`, we require the "
                                                                    "tuple: `(initial, lower, upper)`.")
            self.bounds = fit_bounds.copy()
        else:
            bounds = np.zeros((self.fit_param_number, 3), float)
            for pid, param in enumerate(bounds[:-1]):
                bounds[pid] = [0.1, 1.0e-8, 5.0]
            bounds[-1] = [0.1, 0.0, 1.0]
            self.bounds = bounds.copy()
        
        self.initial_guesses = self.bounds[:, 0]
        self.bounds_lo = self.bounds[:, 1]
        self.bounds_hi = self.bounds[:, 2]
    
    def resample_tacs_evenly(self, fit_thresh_in_mins: float, resample_num: int) -> None:
        r"""
        Resample pTAC and tTAC evenly with respect to time, and at the same times.

        The method takes a threshold in minutes and a resample number as inputs. It starts by sanitizing
        the pTAC and tTAC (prepending a :math:`f(t=0)=0` point to data if necessary). A regularly sampled time is
        then generated using the start, end, and number of samples dictated by resample_num. Following this,
        an interpolation object is created using the :class:`pet_cli.blood_input.BloodInputFunction` class for the pTAC.
        This allows both interpolation and extrapolation for times beyond the pTAC onto the new tTAC times.

        Finally, the method resamples the sanitized tTAC and pTAC across these new evenly distributed
        times to ensure that they are regularly spaced over time. These resampled values are stored for
        future computations. The :math:`\Delta t` for the regularly sampled times is also stored.

        Args:
            fit_thresh_in_mins (float): Threshold in minutes used for defining how to fit half of the pTAC.
                                        The fitting time threshold determines the point at which the pTAC
                                        switches from interpolation to fitting. It should be a positive float value.

            resample_num (int): Number of samples to generate when resampling the tTAC. This will be the total
                                number of samples in tTAC after it has been resampled. It should be a positive integer.

        Returns:
            None

        Side Effects:
            - sanitized_t_tac (np.ndarray): Sanitized version of the original tTAC given during class initialization.
            - sanitized_p_tac (np.ndarray): Sanitized version of the original pTAC given during class initialization.
            - resample_times (np.ndarray): Regularly sampled time points generated from the start and end of sanitized
              tTAC, and the passed resample_num.
            - delta_t (float): Delta between the newly created time steps in resample_times.
            - resampled_t_tac (np.ndarray): tTAC resampled at the time points defined in resample_times.
            - resampled_p_tac (np.ndarray): pTAC resampled and extrapolated (if necessary) at the time points defined in
              resample_times.
              
        See Also:
            - :class:`pet_cli.blood_input.BloodInputFunction`
            
        """
        self.sanitized_t_tac = self.sanitize_tac(*self.raw_t_tac)
        self.sanitized_p_tac = self.sanitize_tac(*self.raw_p_tac)
        
        self.resample_times = np.linspace(self.sanitized_t_tac[0][0], self.sanitized_t_tac[0][-1], resample_num)
        self.delta_t = self.resample_times[1] - self.resample_times[0]
        
        p_tac_interp_obj = pet_bld.BloodInputFunction(time=self.sanitized_p_tac[0], activity=self.sanitized_p_tac[1],
                                                      thresh_in_mins=fit_thresh_in_mins)
        
        self.resampled_t_tac = self.resample_tac_on_new_times(*self.sanitized_t_tac, self.resample_times)
        self.resampled_p_tac = np.asarray(
                [self.resample_times[:], p_tac_interp_obj.calc_blood_input_function(t=self.resample_times)])
    
    def set_weights(self, weights: Union[float, str, None]) -> None:
        r"""
        Sets the weights for handling the residuals in the optimization process.

        The ``weights`` parameter determines how weights will be used:
            - It can be a float which will generate the weights based on an exponential decay formula. We assume that
              the passed in float is the decay constant, :math:`\lambda=\ln(2)/T_{1/2}`, where the half-life is in
              minutes. The weights are generated as: :math:`\sigma_i=\sqrt{e^{-\lambda t_i}C(t_i)}`, to be used as the
              ``sigma`` parameter for :func:`scipy.optimize.curve_fit`.
            - If it's a numpy array, the weights are linearly interpolated on the calculated `resample_times`.
            - If no specific value or an array is given, a numpy array of ones is used (i.e., it assumes equal weight).

        The method asserts that ``resampled_t_tac`` has been computed, thus :meth:`resample_tacs_evenly`
        method should be run before this.

        Args:
            weights (Union[float, str, None]): Determines how weights will be computed. If a float, it is used
                                               as the exponential decay constant. If a numpy array, the provided weights
                                               are linearly interpolated on the calculated resampled times. If None,
                                               equal weights are assumed.

        Returns:
            None

        Side Effects:
            weights (np.ndarray): Sets the weights attribute of the class based on logical conditions. Either
                                  they are based on an exponential decay function, directly supplied, or assumed
                                  as equal weights.
                                  
        """
        assert self.resampled_t_tac is not None, 'This method should be run after `resample_tacs_evenly`'
        
        if isinstance(weights, float):
            tmp_ar = np.sqrt(np.exp(-weights * self.resampled_t_tac[0]) * self.resampled_t_tac[1])
            zero_idx = tmp_ar == 0.0
            tmp_ar[zero_idx] = np.inf
            self.weights = tmp_ar
        elif isinstance(weights, np.ndarray):
            self.weights = np.interp(x=self.resampled_t_tac[0], xp=self.raw_t_tac[0], fp=weights)
        else:
            self.weights = np.ones_like(self.resampled_t_tac[1])
    
    def get_tcm_func_properties(self, tcm_func: Callable) -> None:
        r"""
        Analyzes the provided tissue compartment model (TCM) function, sets it for the current instance, and extracts
        related property information. The ``tcm_func`` should be one of the following:
        
        * :func:`generate_tac_1tcm_c1_from_tac<pet_cli.tcms_as_convolutions.generate_tac_1tcm_c1_from_tac>`
        * :func:`generate_tac_2tcm_with_k4zero_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_2tcm_with_k4zero_cpet_from_tac>`
        * :func:`generate_tac_serial_2tcm_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_serial_2tcm_cpet_from_tac>`

        The function extracts fitting parameter names and their count from the function signature and sets them in the
        current instance for later usage.

        Args:
            tcm_func (Callable): A function that generates a TAC using a specific compartment model.

        Raises:
            AssertionError: If ``tcm_func`` is not one of the allowed TCM functions.
        """
        assert tcm_func in [pet_tcms.generate_tac_1tcm_c1_from_tac,
                            pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac,
                            pet_tcms.generate_tac_serial_2tcm_cpet_from_tac], (
            "`tcm_func should be one of `pet_tcms.generate_tac_1tcm_c1_from_tac`, "
            "`pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac`, "
            "`pet_tcms.generate_tac_serial_2tcm_cpet_from_tac`")
        
        self.tcm_func = tcm_func
        self.fit_param_names = _get_fitting_params_for_tcm_func(self.tcm_func)
        self.fit_param_number = len(self.fit_param_names)
    
    @staticmethod
    def sanitize_tac(tac_times: np.ndarray, tac_vals: np.ndarray) -> np.ndarray:
        r"""
        Makes sure that the Time-Activity Curve (TAC) starts from time zero.

        The method ensures that the TAC starts from time zero by checking the first timestamp. If it's not zero, a zero
        timestamp and value are prepended, otherwise, the first value is set to zero. This method assumes that
        `tac_times` and `tac_vals` arrays have the same shape.

        Args:
            tac_times (numpy.ndarray): The original times of the TAC.
            tac_vals (numpy.ndarray): The original values of the TAC.

        Returns:
            numpy.ndarray: The sanitized TAC: ``[sanitized_times, sanitized_vals]``.
        """
        assert tac_times.shape == tac_vals.shape, "`tac_times` and `tac_vals` must have the same shape."
        if tac_times[0] != 0.0:
            return np.asarray([np.append(0, tac_times), np.append(0, tac_vals)])
        else:
            out_vals = tac_vals[:]
            out_vals[0] = 0.0
            return np.asarray([tac_times, out_vals])
    
    @staticmethod
    def resample_tac_on_new_times(tac_times: np.ndarray, tac_vals: np.ndarray, new_times: np.ndarray) -> np.ndarray:
        r"""
        Resamples the Time-Activity Curve (TAC) on given new time points by linear interpolation.

        The method performs a linear interpolation of `tac_vals` on `new_times` based on `tac_times`.

        Args:
            tac_times (numpy.ndarray): The original times of the TAC.
            tac_vals (numpy.ndarray): The original values of the TAC.
            new_times (numpy.ndarray): The new times to resample the TAC on.

        Returns:
            numpy.ndarray: The resampled TAC: the resampled times and values. ``[new_times, new_vals]``.
            
        See Also:
            :func:`numpy.interp`
            
        """
        return np.asarray([new_times, np.interp(x=new_times, xp=tac_times, fp=tac_vals)])
    
    def fitting_func(self, x: np.ndarray, *params) -> np.ndarray:
        r"""
        A wrapper function to fit the Tissue Compartment Model (TCM) using given parameters.

        It calculates the results of the TCM function with the given times and parameters using the resampled pTAC.

        Args:
            x (np.ndarray): The independent data (time-points for TAC)
            *params: The parameters for the TCM function

        Returns:
            np.ndarray: The values of the TCM function with the given parameters at the given x-values.
        """
        return self.tcm_func(x, self.p_tac_vals, *params)[1]
    
    def run_fit(self) -> None:
        r"""
        Runs the optimization/fitting process on the data, using previously defined function and parameters.

        This method runs the curve fitting process on the TAC data, starting with the initial guesses
        for the parameters and the preset bounds for each. ``fitting_func``, initial guesses and bounds
        should have been set prior to calling this method. Optimized fit results and fit covariances are stored in
        ``fit_results``.

        Returns:
            None

        Side Effects:
            - fit_results (OptimizeResult): The results of the fit, including optimized parameters and covariance matrix.
              Fitted values can be extracted using fit_results.x, among other available attributes (refer to
              :func:`scipy.optimize.curve_fit` documentation for more details).
              
        """
        self.fit_results = sp_cv_fit(f=self.fitting_func, xdata=self.resample_times, ydata=self.tgt_tac_vals,
                                     p0=self.initial_guesses, bounds=(self.bounds_lo, self.bounds_hi),
                                     sigma=self.weights, maxfev=self.max_func_evals)


class TACFitterWithoutBloodVolume(TACFitter):
    r"""
    A sub-class of TACFitter used specifically for fitting Tissue Compartment Models(TCM) to Time Activity Curves (TAC),
    when there is no signal contribution from blood volume in the TAC.

    It uses the functionalities of :class:`TACFitter` and modifies the methods calculating the ``tcm_function``
    properties and the bounds setting, and the wrapped ``fitting_func`` to ignore the blood volume parameter, ``vb``.

    Attributes:
        resample_times (np.ndarray): Times at which TACs are resampled.
        resampled_t_tac (np.ndarray): Tissue TAC values resampled at these times.
        p_tac_vals (np.ndarray): Plasma TAC values used for feeding to TCM function.
        raw_t_tac (np.ndarray): Raw TAC times for tissue, fed at initialization.
        weights (np.ndarray): Weights for handling residuals during the optimization process.
        tgt_tac_vals (np.ndarray): Tissue TAC values to fit TCM model.
        fit_param_number (int): Number of fitting parameters in the TCM function.
        initial_guesses (np.ndarray): Initial guesses for all the parameters for curve fitting.
        bounds_hi (np.ndarray): Upper bounds for all the parameters for curve fitting.
        fit_results (np.optimize.OptimizeResult): The results of the fit, including optimized parameters and covariance
            matrix.
        fit_param_names (List[str]): Names of fitting parameters in the TCM function.
        raw_p_tac (np.ndarray): Raw TAC times for plasma, fed at initialization.
        resampled_p_tac (np.ndarray): Plasma TAC values resampled on these times.
        sanitized_t_tac (np.ndarray): Sanitized version of tissue TAC times.
        bounds_lo (np.ndarray): Lower bounds for all the parameters for curve fitting.
        bounds (np.ndarray): Bounds for each parameter for curve fitting.
        max_func_evals (int): Maximum number of function evaluations (iterations) for the optimization process.
        tcm_func (Callable): The tissue compartment model (TCM) function to fit.
        sanitized_p_tac (np.ndarray): Sanitized version of plasma TAC times.
        delta_t (float): Delta between the newly created time steps in resampled times.
        
    See Also:
        * :class:`TACFitter`

    """
    def __init__(self,
                 pTAC: np.ndarray,
                 tTAC: np.ndarray,
                 weights: np.ndarray = None,
                 tcm_func: Callable = None,
                 fit_bounds: np.ndarray = None,
                 resample_num: int = 2048,
                 aif_fit_thresh_in_mins: float = 30.0,
                 max_iters: int = 2500):
        r"""
        Initializes TACFitterWithoutBloodVolume with provided arguments. Inherits all arguments from parent class TACFitter.

        This ``__init__`` method, in comparison to TACFitter's ``__init__``, executes the same initial operations but
        disregards the blood volume parameter. The significant steps are:
            1. Calls the TACFitter's __init__ with the provided arguments.
            2. Sets the TCM function properties while eliminating blood volume.
            3. Sets the fitting bounds and initial guesses, again excluding blood volume.

        Args:
            pTAC (np.ndarray): The plasma TAC, with the form [times, values].
            tTAC (np.ndarray): The tissue TAC to which we will fit a TCM, with the form [times, values].
            weights (float, np.ndarray or None, optional): Weights for handling residuals during the optimization process.
                If None, all residuals are equally weighted. Defaults to None.
            tcm_func (Callable, optional): The specific TCM function to be used for fitting. Defaults to None.
            fit_bounds (np.ndarray or None, optional): Bounds for each parameter for curve fitting.
                If None, they will be guessed. Defaults to None.
            resample_num (int, optional): The number of time points used when resampling TAC data. Defaults to 512.
            aif_fit_thresh_in_mins (float, optional): The threshold in minutes when resampling. Defaults to 30.0.
            max_iters (int, optional): Maximum number of function evaluations (iterations) for the optimization process.
                Defaults to 2500.

        Side-effect:
            Sets the TCM function properties and initial bounds while disregarding the blood volume parameter.
            
        See Also
            * :class:`TACFitter`
            
        """
        
        super().__init__(pTAC, tTAC, weights, tcm_func, fit_bounds, resample_num, aif_fit_thresh_in_mins, max_iters)
        self.get_tcm_func_properties(tcm_func)
        self.set_bounds_and_initial_guesses(fit_bounds)
    
    def get_tcm_func_properties(self, tcm_func: Callable) -> None:
        r"""
        Overridden method to define a TCM function excluding blood volume.
        
        The ``tcm_func`` should be one of the following:
        
        * :func:`generate_tac_1tcm_c1_from_tac<pet_cli.tcms_as_convolutions.generate_tac_1tcm_c1_from_tac>`
        * :func:`generate_tac_2tcm_with_k4zero_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_2tcm_with_k4zero_cpet_from_tac>`
        * :func:`generate_tac_serial_2tcm_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_serial_2tcm_cpet_from_tac>`

        Args:
            tcm_func: The chosen TCM function model.

        Side-effect:
            Sets ``tcm_func``, ``fit_param_names``, and ``fit_param_number`` attributes.
            
        """
        assert tcm_func in [pet_tcms.generate_tac_1tcm_c1_from_tac,
                            pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac,
                            pet_tcms.generate_tac_serial_2tcm_cpet_from_tac], (
            "`tcm_func should be one of `pet_tcms.generate_tac_1tcm_c1_from_tac`, "
            "`pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac`, "
            "`pet_tcms.generate_tac_serial_2tcm_cpet_from_tac`")
        
        self.tcm_func = tcm_func
        self.fit_param_names = _get_fitting_params_for_tcm_func(self.tcm_func)[:-1]
        self.fit_param_number = len(self.fit_param_names)
    
    def set_bounds_and_initial_guesses(self, fit_bounds: np.ndarray) -> None:
        r"""
        Overridden method to set bounds and initial guesses excluding blood volume parameter.

        Args:
            fit_bounds: The input bounds for fitting parameters.

        Side-effect:
            - Sets ``initial_guesses``, ``bounds_lo``, ``bounds_hi``, and ``bounds``, ignoring the last parameter
              (blood volume).
              
        """
        assert self.tcm_func is not None, "This method should be run after `get_tcm_func_properties`"
        if fit_bounds is not None:
            assert fit_bounds.shape == (self.fit_param_number, 3), ("Fit bounds has the wrong shape. For each potential"
                                                                    " fitting parameter in `tcm_func`, we require the "
                                                                    "tuple: `(initial, lower, upper)`.")
            self.bounds = fit_bounds.copy()
        else:
            bounds = np.zeros((self.fit_param_number, 3), float)
            for pid, param in enumerate(bounds[:]):
                bounds[pid] = [0.1, 1.0e-8, 5.0]
            self.bounds = bounds.copy()
        
        self.initial_guesses = self.bounds[:, 0]
        self.bounds_lo = self.bounds[:, 1]
        self.bounds_hi = self.bounds[:, 2]
    
    def fitting_func(self, x: np.ndarray, *params) -> np.ndarray:
        r"""
        Overridden method to fit the TCM model setting ``vb=0.0`` explicitly.
        
        It calculates the results of the TCM function with the given times and parameters using the resampled pTAC (with
        ``vb=0.0``).

        Args:
            x: The independent data (time-points for TAC).
            *params: Parameters of TCM function (excluding blood volume).

        Returns:
            The values of the TCM function with the given parameters at the given x-values,
            with blood volume (``vb``) set to 0.
        """
        return self.tcm_func(x, self.p_tac_vals, *params, vb=0.0)[1]


# TODO: Use the safe loading of TACs function from an IO module when it is implemented
def _safe_load_tac(filename: str, **kwargs) -> np.ndarray:
    """
    Loads time-activity curves (TAC) from a file.

    Tries to read a TAC from specified file and raises an exception if unable to do so. We assume that the file has two
    columns, the first corresponding to time and second corresponding to activity.

    Args:
        filename (str): The name of the file to be loaded.

    Returns:
        np.ndarray: A numpy array containing the loaded TAC. The first index corresponds to the times, and the second
        corresponds to the activity.

    Raises:
        Exception: An error occurred loading the TAC.
    """
    try:
        return np.array(np.loadtxt(filename).T, dtype=float, order='C', **kwargs)
    except Exception as e:
        print(f"Couldn't read file {filename}. Error: {e}")
        raise e


class FitTCMToTAC(object):
    r"""
    A class dedicated to perform Tissue Compartment Model (TCM) fitting to time-activity curves (TACs).

    This class consolidates all related TAC fitting functionalities. It first initializes relevant fitting parameters
    and paths to TAC data from input arguments. After initialization, an analysis with methods like `run_analysis()`
    and `save_analysis()` can be performed. The results, including fit parameters and properties, can be
    accessed after the analysis.

    Example:
        In the proceeding example, we assume that we have two tacs: an input function tac, and a region of interest
        (ROI) tac named 'input_tac.txt' and 'roi_tac.txt', respectively. Here, we are trying to fit the ROI tac to
        a standard serial 2TCM. Note that we are not fitting any time delay or dispersion corrections.
        
        .. code-block:: python
            
            import pet_cli.tac_fitting as pet_fit
            
            fit_obj = pet_fit.FitTCMToTAC(input_tac_path='./input_tac.txt',
                                          roi_tac_path='./roi_tac.txt',
                                          output_directory='./',
                                          output_filename_prefix='fit',
                                          compartment_model='serial-2tcm',
                                          weights=None,
                                          parameter_bounds=np.asarray([[0.1, 0.0, 1.0],
                                                                       [0.1, 0.0, 1.0],
                                                                       [0.1, 0.0, 1.0],
                                                                       [0.1, 0.0, 1.0],
                                                                       [0.1, 0.0, 1.0]]),
                                          resample_num=512
                              )
            fit_obj.run_analysis()
            fit_obj.save_analysis()
    
    See Also:
        * :class:`TACFitter`
        * :class:`TACFitterWithoutBloodVolume`
    
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tac_path: str,
                 output_directory: str,
                 output_filename_prefix: str,
                 compartment_model: str,
                 parameter_bounds: Union[None, np.ndarray] = None,
                 weights: Union[float, None, np.ndarray] = None,
                 resample_num: int = 512,
                 aif_fit_thresh_in_mins: float = 40.0,
                 max_func_iters: int = 2500,
                 ignore_blood_volume: bool = False):
        r"""
        Initializes an instance of the FitTCMToTAC class.

        The initialization follows these steps:
            1. Saves absolute paths of the TAC files and output directory.
            2. Validates and stores the compartment model and the corresponding fitting function.
            3. Stores the other fitting parameters passed as arguments.
            4. Initializes the appropriate fitting object, depending on whether blood volume is considered or not.
            5. Initializes the analysis properties structure.

        After initialization, you can directly run and save the analysis using bundled methods :meth:`run_analysis` and
        :meth:`save_analysis`, respectively.
        
        See Also:
            * :meth:`validated_tcm`
            
        """
        self.input_tac_path: str = os.path.abspath(input_tac_path)
        self.roi_tac_path: str = os.path.abspath(roi_tac_path)
        self.output_directory: str = os.path.abspath(output_directory)
        self.output_filename_prefix: str = output_filename_prefix
        self.compartment_model: str = self.validated_tcm(compartment_model)
        self._tcm_func: Callable = self._get_tcm_function(self.compartment_model)
        self.bounds: Union[None, np.ndarray] = parameter_bounds
        self.tac_resample_num: int = resample_num
        self.input_tac_fitting_thresh_in_mins: float = aif_fit_thresh_in_mins
        self.max_func_iters: int = max_func_iters
        self.ignore_blood_volume = ignore_blood_volume
        self.weights: Union[float, None, np.ndarray] = weights
        if self.ignore_blood_volume:
            self.fitting_obj = TACFitterWithoutBloodVolume
        else:
            self.fitting_obj = TACFitter
        self.analysis_props: dict = self.init_analysis_props()
        self.fit_results: Union[None, tuple[np.ndarray, np.ndarray]] = None
        self._has_analysis_been_run: bool = False
        
    def init_analysis_props(self):
        r"""
        Initialize the structure of the analysis properties dictionary. This dictionary can be saved as a JSON file for
        parsing the results.

        The dictionary structure is as follows:

            FilePathPTAC -> path of the input TAC file
            FilePathTTAC -> path of the ROI TAC file
            TissueCompartmentModel -> type of tissue compartment model used
            IgnoreBloodVolume -> flag indicating whether blood volume is being considered or not
            PTACFittingThresholdTime -> threshold time for the AIF fitting
            FitProperties -> an inner dictionary with empty lists/arrays as placeholders
                            for FitValues, FitStdErr and Bounds. Also contains ResampleNum and MaxIterations.

        FitProperties will be updated during the analysis process.

        Returns:
            dict: Initialized dictionary with analysis properties structure.
            
        See Also:
            * :meth:`calculate_fit_properties`
            
        """
        props = {
            'FilePathPTAC': self.input_tac_path,
            'FilePathTTAC': self.roi_tac_path,
            'TissueCompartmentModel': self.compartment_model,
            'IgnoreBloodVolume': self.ignore_blood_volume,
            'PTACFittingThresholdTime': self.input_tac_fitting_thresh_in_mins,
            'FitProperties': {
                'FitValues': [],
                'FitStdErr': [],
                'Bounds': [],
                'ResampleNum': self.tac_resample_num,
                'MaxIterations': self.max_func_iters,
                }
            }
        
        return props
    
    @staticmethod
    def validated_tcm(compartment_model: str) -> str:
        r"""
        Validates the type of tissue compartment model.

        This method checks that the provided compartment model is one of the pre-defined options '1tcm', '2tcm-k4zero'
        or 'serial-2tcm'. The input is transformed to lowercase and spaces are replaced with hyphens before checking
        validity.

        Args:
            compartment_model (str): The name of the compartment model.

        Returns:
            str: The transformed name of the validated compartment model if it was valid.

        Raises:
            ValueError: If the provided compartment model is not one of '1tcm', '2tcm-k4zero' or 'serial-2tcm'
            
        """
        tcm = compartment_model.lower().replace(' ', '-')
        if tcm not in ['1tcm', '2tcm-k4zero', 'serial-2tcm']:
            raise ValueError("compartment_model must be one of '1tcm', '2tcm-k4zero', or 'serial-2tcm'")
        return tcm
    
    @staticmethod
    def _get_tcm_function(compartment_model: str) -> Callable:
        """
        Returns the corresponding function for the provided tissue compartment model used for the fitting class

        Args:
            compartment_model: The name of the tissue compartment model.

        Returns:
            function: The corresponding function for the tissue compartment model.

        Raises:
            KeyError: If the provided tissue compartment model name does not correspond to any known models.
        
        See Also:
            * :func:`generate_tac_1tcm_c1_from_tac<pet_cli.tcms_as_convolutions.generate_tac_1tcm_c1_from_tac>`
            * :func:`generate_tac_2tcm_with_k4zero_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_2tcm_with_k4zero_cpet_from_tac>`
            * :func:`generate_tac_serial_2tcm_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_serial_2tcm_cpet_from_tac>`
            * :class:`TACFitter`
            * :class:`TACFitterWithoutBloodVolume`
        
        """
        tcm_funcs = {
                   '1tcm': pet_tcms.generate_tac_1tcm_c1_from_tac,
                   '2tcm-k4zero': pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac,
                   'serial-2tcm': pet_tcms.generate_tac_serial_2tcm_cpet_from_tac
                    }
        
        return tcm_funcs[compartment_model]
    
    def run_analysis(self):
        r"""
        Runs the fitting analysis given the file-paths and method.
        
        :meth:`calculate_fit` and :meth:`calculate_fit_properties` are run, and the analysis-has-been-fun flag is set to
        true.

        This method first calculates the fit, then updates the analysis properties with the
        fit results and finally sets the flag denoting that analysis has been successfully run.
        
        
        """
        self.calculate_fit()
        self.calculate_fit_properties()
        self._has_analysis_been_run = True
    
    def save_analysis(self):
        r"""
        Saves the analysis properties to a json file in the prescribed output directory.

        The saved filename is constructed as a combination of the output directory, filename prefix,
        "analysis", and the used TissueCompartmentModel.
        If the analysis has not been run before calling this method, a RuntimeError is raised.

        Raises:
            RuntimeError: If the method is called before the analysis has been run.
        """
        if not self._has_analysis_been_run:
            raise RuntimeError("'run_analysis' method must be run before running this method.")
        
        file_name_prefix = os.path.join(self.output_directory,
                                        f"{self.output_filename_prefix}_analysis"
                                        f"-{self.analysis_props['TissueCompartmentModel']}")
        analysis_props_file = f"{file_name_prefix}_props.json"
        with open(analysis_props_file, 'w') as f:
            json.dump(obj=self.analysis_props, fp=f, indent=4)
    
    def calculate_fit_properties(self):
        r"""
        Calculates the fit properties and updates the analysis properties.

        This method retrieves the fitting parameters and their standard errors from the fitting
        results, formats them for readability, and stores them in the analysis properties dictionary.
        Bounds to the fitting parameters are also formatted and stored.
        """
        fit_params, fit_covariances = self.fit_results
        fit_stderr = np.sqrt(np.diagonal(fit_covariances))
        format_func = self._generate_pretty_params
        
        self.analysis_props["FitProperties"]["FitValues"] = format_func(fit_params.round(5))
        self.analysis_props["FitProperties"]["FitStdErr"] = format_func(fit_stderr.round(5))
        
        format_func = self._generate_pretty_bounds
        self.analysis_props["FitProperties"]["Bounds"] = format_func(self.bounds.round(5))
    
    def calculate_fit(self):
        r"""
        Performs the fit and stores results to the instance.

        This method has a series of actions which it performs in the following order:
            1. Loads TAC files using the :func:`_safe_load_tac` method, specific to the paths stored in instance
               variables.
            2. Creates a fitting object with the relevant parameters and TACs.
            3. Runs the fit using the fitting object's ``run_fit`` method.
            4. Stores the results of the fit in the ``fit_results`` instance variable.

        As a result of this method, ``fit_results`` instance variable will hold the results of the fit, that
        can be used for further analysis or calculations.
        
        See Also:
            * :class:`TACFitter`
            * :class:`TACFitterWithoutBloodVolume`
        
        """
        p_tac = _safe_load_tac(self.input_tac_path)
        t_tac = _safe_load_tac(self.roi_tac_path)
        self.fitting_obj = self.fitting_obj(pTAC=p_tac, tTAC=t_tac,
                                            weights=self.weights,
                                            tcm_func=self._tcm_func,
                                            fit_bounds=self.bounds,
                                            max_iters=self.max_func_iters,
                                            aif_fit_thresh_in_mins=self.input_tac_fitting_thresh_in_mins,
                                            resample_num=self.tac_resample_num)
        self.fitting_obj.run_fit()
        self.fit_results = self.fitting_obj.fit_results
        if self.bounds is None:
            self.bounds = self.fitting_obj.bounds
    
    def _generate_pretty_params(self, results: np.ndarray) -> dict:
        r"""
        Transforms array of results into a formatted dictionary.

        This method formats the fitting results into a more human-readable form.
        If the fitting was done without blood volume, it formats all values as 'k_i': value.
        Otherwise it formats all but the last as 'k_i': value and the last as 'vb': value.

        Args:
            results (np.ndarray): The array of fitting results.

        Returns:
            dict: The formatted fitting results as {param: value} pairs. In the case of
                  TACFitterWithBloodVolume, the last parameter will be named 'vb', others 'k_i'.
                  In the case of TACFitterWithoutBloodVolume, parameters will be named 'k_i'.
        """
        if isinstance(self.fitting_obj, TACFitterWithoutBloodVolume):
            k_vals = {f'k_{n + 1}': val for n, val in enumerate(results)}
            return k_vals
        else:
            k_vals = {f'k_{n + 1}': val for n, val in enumerate(results[:-1])}
            vb = {f'vb': results[-1]}
            return {**k_vals, **vb}
    
    def _generate_pretty_bounds(self, bounds: Union[np.ndarray, None]) -> dict:
        r"""
        Transforms array of bounds into a formatted dictionary.

        This method creates a dictionary of the fitting parameters and their corresponding
        initial values and lower and upper bounds. The keys of the dictionary are the names
        of the fitting parameters from :meth:`_generate_pretty_params` method.

        Args:
            bounds (np.ndarray): The array of parameter bounds.

        Returns:
            dict: The fitting parameters with their corresponding initial values, and lower and
                  upper bounds, in the following format: {param: {'initial': val, 'lo': lo, 'hi': hi}}
        """
        param_names = list(self._generate_pretty_params(bounds).keys())
        param_bounds = {f'{param}': {'initial': val[0],
                                     'lo': val[1],
                                     'hi': val[2]} for param, val in
                        zip(param_names, bounds)}
        return param_bounds
