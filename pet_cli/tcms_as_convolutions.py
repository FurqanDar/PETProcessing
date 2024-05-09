"""
This module contains a collection of functions to compute Time-Activity Curves (TACs) for common Tissue Compartment
Models (TCMs). These models are commonly used for kinetic analysis of PET TACs.

Note:
    All response functions in this module are decorated with :func:`numba.jit`. It compiles the function to
    machine code at runtime (Just-In-Time compilation), which usually provides a significant speed-up.

Requires:
    The module relies on the :doc:`numpy <numpy:index>` and :doc:`numba <numba:index>` modules.

TODO:
    Add the derivations of the solutions to the Tissue Compartment Models in the module docstring.
    
"""

import numba
import numpy as np
from scipy.optimize import curve_fit as sp_fit


def calc_convolution_with_check(f: np.ndarray, g: np.ndarray, dt: float) -> np.ndarray:
    r"""Performs a discrete convolution of two arrays, assumed to represent time-series data. Checks if the arrays are
    of the same shape.
    
    Let ``f``:math:`=f(t)` and ``g``:math:`=g(t)` where both functions are 0 for :math:`t\leq0`. Then,
    the output, :math:`h(t)`, is
    
    .. math::
    
        h(t) = \int_{0}^{t}f(s)g(s-t)\mathrm{d}s
    
    Args:
        f (np.ndarray): Array containing the values for the input function.
        g (np.ndarray): Array containing values for the response function.
        dt (np.ndarray): The step-size, in the time-domain, between samples for ``f`` and ``g``.

    Returns:
        (np.ndarray): Convolution of the two arrays scaled by ``dt``.
        
    .. important::
        This function does not use :func:`numba.jit`. Therefore, it cannot be used directly inside JIT'ed functions.
        
    """
    assert len(f) == len(g), f"The provided arrays must have the same lengths! f:{len(f):<6} and g:{len(g):<6}."
    vals = np.convolve(f, g, mode='full')
    return vals[:len(f)] * dt


@numba.njit()
def response_function_1tcm_c1(t: np.ndarray, k1: float, k2: float) -> np.ndarray:
    r"""The response function for the 1TCM :math:`f(t)=k_1 e^{-k_{2}t}`
    
    Args:
        t (np.ndarray): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from tissue compartment back to plasma/blood.

    Returns:
        (np.ndarray): Array containing response function values given the constants.
    """
    return k1 * np.exp(-k2 * t)


@numba.njit()
def response_function_2tcm_with_k4zero_c1(t: np.ndarray, k1: float, k2: float, k3: float) -> np.ndarray:
    r"""
    The response function for first compartment in the serial 2TCM with
    :math:`k_{4}=0`; :math:`f(t)=k_{1}e^{-(k_{2} + k_{3})t}`.
    
    Args:
        t (np.ndarray): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to irreversible compartment.

    Returns:
        (np.ndarray): Array containing response function values for first compartment given the constants.
        
    See Also:
        :func:`response_function_2tcm_with_k4zero_c2`
        
    """
    return k1 * np.exp(-(k2 + k3) * t)


@numba.njit()
def response_function_2tcm_with_k4zero_c2(t: np.ndarray, k1: float, k2: float, k3: float) -> np.ndarray:
    r"""The response function for second compartment in the serial 2TCM with :math:`k_{4}=0`;
    :math:`f(t)=\frac{k_{1}k_{3}}{k_{2}+k_{3}}(1-e^{-(k_{2} + k_{3})t})`.

    Args:
        t (np.ndarray): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to irreversible compartment.

    Returns:
        (np.ndarray): Array containing response function values for first compartment given the constants.
    
    See Also:
        :func:`response_function_2tcm_with_k4zero_c1`
    """
    return ((k1 * k3) / (k2 + k3)) * (1.0 - np.exp(-(k2 + k3) * t))


@numba.njit()
def response_function_serial_2tcm_c1(t: np.ndarray, k1: float, k2: float, k3: float, k4: float) -> np.ndarray:
    r"""The response function for first compartment in the *serial* 2TCM.
    
    .. math::
    
        f(t) = \frac{k_{1}}{a} \left[ (k_{4}-\alpha_{1})e^{-\alpha_{1}t} + (\alpha_{2}-k_{4})e^{-\alpha_{2}t}\right]
    
    where
    
    .. math::
    
        a&= k_{2}+k_{3}+k_{4}\\
        \alpha_{1}&=\frac{a-\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \alpha_{1}&=\frac{a+\sqrt{a^{2}-4k_{2}k_{4}}}{2}
    
    Args:
        t (np.ndarray): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from first tissue compartment to second tissue compartment.
        k4 (float): Rate constant for transport from second tissue compartment back to first tissue compartment.

    Returns:
        (np.ndarray): Array containing response function values for first compartment given the constants.
        
    See Also:
        * :func:`response_function_serial_2tcm_c2`
        * :func:`response_function_2tcm_with_k4zero_c1` for when :math:`k_{4}=0` (irreversible second compartment).
        
    """
    a = k2 + k3 + k4
    alpha_1 = (a - np.sqrt((a ** 2.) - 4.0 * k2 * k4)) / 2.0
    alpha_2 = (a + np.sqrt((a ** 2.) - 4.0 * k2 * k4)) / 2.0
    
    return (k1 / a) * ((k4 - alpha_1) * np.exp(-alpha_1 * t) + (alpha_2 - k4) * np.exp(-alpha_2 * t))


@numba.njit()
def response_function_serial_2tcm_c2(t: np.ndarray, k1: float, k2: float, k3: float, k4: float) -> np.ndarray:
    r"""The response function for second compartment in the *serial* 2TCM.

    .. math::
    
        f(t) = \frac{k_{1}k_{3}}{a} \left[ e^{-\alpha_{1}t} - e^{-\alpha_{2}t}\right]

    where

    .. math::
    
        a&= k_{2}+k_{3}+k_{4}\\
        \alpha_{1}&=\frac{a-\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \alpha_{1}&=\frac{a+\sqrt{a^{2}-4k_{2}k_{4}}}{2}

    Args:
        t (np.ndarray): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from first tissue compartment to second tissue compartment.
        k4 (float): Rate constant for transport from second tissue compartment back to first tissue compartment.

    Returns:
        (np.ndarray): Array containing response function values for second compartment given the constants.
        
    See Also:
        * :func:`response_function_serial_2tcm_c2`
        * :func:`response_function_2tcm_with_k4zero_c2` for when :math:`k_{4}=0` (irreversible second compartment).
        
    """
    a = k2 + k3 + k4
    alpha_1 = (a - np.sqrt((a ** 2.) - 4.0 * k2 * k4)) / 2.0
    alpha_2 = (a + np.sqrt((a ** 2.) - 4.0 * k2 * k4)) / 2.0
    
    return (k1 * k3 / a) * (np.exp(-alpha_1 * t) - np.exp(-alpha_2 * t))


def generate_tac_1tcm_c1_from_tac(tac_times: np.ndarray,
                                  tac_vals: np.ndarray,
                                  k1: float,
                                  k2: float) -> np.ndarray:
    r"""Calculate the TTAC, given the input TAC, for a 1TCM as an explicit convolution.
    
    Args:
        tac_times (np.ndarray): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.

    Returns:
        ((np.ndarray, np.ndarray)): Arrays containing the times and TTAC given the input TAC and parameters.
        
    See Also:
        :func:`response_function_1tcm_c1` for more details about the 1TCM response function used for the convolution.
    """
    
    _resp_vals = response_function_1tcm_c1(t=tac_times, k1=k1, k2=k2)
    dt = tac_times[1] - tac_times[0]
    c1 = calc_convolution_with_check(f=tac_vals, g=_resp_vals, dt=dt)
    return np.asarray([tac_times, c1])


def generate_tac_2tcm_with_k4zero_c1_from_tac(tac_times: np.ndarray,
                                              tac_vals: np.ndarray,
                                              k1: float,
                                              k2: float,
                                              k3: float) -> np.ndarray:
    r"""
    Calculate the TTAC of the first comparment, given the input TAC, for a 2TCM (with :math:`k_{4}=0`) as an explicit
    convolution.
    
    Args:
        tac_times (np.ndarray): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to irreversible compartment.

    Returns:
        ((np.ndarray, np.ndarray)): Arrays containing the times and TTAC given the input TAC and parameters.
        
    See Also:
        :func:`response_function_2tcm_with_k4zero_c1` for more details about the 2TCM response function, of the first
        compartment, used for the convolution.
        
    """
    _resp_vals = response_function_2tcm_with_k4zero_c1(t=tac_times, k1=k1, k2=k2, k3=k3)
    dt = tac_times[1] - tac_times[0]
    c1 = calc_convolution_with_check(f=tac_vals, g=_resp_vals, dt=dt)
    return np.asarray([tac_times, c1])


def generate_tac_2tcm_with_k4zero_c2_from_tac(tac_times: np.ndarray,
                                              tac_vals: np.ndarray,
                                              k1: float,
                                              k2: float,
                                              k3: float) -> np.ndarray:
    r"""
    Calculate the TTAC of the second comparment, given the input TAC, for a 2TCM (with :math:`k_{4}=0`) as an
    explicit convolution.
    
    Args:
        tac_times (np.ndarray): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to irreversible compartment.

    Returns:
        ((np.ndarray, np.ndarray)): Arrays containing the times and TTAC given the input TAC and parameters.
        
    See Also:
        :func:`response_function_2tcm_with_k4zero_c2` for more details about the 2TCM response function, of the second
        compartment, used for the convolution.
        
    """
    _resp_vals = response_function_2tcm_with_k4zero_c2(t=tac_times, k1=k1, k2=k2, k3=k3)
    dt = tac_times[1] - tac_times[0]
    c2 = calc_convolution_with_check(f=tac_vals, g=_resp_vals, dt=dt)
    return np.asarray([tac_times, c2])


def generate_tac_2tcm_with_k4zero_cpet_from_tac(tac_times: np.ndarray,
                                                tac_vals: np.ndarray,
                                                k1: float,
                                                k2: float,
                                                k3: float) -> np.ndarray:
    r"""
    Calculate the PET-TTAC (sum of both compartments), given the input TAC, for a 2TCM (with :math:`k_{4}=0`) as an
    explicit convolution.
    
    Args:
        tac_times (np.ndarray): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to irreversible compartment.

    Returns:
        ((np.ndarray, np.ndarray)): Arrays containing the times and TTAC given the input TAC and parameters.
        
    See Also:
        * :func:`response_function_2tcm_with_k4zero_c1` for more details about the 2TCM response function, of the first
            compartment, used for the convolution.
        * :func:`response_function_2tcm_with_k4zero_c2` for more details about the 2TCM response function, of the second
            compartment, used for the convolution.
        
    """
    _resp_vals = response_function_2tcm_with_k4zero_c1(t=tac_times, k1=k1, k2=k2, k3=k3)
    _resp_vals += response_function_2tcm_with_k4zero_c2(t=tac_times, k1=k1, k2=k2, k3=k3)
    dt = tac_times[1] - tac_times[0]
    cpet = calc_convolution_with_check(f=tac_vals, g=_resp_vals, dt=dt)
    return np.asarray([tac_times, cpet])


def generate_tac_serial_2tcm_c1_from_tac(tac_times: np.ndarray,
                                         tac_vals: np.ndarray,
                                         k1: float,
                                         k2: float,
                                         k3: float,
                                         k4: float) -> np.ndarray:
    r"""
    Calculate the TTAC of the first comparment, given the input TAC, for a serial 2TCM as an explicit convolution.
    
    Args:
        tac_times (np.ndarray): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to second compartment.
        k4 (float): Rate constant for transport from second tissue compartment back to first tissue compartment.

    Returns:
        ((np.ndarray, np.ndarray)): Arrays containing the times and TTAC given the input TAC and parameters.
        
    See Also:
        * :func:`response_function_serial_2tcm_c1` for more details about the 2TCM response function, of the first
            compartment, used for the convolution.
        * :func:`response_function_2tcm_with_k4zero_c1` for more details about the 2TCM response function
            (with :math:`k_{4}=0`), of the first compartment, used for the convolution.
        
    """
    _resp_vals = response_function_serial_2tcm_c1(t=tac_times, k1=k1, k2=k2, k3=k3, k4=k4)
    dt = tac_times[1] - tac_times[0]
    c1 = calc_convolution_with_check(f=tac_vals, g=_resp_vals, dt=dt)
    return np.asarray([tac_times, c1])


def generate_tac_serial_2tcm_c2_from_tac(tac_times: np.ndarray,
                                         tac_vals: np.ndarray,
                                         k1: float,
                                         k2: float,
                                         k3: float,
                                         k4: float) -> np.ndarray:
    r"""
    Calculate the TTAC of the second comparment, given the input TAC, for a serial 2TCM as an explicit convolution.

    Args:
        tac_times (np.ndarray): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to second compartment.
        k4 (float): Rate constant for transport from second tissue compartment back to first tissue compartment.

    Returns:
        ((np.ndarray, np.ndarray)): Arrays containing the times and TTAC given the input TAC and parameters.

    See Also:
        * :func:`response_function_serial_2tcm_c2` for more details about the 2TCM response function, of the second
            compartment, used for the convolution.
        * :func:`response_function_2tcm_with_k4zero_c2` for more details about the 2TCM response function
            (with :math:`k_{4}=0`), of the second compartment, used for the convolution.
        
    """
    _resp_vals = response_function_serial_2tcm_c2(t=tac_times, k1=k1, k2=k2, k3=k3, k4=k4)
    dt = tac_times[1] - tac_times[0]
    c2 = calc_convolution_with_check(f=tac_vals, g=_resp_vals, dt=dt)
    return np.asarray([tac_times, c2])


def generate_tac_serial_2tcm_cpet_from_tac(tac_times: np.ndarray,
                                           tac_vals: np.ndarray,
                                           k1: float,
                                           k2: float,
                                           k3: float,
                                           k4: float) -> np.ndarray:
    r"""
    Calculate the PET-TTAC (sum of both compartments), given the input TAC, for a serial 2TCM as an explicit
    convolution.

    Args:
        tac_times (np.ndarray): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to second compartment.
        k4 (float): Rate constant for transport from second tissue compartment back to first tissue compartment.

    Returns:
        ((np.ndarray, np.ndarray)): Arrays containing the times and TTAC given the input TAC and parameters.

    See Also:
        * :func:`response_function_serial_2tcm_c1` for more details about the 2TCM response function, of the first
            compartment, used for the convolution.
        * :func:`response_function_2tcm_with_k4zero_c1` for more details about the 2TCM response function
            (with :math:`k_{4}=0`), of the first compartment, used for the convolution.
        * :func:`response_function_serial_2tcm_c2` for more details about the 2TCM response function, of the second
            compartment, used for the convolution.
        * :func:`response_function_2tcm_with_k4zero_c2` for more details about the 2TCM response function
            (with :math:`k_{4}=0`), of the second compartment, used for the convolution.
        
    """
    _resp_vals = response_function_serial_2tcm_c1(t=tac_times, k1=k1, k2=k2, k3=k3, k4=k4)
    _resp_vals += response_function_serial_2tcm_c2(t=tac_times, k1=k1, k2=k2, k3=k3, k4=k4)
    dt = tac_times[1] - tac_times[0]
    cpet = calc_convolution_with_check(f=tac_vals, g=_resp_vals, dt=dt)
    return np.asarray([tac_times, cpet])


def fit_tac_to_1tcm(tgt_tac_vals: np.ndarray,
                    input_tac_times: np.ndarray,
                    input_tac_vals: np.ndarray,
                    k1_guess: float = 0.5,
                    k2_guess: float = 0.5):
    r"""
    Fits a target Time Activity Curve (TAC) to the one tissue compartment model (1TCM), given the input TAC values,
    times, and starting guesses for the kinetic parameters k1 and k2.

    .. important::
        This function assumes that the input TAC  and target TAC are uniformly sampled with respect to time since we
        perform convolutions.
    
    This is a simple wrapper around :func:`scipy.optimize.curve_fit` and does not use any bounds for the different
    parameters.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit.
        input_tac_times (np.ndarray): Input TAC times,
        input_tac_vals (np.ndarray): Input TAC values.
        k1_guess (float): Starting guess for parameter k1. Defaults to 0.5.
        k2_guess (float): Starting guess for parameter k2. Defaults to 0.5.

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``).

    See Also:
        * :func:`scipy.optimize.curve_fit`
        * :func:`generate_tac_1tcm_c1_from_tac`
        
    """
    def _fitting_tac(tac_times: np.ndarray, k1: float, k2: float):
        tac = generate_tac_1tcm_c1_from_tac(tac_times=tac_times, tac_vals=input_tac_vals, k1=k1, k2=k2)[1]
        return tac
    
    fit_parameters, fit_params_covariance = sp_fit(f=_fitting_tac, xdata=input_tac_times, ydata=tgt_tac_vals,
                                                   p0=(k1_guess, k2_guess))
    return fit_parameters, fit_params_covariance


def fit_tac_to_1tcm_with_bounds(tgt_tac_vals: np.ndarray,
                                input_tac_times: np.ndarray,
                                input_tac_vals: np.ndarray,
                                k1_bounds: tuple[float, float, float] = (0.5, 1e-6, 5.0),
                                k2_bounds: tuple[float, float, float] = (0.5, 1e-6, 5.0)) -> tuple:
    r"""
    Fits a target Time Activity Curve (TAC) to the one tissue compartment model (1TCM), given the input TAC values,
    times, and bounds for the kinetic parameters k1 and k2.

    .. important::
        This function assumes that the input TAC  and target TAC are uniformly sampled with respect to time since we
        perform convolutions.
        
    This function is a wrapper around `scipy.optimize.curve_fit` and uses parameter bounds during optimization. The
    bounds for each parameter are formatted as: ``(starting_value, lo_bound, hi_bound)``.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the 1TCM.
        input_tac_times (np.ndarray): Input TAC times.
        input_tac_vals (np.ndarray): Input TAC values.
        k1_bounds (tuple[float, float, float]): The bounds for parameter k1. Defaults to (0.5, 1e-6, 5.0).
        k2_bounds (tuple[float, float, float]): The bounds for parameter k2. Defaults to (0.5, 1e-6, 5.0).

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``).

    See Also:
        * :func:`scipy.optimize.curve_fit`
        * :func:`generate_tac_1tcm_c1_from_tac`
        
    """
    def _fitting_tac(tac_times: np.ndarray, k1: float, k2: float):
        tac = generate_tac_1tcm_c1_from_tac(tac_times=tac_times, tac_vals=input_tac_vals, k1=k1, k2=k2)[1]
        return tac
    
    st_vals = (k1_bounds[0], k2_bounds[0])
    lo_vals = (k1_bounds[1], k2_bounds[1])
    hi_vals = (k1_bounds[2], k2_bounds[2])
    
    fit_parameters, fit_params_covariance = sp_fit(f=_fitting_tac, xdata=input_tac_times, ydata=tgt_tac_vals,
                                                   p0=st_vals, bounds=(lo_vals, hi_vals))
    return fit_parameters, fit_params_covariance


def fit_tac_to_irreversible_2tcm(tgt_tac_vals: np.ndarray,
                                 input_tac_times: np.ndarray,
                                 input_tac_vals: np.ndarray,
                                 k1_guess: float = 0.5,
                                 k2_guess: float = 0.5,
                                 k3_guess: float = 0.5):
    r"""
    Fits a target Time Activity Curve (TAC) to the irreversible two tissue compartment model (2TCM), given the input TAC
    values, times, and starting guesses for the kinetic parameters k1, k2 and k3.

    .. important::
        This function assumes that the input TAC  and target TAC are uniformly sampled with respect to time since we
        perform convolutions.

    This is a simple wrapper around :func:`scipy.optimize.curve_fit` and does not use any bounds for the different
    parameters.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit.
        input_tac_times (np.ndarray): Input TAC times,
        input_tac_vals (np.ndarray): Input TAC values.
        k1_guess (float): Starting guess for parameter k1. Defaults to 0.5.
        k2_guess (float): Starting guess for parameter k2. Defaults to 0.5.
        k3_guess (float): Starting guess for parameter k3. Defaults to 0.5.

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``).

    See Also:
        * :func:`scipy.optimize.curve_fit`
        * :func:`generate_tac_2tcm_with_k4zero_cpet_from_tac`

    """
    def _fitting_tac(tac_times: np.ndarray, k1: float, k2: float, k3: float):
        _tac_gen = generate_tac_2tcm_with_k4zero_cpet_from_tac
        tac = _tac_gen(tac_times=tac_times, tac_vals=input_tac_vals, k1=k1, k2=k2, k3=k3)[1]
        return tac
    
    fit_parameters, fit_params_covariance = sp_fit(f=_fitting_tac, xdata=input_tac_times, ydata=tgt_tac_vals,
                                                   p0=(k1_guess, k2_guess, k3_guess))
    return fit_parameters, fit_params_covariance


def fit_tac_to_irreversible_2tcm_with_bounds(tgt_tac_vals: np.ndarray,
                                             input_tac_times: np.ndarray,
                                             input_tac_vals: np.ndarray,
                                             k1_bounds: tuple[float, float, float] = (0.5, 1e-6, 5.0),
                                             k2_bounds: tuple[float, float, float] = (0.5, 1e-6, 5.0),
                                             k3_bounds: tuple[float, float, float] = (0.5, 1e-6, 5.0)):
    r"""
    Fits a target Time Activity Curve (TAC) to the irreversible two tissue compartment model (2TCM), given the input TAC
    values, times, and bounds for the kinetic parameters k1, k2 and k3.

    .. important::
        This function assumes that the input TAC  and target TAC are uniformly sampled with respect to time since we
        perform convolutions.

    This function is a wrapper around `scipy.optimize.curve_fit` and uses parameter bounds during optimization. The
    bounds for each parameter are formatted as: ``(starting_value, lo_bound, hi_bound)``.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the 1TCM.
        input_tac_times (np.ndarray): Input TAC times.
        input_tac_vals (np.ndarray): Input TAC values.
        k1_bounds (tuple[float, float, float]): The bounds for parameter k1. Defaults to (0.5, 1e-6, 5.0).
        k2_bounds (tuple[float, float, float]): The bounds for parameter k2. Defaults to (0.5, 1e-6, 5.0).
        k3_bounds (tuple[float, float, float]): The bounds for parameter k3. Defaults to (0.5, 1e-6, 5.0).

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``).

    See Also:
        * :func:`scipy.optimize.curve_fit`
        * :func:`generate_tac_2tcm_with_k4zero_cpet_from_tac`

    """
    def _fitting_tac(tac_times: np.ndarray, k1: float, k2: float, k3: float):
        _tac_gen = generate_tac_2tcm_with_k4zero_cpet_from_tac
        tac = _tac_gen(tac_times=tac_times, tac_vals=input_tac_vals, k1=k1, k2=k2, k3=k3)[1]
        return tac
    
    st_vals = (k1_bounds[0], k2_bounds[0], k3_bounds[0])
    lo_vals = (k1_bounds[1], k2_bounds[1], k3_bounds[1])
    hi_vals = (k1_bounds[2], k2_bounds[2], k3_bounds[2])
    
    fit_parameters, fit_params_covariance = sp_fit(f=_fitting_tac, xdata=input_tac_times, ydata=tgt_tac_vals,
                                                   p0=st_vals, bounds=(lo_vals, hi_vals))
    return fit_parameters, fit_params_covariance


def fit_tac_to_serial_2tcm(tgt_tac_vals: np.ndarray,
                           input_tac_times: np.ndarray,
                           input_tac_vals: np.ndarray,
                           k1_guess: float = 0.5,
                           k2_guess: float = 0.5,
                           k3_guess: float = 0.5,
                           k4_guess: float = 0.5):
    r"""
    Fits a target Time Activity Curve (TAC) to the serial two tissue compartment model (2TCM), given the input TAC
    values, times, and starting guesses for the kinetic parameters k1, k2 and k3.

    .. important::
        This function assumes that the input TAC  and target TAC are uniformly sampled with respect to time since we
        perform convolutions.

    This is a simple wrapper around :func:`scipy.optimize.curve_fit` and does not use any bounds for the different
    parameters.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit.
        input_tac_times (np.ndarray): Input TAC times,
        input_tac_vals (np.ndarray): Input TAC values.
        k1_guess (float): Starting guess for parameter k1. Defaults to 0.5.
        k2_guess (float): Starting guess for parameter k2. Defaults to 0.5.
        k3_guess (float): Starting guess for parameter k3. Defaults to 0.5.
        k4_guess (float): Starting guess for parameter k4. Defaults to 0.5.

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``).

    See Also:
        * :func:`scipy.optimize.curve_fit`
        * :func:`generate_tac_serial_2tcm_cpet_from_tac`

    """
    def _fitting_tac(tac_times: np.ndarray, k1: float, k2: float, k3: float, k4: float):
        
        _tac_gen = generate_tac_serial_2tcm_cpet_from_tac
        tac = _tac_gen(tac_times=tac_times, tac_vals=input_tac_vals, k1=k1, k2=k2, k3=k3, k4=k4)[1]
        return tac
    
    fit_parameters, fit_params_covariance = sp_fit(f=_fitting_tac, xdata=input_tac_times, ydata=tgt_tac_vals,
                                                   p0=(k1_guess, k2_guess, k3_guess, k4_guess))
    return fit_parameters, fit_params_covariance


def fit_tac_to_serial_2tcm_with_bounds(tgt_tac_vals: np.ndarray,
                                       input_tac_times: np.ndarray,
                                       input_tac_vals: np.ndarray,
                                       k1_bounds: tuple[float, float, float] = (0.5, 1e-6, 5.0),
                                       k2_bounds: tuple[float, float, float] = (0.5, 1e-6, 5.0),
                                       k3_bounds: tuple[float, float, float] = (0.5, 1e-6, 5.0),
                                       k4_bounds: tuple[float, float, float] = (0.5, 1e-6, 5.0)):
    r"""
    Fits a target Time Activity Curve (TAC) to the serial two tissue compartment model (2TCM), given the input TAC
    values, times, and bounds for the kinetic parameters k1, k2, k3 and k4.

    .. important::
        This function assumes that the input TAC  and target TAC are uniformly sampled with respect to time since we
        perform convolutions.

    This function is a wrapper around `scipy.optimize.curve_fit` and uses parameter bounds during optimization. The
    bounds for each parameter are formatted as: ``(starting_value, lo_bound, hi_bound)``.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the 1TCM.
        input_tac_times (np.ndarray): Input TAC times.
        input_tac_vals (np.ndarray): Input TAC values.
        k1_bounds (tuple[float, float, float]): The bounds for parameter k1. Defaults to (0.5, 1e-6, 5.0).
        k2_bounds (tuple[float, float, float]): The bounds for parameter k2. Defaults to (0.5, 1e-6, 5.0).
        k3_bounds (tuple[float, float, float]): The bounds for parameter k3. Defaults to (0.5, 1e-6, 5.0).
        k4_bounds (tuple[float, float, float]): The bounds for parameter k4. Defaults to (0.5, 1e-6, 5.0).

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``).

    See Also:
        * :func:`scipy.optimize.curve_fit`
        * :func:`generate_tac_serial_2tcm_cpet_from_tac`

    """
    def _fitting_tac(tac_times: np.ndarray, k1: float, k2: float, k3: float, k4: float):
        _tac_gen = generate_tac_serial_2tcm_cpet_from_tac
        tac = _tac_gen(tac_times=tac_times, tac_vals=input_tac_vals, k1=k1, k2=k2, k3=k3, k4=k4)[1]
        return tac
    
    st_vals = (k1_bounds[0], k2_bounds[0], k3_bounds[0], k4_bounds[0])
    lo_vals = (k1_bounds[1], k2_bounds[1], k3_bounds[1], k4_bounds[1])
    hi_vals = (k1_bounds[2], k2_bounds[2], k3_bounds[2], k4_bounds[2])
    
    fit_parameters, fit_params_covariance = sp_fit(f=_fitting_tac, xdata=input_tac_times, ydata=tgt_tac_vals,
                                                   p0=st_vals, bounds=(lo_vals, hi_vals))
    return fit_parameters, fit_params_covariance
