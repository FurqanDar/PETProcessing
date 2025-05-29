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
from scipy.signal import convolve as sp_conv


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
    vals = sp_conv(f, g, mode='full')
    return vals[:len(f)] * dt


@numba.njit(fastmath=True, cache=True)
def discrete_convolution_with_exponential(func_times: np.ndarray, func_vals: np.ndarray, k1: float, k2: float):
    r"""Computes the convolution of the given function with an exponential kernel.

    Given the provided function :math:`u(t)`, sampled evenly with respect to time, we calculate
    :math:`c(t) = k_{1} \exp(-k_{2} t) \otimes u(t)` where :math:`\otimes` represents
    the convolution operator. This implementation is :math:`\mathcal{O}(N)` due to the
    simple recurrence relationship arising from an exponential kernel. This implementation is based
    heavily on the `KMAPLIB <https://github.com/ShareKM/KMAP-C/blob/caeb10b7bec1e841132f879856949f00a9624883/src/kmaplib_common.cpp#L43-L70>`_
    implementation

    .. important::
        The function assumes that the times are evenly sampled. Answers will be incorrect if this is not the case.

    Args:
        func_vals (np.ndarray): Array containing function values for :math:`t\geq0`.
            Assumed to be evenly sampled with respect to :math:`t`.
        func_times (np.ndarray): Array containing time-points where :math:`t\geq0`.
            Assumed to be evenly sampled with respect to :math:`t`.
        k1 (float): Rate constant for transport from first tissue compartment.
        k2 (float): Rate constant for transport from second tissue compartment.

    Returns:
        (np.ndarray): Array containing the convolution of an exponential function with the provided function.

    See Also:
        :func:`calc_convolution_with_check` for a more general convolution function
    """
    dt = func_times[1] - func_times[0]
    num_times = len(func_times)
    c_out = np.zeros(num_times)

    prev = 0
    if k2 <= 1e-8:
        for i in range(0, num_times):
            prev += func_vals[i]
            c_out[i] = prev
        return k1 * c_out * dt
    else:
        _k1 = k1 * dt
        _k2 = k2 * dt
        ek2 = np.exp(-_k2)
        tmp = _k1 * (1.0 - ek2) / _k2
        u_tmp = tmp * func_vals
        for i in range(0, num_times):
            prev = prev * ek2 + u_tmp[i]
            c_out[i] = prev
        return c_out

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
    
        f(t) = \frac{k_{1}}{\Delta \alpha} \left[ (k_{4}-\alpha_{1})e^{-\alpha_{1}t} + (\alpha_{2}-k_{4})e^{-\alpha_{2}t}\right]
    
    where
    
    .. math::
    
        a&= k_{2}+k_{3}+k_{4}\\
        \alpha_{1}&=\frac{a-\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \alpha_{1}&=\frac{a+\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \Delta \alpha&=\alpha_2 - \alpha_1
    
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
    delta_a = alpha_2 - alpha_1
    
    return (k1 / delta_a) * ((k4 - alpha_1) * np.exp(-alpha_1 * t) + (alpha_2 - k4) * np.exp(-alpha_2 * t))


@numba.njit()
def response_function_serial_2tcm_c2(t: np.ndarray, k1: float, k2: float, k3: float, k4: float) -> np.ndarray:
    r"""The response function for second compartment in the *serial* 2TCM.

    .. math::
    
        f(t) = \frac{k_{1}k_{3}}{a} \left[ e^{-\alpha_{1}t} - e^{-\alpha_{2}t}\right]

    where

    .. math::
    
        a&= k_{2}+k_{3}+k_{4}\\
        \alpha_{1}&=\frac{a-\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \alpha_{1}&=\frac{a+\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \Delta \alpha&=\alpha_2 - \alpha_1

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
    delta_a = alpha_2 - alpha_1
    
    return (k1 * k3 / delta_a) * (np.exp(-alpha_1 * t) - np.exp(-alpha_2 * t))

@numba.njit(cache=True, fastmath=True)
def gen_tac_1tcm_cpet_from_tac(tac_times: np.ndarray,
                               tac_vals: np.ndarray,
                               k1: float,
                               k2: float,
                               vb: float = 0.0) -> list[np.ndarray, np.ndarray]:
    r"""Calculate the TTAC, given the input TAC, for a 1TCM as an explicit convolution.

    .. important::
        This function assumes that the provided input TAC is sampled evenly with respect to time.

    Computes the following:


    .. math::

        C_\mathrm{T} = C_{P}(t) \otimes k_{1}e^{-k_{2}t}


    where :math:`C_\mathrm{T}` is the output TTAC, and :math:`C_{P}(t)` is the input/plasma TAC


    Args:
        tac_times (np.ndarray): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.

    Returns:
        ((np.ndarray, np.ndarray)): Arrays containing the times and TTAC given the input TAC and parameters.
    """
    
    c1 = discrete_convolution_with_exponential(func_times=tac_times, func_vals=tac_vals, k1=k1, k2=k2)
    return [tac_times, (1.0-vb)*c1 + vb*tac_vals]


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


def gen_tac_2tcm_with_k4zero_cpet_from_tac(tac_times: np.ndarray,
                                           tac_vals: np.ndarray,
                                           k1: float,
                                           k2: float,
                                           k3: float,
                                           vb: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
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
        * :func:`gen_tac_2tcm_cpet_from_tac` for more details about the 2TCM function.

    """
    return gen_tac_2tcm_cpet_from_tac(tac_times=tac_times, tac_vals=tac_vals, k1=k1, k2=k2, k3=k3, k4=0, vb=vb)


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


@numba.njit(fastmath=True, cache=True)
def gen_tac_2tcm_cpet_from_tac(tac_times: np.ndarray,
                               tac_vals: np.ndarray,
                               k1: float,
                               k2: float,
                               k3: float,
                               k4: float,
                               vb: float = 0.0) -> list[np.ndarray, np.ndarray]:
    r"""Generates PET TAC values using the serial 2TCM.

    Computes the PET TAC, given the input TAC, using a serial Two Tissue Compartment Model (2TCM).
    Since this function uses :func:`discrete_convolution_with_exponential`, this function
    is also JIT'ed, and is therefore faster than :func:`generate_tac_serial_2tcm_cpet_from_tac`.

    .. important:
        This function assumes that the provided input TAC is sampled evenly with respect to time.


    We calculate

    .. math::

        C_\mathrm{T} = (1-v_{\mathrm{B}}) (C_{1} + C_{2}) + v_{\mathrm{B}} C_\mathrm{P}


    where :math:`C_\mathrm{T}` is the output PET TAC, :math:`v_{\mathrm{B}}` is the blood volume,
    :math:`C_{1}` and :math:`C_{2}` are the TACs for the first and second tissue compartments, and
    :math:`C_\mathrm{P}` is the input/plasma TAC. Furthermore,

    .. math::

        \begin{align}
        C_{1} &= \frac{k_{1}(k_{4} - \alpha_{2})}{\beta}C_{a} + \frac{k_{1}(\alpha_{1} - k_{4})}{\beta}C_{b}\\
        C_{2} &= \frac{k_{1}k_{3}}{\beta}C_{a} - \frac{k_{1}k_{3}}{\beta}C_{b}
        \end{align}

    where

    .. math::

        \begin{align}
        C_{a}&=C_\mathrm{P}(t) \otimes e^{-\alpha_{2}\times t}\\
        C_{b}&=C_\mathrm{P}(t) \otimes e^{-\alpha_{1}\times t}\\
        \beta&= \sqrt{x^{2}-4k_{2}k_{4}}\\
        \alpha_{1}&=\frac{x-\beta}{2}\\
        \alpha_{2}&=\frac{x+\beta}{2}\\
        x&= k_{2}+k_{3}+k_{4}
        \end{align}


    Args:
        tac_times (np.ndarray): Time points for the input TAC values.
        tac_vals (np.ndarray): TAC values corresponding to the time points, assumed to be in minutes.
        k1 (float): Rate constant for blood-to-first-tissue transport.
        k2 (float): Rate constant for first-tissue-to-blood transport.
        k3 (float): Rate constant for 1st compartment to 2nd compartment transport.
        k4 (float): Rate constant for 2nt compartment to 1st compartment transport.
        vb (float, optional): Vascular blood fraction. Defaults to 0.0.

    Returns:
        np.ndarray: Simulated PET TAC computed using the serial 2TCM.
    """
    k234 = k2 + k3 + k4
    beta = np.sqrt(k234 * k234 - 4.0 * k2 * k4)
    a1 = (k234 - beta) / 2.0
    a2 = (k234 + beta) / 2.0

    c_a1 = discrete_convolution_with_exponential(func_times=tac_times, func_vals=tac_vals, k1=1.0, k2=a1)
    c_a2 = discrete_convolution_with_exponential(func_times=tac_times, func_vals=tac_vals, k1=1.0, k2=a2)

    if beta <= 1.0e-8:
        f1 = 0.0
        f2 = 0.0
        b1 = 0.0
        b2 = 0.0
    else:
        f1 = k1 / beta * (k4 - a1)
        f2 = k1 / beta * (a2 - k4)
        b1 = k1 / beta * k3
        b2 = -b1

    c_1 = f1 * c_a1 + f2 * c_a2
    c_2 = b1 * c_a1 + b2 * c_a2
    c_pet = (1.0 - vb) * (c_1 + c_2) + vb * tac_vals

    return [tac_times, c_pet]