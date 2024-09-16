"""
Todo:
    * Add the Ichise paper citations.
    * Add the SRTM and FRTM paper citations.
    * Add implementations for the SRTM2 and FRTM2 analyses.
    
"""
import json
import os.path
from typing import Union
import numpy as np
from scipy.optimize import curve_fit as sp_fit
import numba
from .graphical_analysis import get_index_from_threshold
from .graphical_analysis import cumulative_trapezoidal_integral as cum_trapz
from . import tcms_as_convolutions as tcms_conv
from ..utils.image_io import safe_load_tac


def calc_srtm_tac(tac_times: np.ndarray, ref_tac_vals: np.ndarray, r1: float, k2: float, bp: float) -> np.ndarray:
    r"""
    Calculate the Time Activity Curve (TAC) using the Simplified Reference Tissue Model (SRTM) with the given reference
    TAC and kinetic parameters.
    
    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.
    
    
    The SRTM TAC can be calculated as:
    
    .. math::
        
        C(t)=R_{1}C_\mathrm{R}(t) + \left(k_{2} - \frac{R_{1}k_{2}}{1+\mathrm{BP}}\right)C_\mathrm{R}(t)\otimes
        \exp\left(- \frac{k_{2}t}{1+\mathrm{BP}}\right),
    
    
    where :math:`C_\mathrm{R}(t)` is the reference TAC, :math:`R_{1}=\frac{k_1^\prime}{k_1}`, :math:`k_{2}` is the
    rate-constant from the tissue compartment to plasma, and :math:`\mathrm{BP}` is the binding potential.
    
    
    Args:
        tac_times (np.ndarray): Times for the reference TAC.
        r1 (float): The ratio of the clearance rate of tracer from plasma to the reference to the transfer rate of the
            tracer from plasma to the tissue; :math:`R_{1}\equiv\frac{k_1^\prime}{k_1}`.
        k2 (float): The rate constant for the transfer of the tracer from tissue compartment to plasma.
        bp (float): The binding potential of the tracer in the tissue.
        ref_tac_vals (np.ndarray): The values of the reference TAC.

    Returns:
        np.ndarray: TAC values calculated using SRTM.


    Raises:
        AssertionError: If the reference TAC and times are different dimensions.
        
        
    """
    first_term = r1 * ref_tac_vals
    bp_coeff = k2 / (1.0 + bp)
    exp_term = np.exp(-bp_coeff * tac_times)
    dt = tac_times[1] - tac_times[0]
    second_term = (k2 - r1 * bp_coeff) * tcms_conv.calc_convolution_with_check(f=exp_term, g=ref_tac_vals, dt=dt)

    return first_term + second_term


def _calc_simplified_frtm_tac(tac_times: np.ndarray,
                              ref_tac_vals: np.ndarray,
                              r1: float,
                              a1: float,
                              a2: float,
                              alpha_1: float,
                              alpha_2: float) -> np.ndarray:
    r"""
    Calculate the Time Activity Curve (TAC) for the Full Reference Tissue Model (FRTM) given the reference TAC and
    simplified coefficients. The coefficients can be generated from kinetic constants using
    :func:`_calc_frtm_params_from_kinetic_params`
    
    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.
    
    We use a more compact form for the FRTM:
    
    .. math::
    
        C(t) = R_{1}C_\mathrm{R}(t) + \left[ A_{1}e^{-\alpha_{1}t} + A_{2}e^{-\alpha_{2}t} \right] \otimes
        C_\mathrm{R}(t),
    
    where :math:`R_{1}\equiv\frac{k_1^\prime}{k_1}`, and :math:`A_{1},\,A_{2},\,\alpha_1,\,\alpha_2` can be calculated
    from the underlying kinetic constants. See :func:`_calc_frtm_params_from_kinetic_params` for more details about
    the parameter calculation.
    
    Args:
        tac_times (np.ndarray): Times for the reference TAC.
        r1 (float): The ratio of the clearance rate of tracer from plasma to the reference to the transfer rate of the
            tracer from plasma to the tissue; :math:`R_{1}\equiv\frac{k_1^\prime}{k_1}`.
        a1 (float): Coefficient of the first exponential term.
        a2 (float): Coefficient of the second exponential term.
        alpha_1 (float): Coefficient inside the first exponential.
        alpha_2 (float): Coefficient inside the second exponential.
        ref_tac_vals (np.ndarray): The values of the reference TAC.

    Returns:
        np.ndarray: TAC values calculated using FRTM.

    
    Raises:
        AssertionError: If the reference TAC and times are different dimensions.

    """
    first_term = r1 * ref_tac_vals
    exp_funcs = a1 * np.exp(-alpha_1 * tac_times) + a2 * np.exp(-alpha_2 * tac_times)
    dt = tac_times[1] - tac_times[0]
    second_term = tcms_conv.calc_convolution_with_check(f=exp_funcs, g=ref_tac_vals, dt=dt)
    return first_term + second_term


def _calc_frtm_params_from_kinetic_params(r1: float,
                                          k2: float,
                                          k3: float,
                                          k4: float) -> tuple[float, float, float, float, float]:
    r"""
    Calculates the parameters (coefficients) for the simplified FRTM function (:func:`_calc_simplified_frtm_tac`) given
    the kinetic constants.
    
    The parameters are defined as:
    
    .. math::
    
        \alpha_{1} &= \frac{\beta - \chi}{2}\\
        \alpha_{2} &= \frac{\beta + \chi}{2}\\
        A_{1} &= \left(\frac{k_{3} + k_{4} -\alpha_{2}}{\chi} \right)\left( \frac{k_{2}}{R_{1}} - \alpha_{2} \right)\\
        A_{2} &= \left(\frac{\alpha_{1}-k_{3} - k_{4} }{\chi} \right)\left( \frac{k_{2}}{R_{1}} - \alpha_{1} \right),
    
    where additionally we have:
    
    .. math::
    
        \alpha_{1} &= \frac{k_{2} + k_{3} + k_{4} - \sqrt{\left( k_{2} + k_{3} + k_{4} \right)^2 - 4k_{2}k_{4}}}{2}\\
        \alpha_{2} &= \frac{k_{2} + k_{3} + k_{4} + \sqrt{\left( k_{2} + k_{3} + k_{4} \right)^2 - 4k_{2}k_{4}}}{2}\\
        A_{1} &= \left( \frac{k_{3} + k_{4} -\alpha_{2}}{\alpha_{1} - \alpha_{2}} \right)\left( \frac{k_{2}}{R_{1}}
        - \alpha_{2} \right)\\
        A_{2} &= \left(  \frac{\alpha_{1}-k_{3} - k_{4} }{\alpha_{1} - \alpha_{2}} \right)\left( \frac{k_{2}}{R_{1}}
        - \alpha_{1} \right)
    
    
    Args:
        r1 (float): The ratio of the clearance rate of tracer from plasma to the reference to the transfer rate of the
            tracer from plasma to the tissue; :math:`R_{1}\equiv\frac{k_1^\prime}{k_1}`.
        k2 (float): The rate of tracer transfer from the first tissue compartment to plasma.
        k3 (float): The rate of tracer transfer from the first tissue compartment to the second tissue compartment.
        k4 (float): The rate of tracer transfer from the second tissue compartment to the first tissue compartment.

    Returns:
        tuple: (``r1``, ``a1``, ``a2``, ``alpha_1``, ``alpha_2``) parameters for :func:`_calc_simplified_frtm_tac`.
    """
    beta = k2 + k3 + k4
    chi = np.sqrt(beta ** 2. - 4.0 * k2 * k4)
    alpha_1 = (beta - chi) / 2.0
    alpha_2 = (beta + chi) / 2.0
    a1 = (k3 + k4 - alpha_2) / chi * (k2 / r1 - alpha_2)
    a2 = (alpha_1 - k3 - k4) / chi * (k2 / r1 - alpha_1)
    return r1, a1, a2, alpha_1, alpha_2


def calc_frtm_tac(tac_times: np.ndarray,
                  ref_tac_vals: np.ndarray,
                  r1: float,
                  k2: float,
                  k3: float,
                  k4: float) -> np.ndarray:
    r"""
    Calculate the Time Activity Curve (TAC) using the Full Reference Tissue Model (SRTM) with the given reference
    TAC and kinetic parameters.
    
    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.
        
    The FRTM TAC can be calculated as:
    
    .. math::
    
        C(t) = R_{1}C_\mathrm{R}(t) + \left[ A_{1}e^{-\alpha_{1}t} + A_{2}e^{-\alpha_{2}t} \right] \otimes
        C_\mathrm{R}(t),
        
    where additionally we have:
    
    .. math::
    
        \alpha_{1} &= \frac{k_{2} + k_{3} + k_{4} - \sqrt{\left( k_{2} + k_{3} + k_{4} \right)^2 - 4k_{2}k_{4}}}{2}\\
        \alpha_{2} &= \frac{k_{2} + k_{3} + k_{4} + \sqrt{\left( k_{2} + k_{3} + k_{4} \right)^2 - 4k_{2}k_{4}}}{2}\\
        A_{1} &= \left( \frac{k_{3} + k_{4} -\alpha_{2}}{\alpha_{1} - \alpha_{2}} \right)\left( \frac{k_{2}}{R_{1}}
        - \alpha_{2} \right)\\
        A_{2} &= \left(  \frac{\alpha_{1}-k_{3} - k_{4} }{\alpha_{1} - \alpha_{2}} \right)\left( \frac{k_{2}}{R_{1}}
        - \alpha_{1} \right)
        
    
    Args:
        tac_times (np.ndarray): Times for the reference TAC.
        r1 (float): The ratio of the clearance rate of tracer from plasma to the reference to the transfer rate of the
            tracer from plasma to the tissue; :math:`R_{1}\equiv\frac{k_1^\prime}{k_1}`.
        k2 (float): The rate of tracer transfer from the first tissue compartment to plasma.
        k3 (float): The rate of tracer transfer from the first tissue compartment to the second tissue compartment.
        k4 (float): The rate of tracer transfer from the second tissue compartment to the first tissue compartment.
        ref_tac_vals (np.ndarray): The values of the reference TAC.

    Returns:
        np.ndarray: TAC values calculated using FRTM.
        
    Raises:
        AssertionError: If the reference TAC and times are different dimensions.
        
    See Also:
        * :func:`_calc_simplified_frtm`
        * :func:`_calc_frtm_params_from_kinetic_params`

    """
    r1_n, a1, a2, alpha_1, alpha_2 = _calc_frtm_params_from_kinetic_params(r1=r1, k2=k2, k3=k3, k4=k4)
    return _calc_simplified_frtm_tac(tac_times=tac_times, ref_tac_vals=ref_tac_vals, r1=r1_n, a1=a1, a2=a2,
                                     alpha_1=alpha_1, alpha_2=alpha_2)


def fit_srtm_to_tac(tgt_tac_vals: np.ndarray,
                    ref_tac_times: np.ndarray,
                    ref_tac_vals: np.ndarray,
                    r1_start: float = 0.5,
                    k2_start: float = 0.5,
                    bp_start: float = 0.5) -> tuple:
    r"""
    Fit SRTM to the provided target Time Activity Curve (TAC), given the reference TAC, times, and starting guesses for
    the kinetic parameters.
    
    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.
    
    This is a simple wrapper around :func:`scipy.optimize.curve_fit` and does not use any bounds for the different
    parameters.
    
    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the SRTM.
        ref_tac_times (np.ndarray): Reference TAC values.
        ref_tac_vals (np.ndarray): Reference (and Target) TAC times.
        r1_start (float): Starting guess for the :math:`R_1\equiv\frac{k_1^\prime}{k_1}` parameter.
        k2_start (float): Starting guess for :math:`k_2` parameter.
        bp_start (float): Starting guess for the binding potential.

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``). Output from :func:`scipy.optimize.curve_fit`
        
    Raises:
        AssertionError: If the reference TAC and times are different dimensions.
        
    See Also:
        * :func:`calc_srtm_tac`
        
    """
    def _fitting_srtm(tac_times, r1, k2, bp):
        return calc_srtm_tac(tac_times=tac_times, ref_tac_vals=ref_tac_vals, r1=r1, k2=k2, bp=bp)
    
    starting_values = [r1_start, k2_start, bp_start]
    
    return sp_fit(f=_fitting_srtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=starting_values)


def fit_srtm2_to_tac(tgt_tac_vals: np.ndarray,
                     ref_tac_times: np.ndarray,
                     ref_tac_vals: np.ndarray,
                     k2_prime: float = 0.5,
                     r1_start: float = 0.5,
                     bp_start: float = 0.5) -> tuple:
    r"""
    Fit SRTM2 to the provided target Time Activity Curve (TAC), given the reference TAC, times, and starting guesses for
    the kinetic parameters.

    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.

    This is a simple wrapper around :func:`scipy.optimize.curve_fit` and does not use any bounds for the different
    parameters.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the SRTM2.
        ref_tac_times (np.ndarray): Reference TAC values.
        ref_tac_vals (np.ndarray): Reference (and Target) TAC times.
        k2_prime (float): The :math:`k_2^\prime` value.` Defaults to 0.5.
        r1_start (float): Starting guess for the :math:`R_1\equiv\frac{k_1^\prime}{k_1}` parameter.
        bp_start (float): Starting guess for the binding potential.

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``). Output from :func:`scipy.optimize.curve_fit`

    Raises:
        AssertionError: If the reference TAC and times are different dimensions.

    See Also:
        * :func:`calc_srtm_tac`
        * :func:`fit_srtm_to_tac`

    """

    def _fitting_srtm(tac_times, r1, bp):
        return calc_srtm_tac(tac_times=tac_times, ref_tac_vals=ref_tac_vals, r1=r1, k2=k2_prime, bp=bp)

    starting_values = [r1_start, bp_start]

    return sp_fit(f=_fitting_srtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=starting_values)



def fit_srtm_to_tac_with_bounds(tgt_tac_vals: np.ndarray,
                                ref_tac_times: np.ndarray,
                                ref_tac_vals: np.ndarray,
                                r1_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                k2_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                bp_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0])) -> tuple:
    r"""
    Fit SRTM to the provided target Time Activity Curve (TAC), given the reference TAC, times, and bounds for
    the kinetic parameters.
    
    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.

    This function is a wrapper around `scipy.optimize.curve_fit` and uses parameter bounds during optimization. The
    bounds for each parameter are formatted as: ``(starting_value, lo_bound, hi_bound)``.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the SRTM.
        ref_tac_times (np.ndarray): Times of the reference TAC data.
        ref_tac_vals (np.ndarray): Reference TAC values.
        r1_bounds (np.ndarray): The bounds for the :math:`R_1\equiv\frac{k_1^\prime}{k_1}` parameter.
        Defaults to [0.5, 0.0, 10.0].
        k2_bounds (np.ndarray): The bounds for :math:`k_2` parameter. Defaults to [0.5, 0.0, 10.0].
        bp_bounds (np.ndarray): The bounds for the binding potential parameter. Defaults to [0.5, 0.0, 10.0].

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``). Output from `scipy.optimize.curve_fit`.
        
    Raises:
        AssertionError: If the target TAC and times are different dimensions.

    See Also:
        * :func:`calc_srtm_tac`

    """
    def _fitting_srtm(tac_times, r1, k2, bp):
        return calc_srtm_tac(tac_times=tac_times, ref_tac_vals=ref_tac_vals, r1=r1, k2=k2, bp=bp)

    st_values = (r1_bounds[0], k2_bounds[0], bp_bounds[0])
    lo_values = (r1_bounds[1], k2_bounds[1], bp_bounds[1])
    hi_values = (r1_bounds[2], k2_bounds[2], bp_bounds[2])

    return sp_fit(f=_fitting_srtm, xdata=ref_tac_times, ydata=tgt_tac_vals,
                  p0=st_values, bounds=[lo_values, hi_values])


def fit_srtm2_to_tac_with_bounds(tgt_tac_vals: np.ndarray,
                                 ref_tac_times: np.ndarray,
                                 ref_tac_vals: np.ndarray,
                                 k2_prime: float = 0.5,
                                 r1_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                 bp_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0])) -> tuple:
    r"""
    Fit SRTM2 to the provided target Time Activity Curve (TAC), given the reference TAC, times, and bounds for
    the kinetic parameters.

    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.

    This function is a wrapper around `scipy.optimize.curve_fit` and uses parameter bounds during optimization. The
    bounds for each parameter are formatted as: ``(starting_value, lo_bound, hi_bound)``.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the SRTM2.
        ref_tac_times (np.ndarray): Times of the reference TAC data.
        ref_tac_vals (np.ndarray): Reference TAC values.
        k2_prime (int): The value for :math:`k_2^\prime`. Defaults to 0.5.
        r1_bounds (np.ndarray): The bounds for the :math:`R_1\equiv\frac{k_1^\prime}{k_1}` parameter.
        Defaults to [0.5, 0.0, 10.0].
        bp_bounds (np.ndarray): The bounds for the binding potential parameter. Defaults to [0.5, 0.0, 10.0].

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``). Output from `scipy.optimize.curve_fit`.

    Raises:
        AssertionError: If the target TAC and times are different dimensions.

    See Also:
        * :func:`calc_srtm_tac`
        * :func:`fit_srtm2_tac`
        * :func:`fit_srtm_to_tac_with_bounds`

    """

    def _fitting_srtm(tac_times, r1, bp):
        return calc_srtm_tac(tac_times=tac_times, ref_tac_vals=ref_tac_vals, r1=r1, k2=k2_prime, bp=bp)

    st_values = (r1_bounds[0], bp_bounds[0])
    lo_values = (r1_bounds[1], bp_bounds[1])
    hi_values = (r1_bounds[2], bp_bounds[2])

    return sp_fit(f=_fitting_srtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=st_values, bounds=[lo_values, hi_values])


def fit_frtm_to_tac(tgt_tac_vals: np.ndarray,
                    ref_tac_times: np.ndarray,
                    ref_tac_vals: np.ndarray,
                    r1_start: float = 0.5,
                    k2_start: float = 0.5,
                    k3_start: float = 0.5,
                    k4_start: float = 0.5) -> tuple:
    r"""
    Fit FRTM to the provided target Time Activity Curve (TAC), given the reference TAC, times, and starting guesses for
    the kinetic parameters.

    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.

    This is a simple wrapper around :func:`scipy.optimize.curve_fit` and does not use any bounds for the different
    parameters.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the SRTM.
        ref_tac_times (np.ndarray): Reference TAC values.
        ref_tac_vals (np.ndarray): Reference (and Target) TAC times.
        r1_start (float): Starting guess for the :math:`R_1\equiv\frac{k_1^\prime}{k_1}` parameter.
        k2_start (float): Starting guess for :math:`k_2` parameter.
        k3_start (float): Starting guess for :math:`k_3` parameter.
        k4_start (float): Starting guess for :math:`k_4` parameter.

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``). Output from :func:`scipy.optimize.curve_fit`

    Raises:
        AssertionError: If the reference TAC and times are different dimensions.

    See Also:
        * :func:`calc_frtm_tac`
        
    """
    def _fitting_frtm(tac_times, r1, k2, k3, k4):
        return calc_frtm_tac(tac_times=tac_times, ref_tac_vals=ref_tac_vals, r1=r1, k2=k2, k3=k3, k4=k4)

    starting_values = (r1_start, k2_start, k3_start, k4_start)
    return sp_fit(f=_fitting_frtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=starting_values)


def fit_frtm2_to_tac(tgt_tac_vals: np.ndarray,
                    ref_tac_times: np.ndarray,
                    ref_tac_vals: np.ndarray,
                    k2_prime: float = 0.5,
                    r1_start: float = 0.5,
                    k3_start: float = 0.5,
                    k4_start: float = 0.5) -> tuple:
    r"""
    Fit FRTM2 to the provided target Time Activity Curve (TAC), given the reference TAC, times, and starting guesses for
    the kinetic parameters.

    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.

    This is a simple wrapper around :func:`scipy.optimize.curve_fit` and does not use any bounds for the different
    parameters.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the SRTM.
        ref_tac_times (np.ndarray): Reference TAC values.
        ref_tac_vals (np.ndarray): Reference (and Target) TAC times.
        k2_prime (float): Value for the :math:`k_2^\prime` parameter. Defaults to 0.5.
        r1_start (float): Starting guess for the :math:`R_1\equiv\frac{k_1^\prime}{k_1}` parameter.
        k3_start (float): Starting guess for :math:`k_3` parameter.
        k4_start (float): Starting guess for :math:`k_4` parameter.

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``). Output from :func:`scipy.optimize.curve_fit`

    Raises:
        AssertionError: If the reference TAC and times are different dimensions.

    See Also:
        * :func:`calc_frtm_tac`

    """

    def _fitting_frtm(tac_times, r1, k3, k4):
        return calc_frtm_tac(tac_times=tac_times, ref_tac_vals=ref_tac_vals, r1=r1, k2=k2_prime, k3=k3, k4=k4)

    starting_values = (r1_start, k3_start, k4_start)
    return sp_fit(f=_fitting_frtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=starting_values)


def fit_frtm_to_tac_with_bounds(tgt_tac_vals: np.ndarray,
                                ref_tac_times: np.ndarray,
                                ref_tac_vals: np.ndarray,
                                r1_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                k2_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                k3_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                k4_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0])) -> tuple:
    r"""
    Fit FRTM to the provided target Time Activity Curve (TAC), given the reference TAC, times, and bounds for
    the kinetic parameters.

    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.

    This function is a wrapper around `scipy.optimize.curve_fit` and uses parameter bounds during optimization. The
    bounds for each parameter are formatted as: ``(starting_value, lo_bound, hi_bound)``.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the SRTM.
        ref_tac_times (np.ndarray): Times of the reference TAC data.
        ref_tac_vals (np.ndarray): Reference TAC values.
        r1_bounds (np.ndarray): The bounds for the :math:`R_1\equiv\frac{k_1^\prime}{k_1}` parameter.
        Defaults to [0.5, 0.0, 10.0].
        k2_bounds (np.ndarray): The bounds for :math:`k_2` parameter. Defaults to [0.5, 0.0, 10.0].
        k3_bounds (np.ndarray): The bounds for :math:`k_3` parameter. Defaults to [0.5, 0.0, 10.0].
        k4_bounds (np.ndarray): The bounds for :math:`k_4` parameter. Defaults to [0.5, 0.0, 10.0].

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``). Output from `scipy.optimize.curve_fit`.

    Raises:
        AssertionError: If the target TAC and times are different dimensions.
        
    See Also:
        * :func:`calc_frtm_tac`

    """
    def _fitting_frtm(tac_times, r1, k2, k3, k4):
        return calc_frtm_tac(tac_times=tac_times, ref_tac_vals=ref_tac_vals, r1=r1, k2=k2, k3=k3, k4=k4)

    st_values = (r1_bounds[0], k2_bounds[0], k3_bounds[0], k4_bounds[0])
    lo_values = (r1_bounds[1], k2_bounds[1], k3_bounds[1], k4_bounds[1])
    hi_values = (r1_bounds[2], k2_bounds[2], k3_bounds[2], k4_bounds[2])

    return sp_fit(f=_fitting_frtm, xdata=ref_tac_times, ydata=tgt_tac_vals,
                  p0=st_values, bounds=[lo_values, hi_values])


def fit_frtm2_to_tac_with_bounds(tgt_tac_vals: np.ndarray,
                                 ref_tac_times: np.ndarray,
                                 ref_tac_vals: np.ndarray,
                                 k2_prime: float = 0.5,
                                 r1_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                 k3_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                 k4_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0])) -> tuple:
    r"""
    Fit FRTM2 to the provided target Time Activity Curve (TAC), given the reference TAC, times, and bounds for
    the kinetic parameters.

    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.

    This function is a wrapper around `scipy.optimize.curve_fit` and uses parameter bounds during optimization. The
    bounds for each parameter are formatted as: ``(starting_value, lo_bound, hi_bound)``.

    Args:
        tgt_tac_vals (np.ndarray): Target TAC to fit with the SRTM.
        ref_tac_times (np.ndarray): Times of the reference TAC data.
        ref_tac_vals (np.ndarray): Reference TAC values.
        k2_prime (float): The value for the :math:`k_2^\prime` parameter. Defaults to 0.5.
        r1_bounds (np.ndarray): The bounds for the :math:`R_1\equiv\frac{k_1^\prime}{k_1}` parameter.
        Defaults to [0.5, 0.0, 10.0].
        k3_bounds (np.ndarray): The bounds for :math:`k_3` parameter. Defaults to [0.5, 0.0, 10.0].
        k4_bounds (np.ndarray): The bounds for :math:`k_4` parameter. Defaults to [0.5, 0.0, 10.0].

    Returns:
        tuple: (``fit_parameters``, ``fit_covariance``). Output from `scipy.optimize.curve_fit`.

    Raises:
        AssertionError: If the target TAC and times are different dimensions.

    See Also:
        * :func:`calc_frtm_tac`
        * :func:`fit_frtm2_to_tac`

    """

    def _fitting_frtm(tac_times, r1, k3, k4):
        return calc_frtm_tac(tac_times=tac_times, ref_tac_vals=ref_tac_vals, r1=r1, k2=k2_prime, k3=k3, k4=k4)

    st_values = (r1_bounds[0], k3_bounds[0], k4_bounds[0])
    lo_values = (r1_bounds[1], k3_bounds[1], k4_bounds[1])
    hi_values = (r1_bounds[2], k3_bounds[2], k4_bounds[2])

    return sp_fit(f=_fitting_frtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=st_values, bounds=[lo_values, hi_values])

@numba.njit(fastmath=True)
def fit_mrtm_original_to_tac(tgt_tac_vals: np.ndarray,
                             ref_tac_times: np.ndarray,
                             ref_tac_vals: np.ndarray,
                             t_thresh_in_mins: float):
    r"""
    Fit the original (1996) Multilinear Reference Tissue Model (MRTM) to the provided target Time Activity Curve (TAC)
    values given the reference TAC, times, and threshold time (in minutes). The data are fit for all values beyond the
    threshold. We assume that the target TAC and reference TAC are sampled at the same times.

    .. important::
        This function assumes that both TACs are sampled at the same time, and that the time is in minutes.


    We have the following multilinear regression:
    
    .. math::
    
        \frac{\int_{0}^{T}C(t)\mathrm{d}t}{C(T)}=\frac{V}{V^{\prime}} \frac{\int_{0}^{T}C^{\prime}(t)\mathrm{d}t}{C(T)}
        - \frac{V}{V^{\prime}k_{2}^{\prime}} \frac{C^{\prime}(T)}{C(T)} + b


    Args:
        tgt_tac_vals (np.ndarray): Target TAC values to fit the MRTM.
        ref_tac_times (np.ndarray): Times of the reference TAC (in minutes).
        ref_tac_vals (np.ndarray): Reference TAC values.
        t_thresh_in_mins (float): Threshold time in minutes.

    Returns:
        np.ndarray: Array containing fit results. (:math:`\frac{V}{V^{\prime}}`,
        :math:`\frac{V}{V^{\prime}k_{2}^{\prime}}`, :math:`b`)

    Note:
        This function is implemented with numba for improved performance.
        
    """

    non_zero_indices = np.argwhere(tgt_tac_vals != 0.).T[0]

    if len(non_zero_indices) <= 2:
        return np.asarray([np.nan, np.nan, np.nan])

    t_thresh = get_index_from_threshold(times_in_minutes=ref_tac_times[non_zero_indices],
                                        t_thresh_in_minutes=t_thresh_in_mins)

    if len(ref_tac_times[non_zero_indices][t_thresh:]) <= 2:
        return np.asarray([np.nan, np.nan, np.nan])

    y = cum_trapz(xdata=ref_tac_times, ydata=tgt_tac_vals, initial=0.0)
    y = y[non_zero_indices] / tgt_tac_vals[non_zero_indices]

    x1 = cum_trapz(xdata=ref_tac_times, ydata=ref_tac_vals, initial=0.0)
    x1 = x1[non_zero_indices] / tgt_tac_vals[non_zero_indices]

    x2 = ref_tac_vals[non_zero_indices] / tgt_tac_vals[non_zero_indices]

    x_matrix = np.ones((len(y), 3), float)
    x_matrix[:, 0] = x1[:]
    x_matrix[:, 1] = x2[:]

    fit_ans = np.linalg.lstsq(x_matrix[t_thresh:], y[t_thresh:])[0]
    return fit_ans


@numba.njit(fastmath=True)
def fit_mrtm_2003_to_tac(tgt_tac_vals: np.ndarray,
                         ref_tac_times: np.ndarray,
                         ref_tac_vals: np.ndarray,
                         t_thresh_in_mins: float):
    r"""
    Fit the 2003 Multilinear Reference Tissue Model (MRTM) to the provided target Time Activity Curve (TAC) values given
    the reference TAC, times, and threshold time (in minutes). The data are fit for all values beyond the threshold. We
    assume that the target TAC and reference TAC are sampled at the same times.

    .. important::
        This function assumes that both TACs are sampled at the same time, and that the time is in minutes.

    We have the following multilinear regression:

    .. math::

        C(T)=-\frac{V}{V^{\prime}b} \int_{0}^{T}C^{\prime}(t)\mathrm{d}t + \frac{1}{b} \int_{0}^{T}C(t)\mathrm{d}t
        - \frac{V}{V^{\prime}k_{2}^{\prime}b}C^{\prime}(T)


    Args:
        tgt_tac_vals (np.ndarray): Target TAC values to fit the MRTM.
        ref_tac_times (np.ndarray): Times of the reference TAC (in minutes).
        ref_tac_vals (np.ndarray): Reference TAC values.
        t_thresh_in_mins (float): Threshold time in minutes.

    Returns:
        np.ndarray: Array containing fit results. (:math:`-\frac{V}{V^{\prime}b}`,
        :math:`\frac{1}{b}`, :math:`-\frac{V}{V^{\prime}k_{2}^{\prime}b}`)

    Note:
        This function is implemented with numba for improved performance.

    """

    t_thresh = get_index_from_threshold(times_in_minutes=ref_tac_times, t_thresh_in_minutes=t_thresh_in_mins)
    if t_thresh == -1:
        return np.asarray([np.nan, np.nan, np.nan])

    y = tgt_tac_vals
    x_matrix = np.ones((len(y), 3), float)
    x_matrix[:, 0] = cum_trapz(xdata=ref_tac_times, ydata=ref_tac_vals, initial=0.0)
    x_matrix[:, 1] = cum_trapz(xdata=ref_tac_times, ydata=tgt_tac_vals, initial=0.0)
    x_matrix[:, 2] = ref_tac_vals

    fit_ans = np.linalg.lstsq(x_matrix[t_thresh:], y[t_thresh:])[0]
    return fit_ans


@numba.njit(fastmath=True)
def fit_mrtm2_2003_to_tac(tgt_tac_vals: np.ndarray,
                          ref_tac_times: np.ndarray,
                          ref_tac_vals: np.ndarray,
                          t_thresh_in_mins: float,
                          k2_prime: float):
    r"""
    Fit the second version of Multilinear Reference Tissue Model (MRTM2) to the provided target Time Activity Curve
    (TAC) values given the reference TAC, times, threshold time (in minutes), and k2_prime. The data are fit for all
    values beyond the threshold. We assume that the target TAC and reference TAC are sampled at the same times.
    
    .. important::
        This function assumes that both TACs are sampled at the same time, and that the time is in minutes.

    We have the following multilinear regression:

    .. math::

        C(T) = -\frac{V}{V^{\prime}b}\left(\int_{0}^{T}C^{\prime}(t)\mathrm{d}t -\frac{1}{k_{2}^{\prime}}C^{\prime}(T) \right)
        + \frac{1}{b} \int_{0}^{T}C(t)\mathrm{d}t


    Args:
        tgt_tac_vals (np.ndarray): Target TAC values to fit the MRTM2.
        ref_tac_times (np.ndarray): Times of the reference TAC (in minutes).
        ref_tac_vals (np.ndarray): Reference TAC values.
        t_thresh_in_mins (float): Threshold time in minutes.
        k2_prime (float): Kinetic parameter: washout rate for the reference region.

    Returns:
        np.ndarray: Array containing fit results. (:math:`-\frac{V}{V^{\prime}b}`, :math:`\frac{1}{b}`)

    Note:
        This function is implemented with numba for improved performance.
        
    """

    t_thresh = get_index_from_threshold(times_in_minutes=ref_tac_times, t_thresh_in_minutes=t_thresh_in_mins)
    if t_thresh == -1:
        return np.asarray([np.nan, np.nan])

    x1 = cum_trapz(xdata=ref_tac_times, ydata=ref_tac_vals, initial=0.0)
    x1 += ref_tac_vals / k2_prime
    x2 = cum_trapz(xdata=ref_tac_times, ydata=tgt_tac_vals, initial=0.0)

    y = tgt_tac_vals
    x_matrix = np.ones((len(y), 2), float)
    x_matrix[:, 0] = x1[:]
    x_matrix[:, 1] = x2[:]

    fit_ans = np.linalg.lstsq(x_matrix[t_thresh:], y[t_thresh:])[0]
    return fit_ans


def calc_BP_from_mrtm_original_fit(fit_vals: np.ndarray) -> float:
    r"""
    Given the original MRTM (`Ichise et al., 1996`) fit values, we calculate the binding potential (BP).
    
    The binding potential (BP) is defined as:
    
    .. math::
    
        \mathrm{BP} = \beta_0 - 1
        
    where :math:`\beta_0` is the first fit coefficient.
    
    
    Args:
        fit_vals (np.ndarray): The multilinear regression fit values for the original MRTM.
            Output of :func:`fit_mrtm_original_to_tac`.

    Returns:
        float: Binding potential (BP) value.
        
    See Also:
        :func:`fit_mrtm_original_to_tac` where the order of the regression coefficients is laid out.
        
    """
    return fit_vals[0] - 1.0


def calc_BP_from_mrtm_2003_fit(fit_vals: np.ndarray) -> float:
    r"""
    Given the 2003 MRTM (`Ichise et al., 1996`) fit values, we calculate the binding potential (BP).

    The binding potential (BP) is calculated as:

    .. math::

        \mathrm{BP} = -\left(\frac{\beta_0}{\beta_1} + 1\right)

    where :math:`\beta_0` is the first fit coefficient, and :math:`\beta_1` is the second fit coefficient.

    Args:
        fit_vals (np.ndarray): The multilinear regression fit values for the 2003 MRTM.
            Output of :func:`fit_mrtm_2003_to_tac`.

    Returns:
        float: Binding potential (BP) value.

    See Also:
        :func:`fit_mrtm_2003_to_tac` where the order of the regression coefficients is laid out.

    """
    return (-fit_vals[0]/fit_vals[1] + 1.0)


def calc_BP_from_mrtm2_2003_fit(fit_vals: np.ndarray) -> float:
    r"""
    Given the 2003 MRTM2 (`Ichise et al., 1996`) fit values, we calculate the binding potential (BP).

    The binding potential (BP) is calculated as:

    .. math::

        \mathrm{BP} = -\left(\frac{\beta_0}{\beta_1} + 1\right)

    where :math:`\beta_0` is the first fit coefficient, and :math:`\beta_1` is the second fit coefficient.

    Args:
        fit_vals (np.ndarray): The multilinear regression fit values for the original MRTM.
            Output of :func:`fit_mrtm2_2003_to_tac`.

    Returns:
        float: Binding potential (BP) value.

    See Also:
        :func:`fit_mrtm2_2003_to_tac` where the order of the regression coefficients is laid out.

    """
    return -(fit_vals[0]/fit_vals[1] + 1.0)


def calc_k2prime_from_mrtm_original_fit(fit_vals: np.ndarray):
    r"""
    Given the original MRTM (`Ichise et al., 1996`) fit values, we calculate :math:`k_{2}^{\prime}`.

    The :math:`k_{2}^{\prime}` is calculated as:

    .. math::

         k_{2}^{\prime}= \frac{\beta_{0}}{\beta_{1}}

    where :math:`\beta_0` is the first fit coefficient and :math:`\beta_1` is the second fit coefficient.


    Args:
        fit_vals (np.ndarray): The multilinear regression fit values for the original MRTM.
            Output of :func:`fit_mrtm_original_to_tac`.

    Returns:
        float: :math:`k_2^\prime` value.

    See Also:
        :func:`fit_mrtm_original_to_tac` where the order of the regression coefficients is laid out.

    """
    return fit_vals[0]/fit_vals[1]


def calc_k2prime_from_mrtm_2003_fit(fit_vals: np.ndarray):
    r"""
    Given the 2003 MRTM (`Ichise et al., 2003`) fit values, we calculate :math:`k_{2}^{\prime}`.

    The :math:`k_{2}^{\prime}` is calculated as:

    .. math::

         k_{2}^{\prime}= \frac{\beta_{0}}{\beta_{2}}

    where :math:`\beta_0` is the first fit coefficient and :math:`\beta_2` is the third fit coefficient.


    Args:
        fit_vals (np.ndarray): The multilinear regression fit values for the original MRTM.
            Output of :func:`fit_mrtm_2003_to_tac`.

    Returns:
        float: :math:`k_2^\prime` value.

    See Also:
        :func:`fit_mrtm_2003_to_tac` where the order of the regression coefficients is laid out.

    """
    return fit_vals[0]/fit_vals[-1]


class FitTACWithRTMs:
    r"""
    A class used to fit a kinetic model to both a target and a reference Time Activity Curve (TAC).

    The :class:`FitTACWithRTMs` class simplifies the process of kinetic model fitting by providing methods for validating
    input data, choosing a model to fit, and then performing the fit. It takes in raw intensity values of TAC for both
    target and reference regions as inputs, which are then used in curve fitting.

    This class supports various kinetic models, including but not limited to: the simplified and full reference tissue
    models (SRTM & FRTM), and the multilinear reference tissue models (Orignial MRMT, MRTM & MRTM2). Each model type '
    can be bounded or unbounded.

    The fitting result contains the estimated kinetic parameters depending on the chosen model.

    Attributes:
        target_tac_vals (np.ndarray): The target TAC values.
        reference_tac_times (np.ndarray): The time points of the reference TAC.
        reference_tac_vals (np.ndarray): The reference TAC values.
        method (str): Optional. The kinetic model to use. Defaults to 'mrtm'.
        bounds (np.ndarray): Optional. Parameter bounds for the specified kinetic model. Defaults to None.
        t_thresh_in_mins (float): Optional. The times at which the reference TAC was sampled. Defaults to None.
        k2_prime (float): Optional. The estimated efflux rate constant for the non-displaceable compartment. Defaults to
            None.
        fit_results (np.ndarray): The result of the fit.

    Example:
        The following example shows how to use the :class:`FitTACWithRTMs` class to fit the SRTM to a target and reference
        TAC.

        .. code-block:: python

            import numpy as np
            import petpal.kinetic_modeling.tcms_as_convolutions as pet_tcm
            import petpal.kinetic_modeling.reference_tissue_models as pet_rtms
            
            # loading the input tac to generate a reference region tac
            input_tac_times, input_tac_vals = np.asarray(np.loadtxt("../../data/tcm_tacs/fdg_plasma_clamp_evenly_resampled.txt").T,
                                                         float)
            
            # generating a reference region tac
            ref_tac_times, ref_tac_vals = pet_tcm.generate_tac_1tcm_c1_from_tac(tac_times=input_tac_times, tac_vals=input_tac_vals,
                                                                                k1=1.0, k2=0.2)
            
            # generating an SRTM tac
            srtm_tac_vals = pet_rtms.calc_srtm_tac(tac_times=ref_tac_times, ref_tac_vals=ref_tac_vals, r1=1.0, k2=0.25, bp=3.0)
            
            rtm_analysis = pet_rtms.FitTACWithRTMs(target_tac_vals=srtm_tac_vals,
                                                reference_tac_times=ref_tac_times,
                                                reference_tac_vals=ref_tac_vals,
                                                method='srtm')
            
            # performing the fit
            rtm_analysis.fit_tac_to_model()
            fit_results = rtm_analysis.fit_results[1]


    This will give you the kinetic parameter values of the SRTM for the provided TACs.

    See Also:
        * :meth:`validate_bounds`
        * :meth:`validate_method_inputs`
        * :meth:`fit_tac_to_model`
        
    """
    def __init__(self,
                 target_tac_vals: np.ndarray,
                 reference_tac_times: np.ndarray,
                 reference_tac_vals: np.ndarray,
                 method: str = 'mrtm',
                 bounds: Union[None, np.ndarray] = None,
                 t_thresh_in_mins: float = None,
                 k2_prime: float = None):
        r"""
        Initialize the FitTACWithRTMs object with specified parameters.

        This method sets up input parameters and validates them. We check if the bounds are correct for the given
        'method', and we make sure that any fitting threshold are defined for the MRTM analyses.
        

        Args:
            target_tac_vals (np.ndarray): The array representing the target TAC values.
            reference_tac_times (np.ndarray): The array representing time points associated with the reference TAC.
            reference_tac_vals (np.ndarray): The array representing values of the reference TAC.
            method (str, optional): The kinetics method to be used. Default is 'mrtm'.
            bounds (Union[None, np.ndarray], optional): Bounds for kinetic parameters used in optimization. None
                represents absence of bounds. Default is None.
            t_thresh_in_mins (float, optional): Threshold for time separation in minutes. Default is None.
            k2_prime (float, optional): The estimated rate constant related to the flush-out rate of the reference
                compartment. Default is None.

        Raises:
            ValueError: If a parameter necessary for chosen method is not provided.
            AssertionError: If rate constant k2_prime is non-positive.
        """
        
        self.target_tac_vals: np.ndarray = target_tac_vals
        self.reference_tac_times: np.ndarray = reference_tac_times
        self.reference_tac_vals: np.ndarray = reference_tac_vals
        self.method: str = method.lower()
        self.bounds: Union[None, np.ndarray] = bounds
        self.validate_bounds()
        
        self.t_thresh_in_mins: float = t_thresh_in_mins
        self.k2_prime: float = k2_prime
        
        self.validate_method_inputs()
        
        self.fit_results: Union[None, np.ndarray] = None
    
    def validate_method_inputs(self):
        r"""Validates the inputs for different methods

        This method validates the inputs depending on the chosen method in the object.

        - If the method is of type 'mrtm', it checks if `t_thresh_in_mins` is defined and positive.
        - If the method ends with a '2' (the reduced/modified methods), it checks if `k2_prime` is defined and positive.

        Raises:
            ValueError: If ``t_thresh_in_mins`` is not defined while the method starts with 'mrtm'.
            AssertionError: If ``t_thresh_in_mins`` is not a positive number.
            ValueError: If ``k2_prime`` is not defined while the method ends with '2'.
            AssertionError: If ``k2_prime`` is not a positive number.
        
        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`
        
        """
        if self.method.startswith("mrtm"):
            if self.t_thresh_in_mins is None:
                raise ValueError("t_t_thresh_in_mins must be defined if method is 'mrtm'")
            else:
                assert self.t_thresh_in_mins >= 0, "t_thresh_in_mins must be a positive number."
        if self.method.endswith("2"):
            if self.k2_prime is None:
                raise ValueError("k2_prime must be defined if we are using the reduced models: FRTM2, SRTM2, "
                                 "and MRTM2.")
            assert self.k2_prime >= 0,"k2_prime must be a positive number."
    
    def validate_bounds(self):
        r"""Validates the bounds for different methods

        This method validates the shape of the bounds depending on the chosen method in the object.

        - If the method is 'srtm', it checks that bounds shape is (3, 3).
        - If the method is 'frtm', it checks that bounds shape is (4, 3).

        Raises:
            AssertionError: If the bounds shape for method 'srtm' is not (3, 3)
            AssertionError: If the bounds shape for method 'frtm' is not (4, 3).
            ValueError: If the method is not 'srtm' or 'frtm' while providing bounds.
            
        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`
            
        """
        if self.bounds is not None:
            num_params, num_vals = self.bounds.shape
            if self.method == "srtm":
                assert num_params == 3 and num_vals == 3, ("The bounds have the wrong shape. Bounds must "
                                                           "be (start, lo, hi) for each of the fitting "
                                                           "parameters: r1, k2, bp")
            elif self.method == "frtm":
                assert num_params == 4 and num_vals == 3, (
                    "The bounds have the wrong shape. Bounds must be (start, lo, hi) "
                    "for each of the fitting parameters: r1, k2, k3, k4")
                
            elif self.method == "srtm2":
                assert num_params == 2 and num_vals == 3, ("The bounds have the wrong shape. Bounds must "
                                                           "be (start, lo, hi) for each of the fitting "
                                                           "parameters: r1, bp")
            elif self.method == "frtm2":
                assert num_params == 3 and num_vals == 3, (
                    "The bounds have the wrong shape. Bounds must be (start, lo, hi) "
                    "for each of the fitting parameters: r1, k3, k4")
            else:
                raise ValueError(f"Invalid method! Must be either 'srtm', 'frtm', 'srtm2' or 'frtm2' if bounds are "
                                 f"provided. Got {self.method}.")
    
    def fit_tac_to_model(self):
        r"""Fits TAC vals to model

        This method fits the target TAC values to the model depending on the chosen method in the object.

        - If the method is 'srtm' or 'frtm', and bounds are provided, fitting functions with bounds are used.
        - If the method is 'srtm' or 'frtm', and bounds are not provided, fitting functions without bounds are used.
        - If the method is 'mrtm-original', 'mrtm' or 'mrtm2', related fitting methods are utilized.

        Raises:
            ValueError: If the method name is invalid and not one of 'srtm', 'frtm', 'mrtm-original', 'mrtm' or 'mrtm2'.
            
            
        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_srtm2_to_tac_with_bounds`
            * :func:`fit_srtm2_to_tac`
            * :func:`fit_frtm2_to_tac_with_bounds`
            * :func:`fit_frtm2_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`
            
        """
        if self.method == "srtm":
            if self.bounds is not None:
                self.fit_results = fit_srtm_to_tac_with_bounds(tgt_tac_vals=self.target_tac_vals,
                                                               ref_tac_times=self.reference_tac_times,
                                                               ref_tac_vals=self.reference_tac_vals,
                                                               r1_bounds=self.bounds[0],
                                                               k2_bounds=self.bounds[1],
                                                               bp_bounds=self.bounds[2])
            else:
                self.fit_results = fit_srtm_to_tac(tgt_tac_vals=self.target_tac_vals,
                                                   ref_tac_times=self.reference_tac_times,
                                                   ref_tac_vals=self.reference_tac_vals)

        elif self.method == "srtm2":
            if self.bounds is not None:
                self.fit_results = fit_srtm2_to_tac_with_bounds(tgt_tac_vals=self.target_tac_vals,
                                                                ref_tac_times=self.reference_tac_times,
                                                                ref_tac_vals=self.reference_tac_vals,
                                                                k2_prime=self.k2_prime,
                                                                r1_bounds=self.bounds[0],
                                                                bp_bounds=self.bounds[1])
            else:
                self.fit_results = fit_srtm2_to_tac(tgt_tac_vals=self.target_tac_vals,
                                                    ref_tac_times=self.reference_tac_times,
                                                    ref_tac_vals=self.reference_tac_vals,
                                                    k2_prime=self.k2_prime)
        elif self.method == "frtm":
            if self.bounds is not None:
                self.fit_results = fit_frtm_to_tac_with_bounds(tgt_tac_vals=self.target_tac_vals,
                                                               ref_tac_times=self.reference_tac_times,
                                                               ref_tac_vals=self.reference_tac_vals,
                                                               r1_bounds=self.bounds[0],
                                                               k2_bounds=self.bounds[1],
                                                               k3_bounds=self.bounds[2],
                                                               k4_bounds=self.bounds[3])
            else:
                self.fit_results = fit_frtm_to_tac(tgt_tac_vals=self.target_tac_vals,
                                                   ref_tac_times=self.reference_tac_times,
                                                   ref_tac_vals=self.reference_tac_vals)

        elif self.method == "frtm2":
            if self.bounds is not None:
                self.fit_results = fit_frtm2_to_tac_with_bounds(tgt_tac_vals=self.target_tac_vals,
                                                                ref_tac_times=self.reference_tac_times,
                                                                ref_tac_vals=self.reference_tac_vals,
                                                                k2_prime=self.k2_prime,
                                                                r1_bounds=self.bounds[0],
                                                                k3_bounds=self.bounds[1],
                                                                k4_bounds=self.bounds[2])
            else:
                self.fit_results = fit_frtm2_to_tac(tgt_tac_vals=self.target_tac_vals,
                                                    ref_tac_times=self.reference_tac_times,
                                                    ref_tac_vals=self.reference_tac_vals,
                                                    k2_prime=self.k2_prime)

        elif self.method == "mrtm-original":
            self.fit_results = fit_mrtm_original_to_tac(tgt_tac_vals=self.target_tac_vals,
                                                        ref_tac_times=self.reference_tac_times,
                                                        ref_tac_vals=self.reference_tac_vals,
                                                        t_thresh_in_mins=self.t_thresh_in_mins)

        elif self.method == "mrtm":
            self.fit_results = fit_mrtm_2003_to_tac(tgt_tac_vals=self.target_tac_vals,
                                                    ref_tac_times=self.reference_tac_times,
                                                    ref_tac_vals=self.reference_tac_vals,
                                                    t_thresh_in_mins=self.t_thresh_in_mins)

        elif self.method == "mrtm2":
            self.fit_results = fit_mrtm2_2003_to_tac(tgt_tac_vals=self.target_tac_vals,
                                                     ref_tac_times=self.reference_tac_times,
                                                     ref_tac_vals=self.reference_tac_vals,
                                                     t_thresh_in_mins=self.t_thresh_in_mins,
                                                     k2_prime=self.k2_prime)
        else:
            raise ValueError(f"Invalid method! Must be either 'srtm', 'frtm', 'mrtm-original', 'mrtm' or 'mrtm2'. Got {self.method}.")


class RTMAnalysis:
    r"""
    A class designed to carry out various Reference Tissue Model (RTM) analyses on Time Activity Curves (TACs).

    This class eases the process of conducting RTM analysis on TACs. Paths to both reference and region-of-interest
    (ROI) TACs are taken as inputs at initialization. The class further provides multiple utility functions for
    initializing and running the RTM analysis, and also for validating the inputs based on the RTM method chosen.

    This class currently supports various RTM methods such as :'srtm', 'frtm', 'mrtm-original', 'mrtm', and 'mrtm2'.

    Attributes:
        ref_tac_path (str): Absolute path for reference TAC
        roi_tac_path (str): Absolute path for ROI TAC
        output_directory (str): Absolute path for the output directory
        output_filename_prefix (str): Prefix for the output filename of the result
        method (str): RTM analysis method. Converts to lower case at initialization.
        analysis_props (dict): Analysis properties dictionary initialized with method-specific property keys and
            default values.
        _has_analysis_been_run (bool): Flag representing if the RTM analysis has been run to ensure correct order of
            operations.

    Example:
        In the proceeding example, we assume that we have two tacs: a reference region tac, and a region of interest
        (ROI) tac named 'ref_tac.txt' and 'roi_tac.txt', respectively. Furthermore, we assume that both TACs are sampled
        at the same times, and are evenly sampled with respect to time.
        
        .. code-block:: python
            
            import numpy as np
            from petpal.kinetic_modeling.reference_tissue_models as pet_rtms
            
            file_rtm = pet_rtms.RTMAnalysis(ref_tac_path="ref_tac.txt",
                                            roi_tac_path="roi_tac.txt",
                                            output_directory="./",
                                            output_filename_prefix='pre',
                                            method="mrtm")
            file_rtm.run_analysis(t_thresh_in_mins=40.0)
            file_rtm.save_analysis()
        

    See Also:
        * :class:`FitTACWithRTMs`: a class for analyzing TACs with RTMs.

    """
    def __init__(self,
                 ref_tac_path: str,
                 roi_tac_path: str,
                 output_directory: str,
                 output_filename_prefix: str,
                 method: str):
        r"""
        Initialize RTMAnalysis with provided arguments.

        The init method executes the following operations:
            1. It converts the provided analysis method to lower case for consistency in internal processing.
            2. It obtains the absolute paths for reference and ROI TAC files and the output directory, to ensure
               they are consistently accessible.
            3. It initializes the analysis properties dictionary using `init_analysis_props` method.
            4. It initializes the `_has_analysis_been_run` flag to False, to indicate that the RTM analysis has not yet been run.

        Args:
            ref_tac_path (str): Path to the file containing reference TAC.
            roi_tac_path (str): Path to the file containing ROI TAC.
            output_directory (str): Path to the directory where the output will be saved.
            output_filename_prefix (str): Prefix that will be used for the output filename.
            method (str): The RTM analysis method to be used. Could be one of 'srtm', 'frtm', 'mrtm-original',
                'mrtm' or 'mrtm2'.
                
        """
        self.ref_tac_path: str = os.path.abspath(ref_tac_path)
        self.roi_tac_path: str = os.path.abspath(roi_tac_path)
        self.output_directory: str = os.path.abspath(output_directory)
        self.output_filename_prefix: str = output_filename_prefix
        self.method = method.lower()
        self.analysis_props: dict = self.init_analysis_props(self.method)
        self._has_analysis_been_run: bool = False

    def init_analysis_props(self, method: str) -> dict:
        r"""
        Initializes the analysis properties dict based on the specified RTM analysis method.

        Args:
            method (str): RTM analysis method. Must be one of 'srtm', 'frtm', 'mrtm-original',
                'mrtm' or 'mrtm2'.

        Returns:
            dict: A dictionary containing method-specific property keys and default values.

        Raises:
            ValueError: If input `method` is not one of the supported RTM methods.
        """
        common_props = {'FilePathRTAC': self.ref_tac_path,
                        'FilePathTTAC': self.roi_tac_path,
                        'MethodName': method.upper()}
        if method.startswith("mrtm"):
            props = {
                'BP': None,
                'k2Prime': None,
                'ThresholdTime': None,
                'StartFrameTime': None,
                'EndFrameTime' : None,
                'NumberOfPointsFit': None,
                'RawFits': None,
                **common_props
                }
        elif method.startswith("srtm") or method.startswith("frtm"):
            props = {
                'FitValues': None,
                'FitStdErr': None,
                **common_props
                }
        else:
            raise ValueError(f"Invalid method! Must be either 'srtm', 'frtm', 'srtm2', 'frtm2', "
                             f"'mrtm-original', 'mrtm' or 'mrtm2'. Got {method}.")
        return props

    def run_analysis(self,
                     bounds: Union[None, np.ndarray] = None,
                     t_thresh_in_mins: float = None,
                     k2_prime: float = None,
                     **tac_load_kwargs):
        r"""
        Runs the full RTM analysis process which involves validating inputs, calculation fits, and deducing fit
        properties.

        Specifically, it executes the following sequence:
            1. :meth:`validate_analysis_inputs`
            2. :meth:`calculate_fit`
            3. :meth:`calculate_fit_properties`

        Args:
            bounds (Union[None, np.ndarray], optional): Optional boundaries for parameters for fitting function.
            t_thresh_in_mins (float, optional): Threshold time in minutes for the MRTM analyses.
            k2_prime (float, optional): Input for the modified RTM (MRTM2, FRTM2, and SRTM2) analyses.

        Returns:
            None
        """
        self.validate_analysis_inputs(k2_prime=k2_prime, t_thresh_in_mins=t_thresh_in_mins)

        fit_results = self.calculate_fit(bounds=bounds,
                                         t_thresh_in_mins=t_thresh_in_mins,
                                         k2_prime=k2_prime,
                                         **tac_load_kwargs)
        self.calculate_fit_properties(fit_results=fit_results,
                                      t_thresh_in_mins=t_thresh_in_mins,
                                      k2_prime=k2_prime)
        self._has_analysis_been_run = True

    def validate_analysis_inputs(self, k2_prime, t_thresh_in_mins):
        r"""
        Validates the provided inputs for the RTM analysis.

        If MRTM type of analysis is being run, it ensures that ``t_thresh_in_mins`` is not None.
        If modified analysis is being done (MRTM2, FRTM2, SRTM2), it ensures ``k2_prime`` is not None.

        Args:
            k2_prime (float): k2 prime value.
            t_thresh_in_mins (float): Threshold time for MRTM analyses.

        Raises:
            ValueError: If an input required for the selected method is `None`.
        """
        if self.method.startswith("mrtm") and t_thresh_in_mins is None:
            raise ValueError("t_thresh_in_mins must be set for the MRTM analyses.")
        if self.method.endswith("2") and k2_prime is None:
            raise ValueError("k2_prime must be set for the modified RTM (MRTM2, FRTM2, and SRTM2) analyses.")

    def calculate_fit(self,
                      bounds: Union[None, np.ndarray] = None,
                      t_thresh_in_mins: float = None,
                      k2_prime: float = None,
                      **tac_load_kwargs):
        r"""
        Calculates the model fitting parameters for TACs using the chosen RTM analysis method.

        This method executes the following sequence:
            1. :meth:`validate_analysis_inputs`
            2. :meth:`safe_load_tac` for both reference and ROI TACs
            3. Creates a :class:`FitTACWithRTMs` instance and fits TAC to the model

        Args:
            bounds (Union[None, np.ndarray]): Boundaries for parameters for fitting function.
            t_thresh_in_mins (float): Threshold time for MRTM analyses.
            k2_prime (float): k2 prime value.
            tac_load_kwargs (Any): Additional keyword arguments for the loading TAC function.

        Returns:
            FitResults: Object containing fit results.
        """
        self.validate_analysis_inputs(k2_prime=k2_prime, t_thresh_in_mins=t_thresh_in_mins)

        ref_tac_times, ref_tac_vals = safe_load_tac(filename=self.ref_tac_path, **tac_load_kwargs)
        _tgt_tac_times, tgt_tac_vals = safe_load_tac(filename=self.roi_tac_path, **tac_load_kwargs)
        analysis_obj = FitTACWithRTMs(target_tac_vals=tgt_tac_vals,
                                      reference_tac_times=ref_tac_times,
                                      reference_tac_vals=ref_tac_vals,
                                      method=self.method,
                                      bounds=bounds,
                                      t_thresh_in_mins=t_thresh_in_mins,
                                      k2_prime=k2_prime)
        analysis_obj.fit_tac_to_model()

        return analysis_obj.fit_results

    def calculate_fit_properties(self, fit_results: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
                                 t_thresh_in_mins: float = None,
                                 k2_prime: float = None):
        r"""
        Calculates additional fitting properties based on the raw fit results.

        It delegates the calculation to method-specific functions:
            1. For 'srtm' or 'frtm' methods: :meth:`_calc_frtm_or_srtm_fit_props` is used.
            2. For 'mrtm' methods: :meth:`_calc_mrtm_fit_props` is used.

        Args:
            fit_results (Union[np.ndarray, tuple[np.ndarray, np.ndarray]]): The fit results.
            t_thresh_in_mins (float): Threshold time for MRTM analyses.
            k2_prime (float): k2 prime value for 'mrtm' based methods.

        Returns:
            None
        """
        if self.method.startswith("frtm") or self.method.startswith("srtm"):
            self._calc_frtm_or_srtm_fit_props(fit_results=fit_results, k2_prime=k2_prime)
        else:
            self._calc_mrtm_fit_props(fit_results=fit_results,
                                      k2_prime=k2_prime,
                                      t_thresh_in_mins=t_thresh_in_mins)

    def save_analysis(self):
        r"""
        Save the analysis results in JSON format.

        The results are only saved if the analysis has been run (_has_analysis_been_run flag is checked).

        Raises:
            RuntimeError: If the :meth:'run_analysis' method has not been called yet.
        """
        if not self._has_analysis_been_run:
            raise RuntimeError("'run_analysis' method must be called before 'save_analysis'.")
        file_name_prefix = os.path.join(self.output_directory,
                                        f"{self.output_filename_prefix}_analysis-{self.analysis_props['MethodName']}")
        analysis_props_file = f"{file_name_prefix}_props.json"
        with open(analysis_props_file, 'w',encoding='utf-8') as f:
            json.dump(obj=self.analysis_props, fp=f, indent=4)

    def _calc_mrtm_fit_props(self, fit_results: np.ndarray,
                             k2_prime: float,
                             t_thresh_in_mins: float):
        r"""
        Internal function used to calculate additional fitting properties for 'mrtm' type analyses.

        This method is used internally within :meth:`calculate_fit_properties`.

        Args:
            fit_results (np.ndarray): Resulting fit parameters.
            k2_prime (float): k2 prime value for 'mrtm' based methods.
            t_thresh_in_mins (float): Threshold time for MRTM analyses.
        """
        self.validate_analysis_inputs(k2_prime=k2_prime, t_thresh_in_mins=t_thresh_in_mins)
        if self.method == 'mrtm-original':
            bp_val = calc_BP_from_mrtm_original_fit(fit_results)
            k2_val = calc_k2prime_from_mrtm_original_fit(fit_results)
        elif self.method == 'mrtm':
            bp_val = calc_BP_from_mrtm_2003_fit(fit_results)
            k2_val = calc_k2prime_from_mrtm_2003_fit(fit_results)
        else:
            bp_val = calc_BP_from_mrtm2_2003_fit(fit_results)
            k2_val = None
        self.analysis_props["k2Prime"] = k2_val.round(5)
        self.analysis_props["BP"] = bp_val.round(5)
        self.analysis_props["RawFits"] = list(fit_results.round(5))

        ref_tac_times, _ = safe_load_tac(filename=self.ref_tac_path)
        t_thresh_index = get_index_from_threshold(times_in_minutes=ref_tac_times, t_thresh_in_minutes=t_thresh_in_mins)
        self.analysis_props['ThresholdTime'] = t_thresh_in_mins
        self.analysis_props['StartFrameTime'] = ref_tac_times[t_thresh_index]
        self.analysis_props['EndFrameTime'] = ref_tac_times[-1]
        self.analysis_props['NumberOfPointsFit'] = len(ref_tac_times[t_thresh_index:])

    def _calc_frtm_or_srtm_fit_props(self, fit_results: tuple[np.ndarray, np.ndarray], k2_prime: float):
        r"""
        Internal function used to calculate additional fitting properties for 'frtm' and 'srtm' type analyses.

        This method is used internally within :meth:`calculate_fit_properties`.

        Args:
            fit_results (tuple[np.ndarray, np.ndarray]): Tuple containing the fit parameters and their corresponding fit
                covariances.
            
        """
        fit_params, fit_covariances = fit_results
        fit_stderr = np.sqrt(np.diagonal(fit_covariances))

        if self.method.startswith('srtm'):
            format_func =  self._get_pretty_srtm_fit_param_vals
        else:
            format_func = self._get_pretty_frtm_fit_param_vals

        if self.method.endswith('2'):
            self.analysis_props["k2Prime"] = k2_prime
            self.analysis_props["FitValues"] = format_func(fit_params.round(5), True)
            self.analysis_props["FitStdErr"] = format_func(fit_stderr.round(5), True)
        else:
            self.analysis_props["FitValues"] = format_func(fit_params.round(5), False)
            self.analysis_props["FitStdErr"] = format_func(fit_stderr.round(5), False)

    @staticmethod
    def _get_pretty_srtm_fit_param_vals(param_fits: np.ndarray, reduced: bool = False) -> dict:
        r"""
        Utility function to get nicely formatted fit parameters for 'srtm(2)' analysis.

        Returns a dictionary with keys: 'R1', 'k2', and 'BP' and the corresponding values from ``param_fits``.

        Args:
            param_fits (np.ndarray): array containing the fit parameters.

        Returns:
            dict: Dictionary of fit parameters and their corresponding values.
        """
        if reduced:
            return {name: val for name, val in zip(['R1', 'BP'], param_fits)}
        else:
            return {name: val for name, val in zip(['R1', 'k2', 'BP'], param_fits)}

    @staticmethod
    def _get_pretty_frtm_fit_param_vals(param_fits: np.ndarray, reduced: bool = False) -> dict:
        r"""
        Utility function to get nicely formatted fit parameters for 'frtm(2)' analysis.

        Returns a dictionary with keys: 'R1', 'k2', 'k3', and 'k4' and the corresponding values from ``param_fits``.

        Args:
            param_fits (np.ndarray): array containing the fit parameters.

        Returns:
            dict: Dictionary of fit parameters and their corresponding values.
        """
        if reduced:
            return {name: val for name, val in zip(['R1', 'k3', 'k4'], param_fits)}
        else:
            return {name: val for name, val in zip(['R1', 'k2', 'k3', 'k4'], param_fits)}
