"""
Class for doing RTM analysis
"""
import os
from typing import Union
import json
import numpy as np
from petpal.kinetic_modeling.fit_tac_with_rtms import FitTACWithRTMs
from petpal.kinetic_modeling.graphical_analysis import get_index_from_threshold
from petpal.kinetic_modeling.reference_tissue_models import (calc_k2prime_from_mrtm_2003_fit,
                                                             calc_k2prime_from_mrtm_original_fit,
                                                             calc_bp_from_mrtm2_2003_fit,
                                                             calc_bp_from_mrtm_original_fit,
                                                             calc_bp_from_mrtm_2003_fit)
from petpal.utils.image_io import safe_load_tac


class RTMAnalysis:
    r"""
    A class designed to carry out various Reference Tissue Model (RTM) analyses on Time Activity
    Curves (TACs).

    This class eases the process of conducting RTM analysis on TACs. Paths to both reference and
    region-of-interest (ROI) TACs are taken as inputs at initialization. The class further provides
    multiple utility functions for initializing and running the RTM analysis, and also for
    validating the inputs based on the RTM method chosen.

    This class currently supports various RTM methods such as :'srtm', 'frtm', 'mrtm-original',
    'mrtm', and 'mrtm2'.

    Attributes:
        ref_tac_path (str): Absolute path for reference TAC
        roi_tac_path (str): Absolute path for ROI TAC
        output_directory (str): Absolute path for the output directory
        output_filename_prefix (str): Prefix for the output filename of the result
        method (str): RTM analysis method. Converts to lower case at initialization.
        analysis_props (dict): Analysis properties dictionary initialized with method-specific 
            property keys and default values.
        _has_analysis_been_run (bool): Flag representing if the RTM analysis has been run to ensure
            correct order of operations.

    Example:
        In the proceeding example, we assume that we have two tacs: a reference region tac, and a
        region of interest (ROI) tac named 'ref_tac.txt' and 'roi_tac.txt', respectively.
        Furthermore, we assume that both TACs are sampled at the same times, and are evenly sampled
        with respect to time.

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
            1. It converts the provided analysis method to lower case for consistency in internal
                processing.
            2. It obtains the absolute paths for reference and ROI TAC files and the output
                directory, to ensure they are consistently accessible.
            3. It initializes the analysis properties dictionary using `init_analysis_props` method.
            4. It initializes the `_has_analysis_been_run` flag to False, to indicate that the RTM
                analysis has not yet been run.

        Args:
            ref_tac_path (str): Path to the file containing reference TAC.
            roi_tac_path (str): Path to the file containing ROI TAC.
            output_directory (str): Path to the directory where the output will be saved.
            output_filename_prefix (str): Prefix that will be used for the output filename.
            method (str): The RTM analysis method to be used. Could be one of 'srtm', 'frtm',
                'mrtm-original', 'mrtm' or 'mrtm2'.

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
        Runs the full RTM analysis process which involves validating inputs, calculation fits, and
        deducing fit properties.

        Specifically, it executes the following sequence:
            1. :meth:`validate_analysis_inputs`
            2. :meth:`calculate_fit`
            3. :meth:`calculate_fit_properties`

        Args:
            bounds (Union[None, np.ndarray], optional): Optional boundaries for parameters for
                fitting function.
            t_thresh_in_mins (float, optional): Threshold time in minutes for the MRTM analyses.
            k2_prime (float, optional): Input for the modified RTM (MRTM2, FRTM2, and SRTM2)
                analyses.

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
        If modified analysis is being done (MRTM2, FRTM2, SRTM2), it ensures ``k2_prime`` is not
        None.

        Args:
            k2_prime (float): k2 prime value.
            t_thresh_in_mins (float): Threshold time for MRTM analyses.

        Raises:
            ValueError: If an input required for the selected method is `None`.
        """
        if self.method.startswith("mrtm") and t_thresh_in_mins is None:
            raise ValueError("t_thresh_in_mins must be set for the MRTM analyses.")
        if self.method.endswith("2") and k2_prime is None:
            raise ValueError("k2_prime must be set for the modified RTM (MRTM2, FRTM2, and SRTM2) "
                             "analyses.")

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
        analysis_obj = FitTACWithRTMs(tac_times_in_minutes=ref_tac_times,
                                      target_tac_vals=tgt_tac_vals,
                                      reference_tac_vals=ref_tac_vals,
                                      method=self.method, bounds=bounds,
                                      t_thresh_in_mins=t_thresh_in_mins,
                                      k2_prime=k2_prime)
        analysis_obj.fit_tac_to_model()

        return analysis_obj.fit_results

    def calculate_fit_properties(self,
                                 fit_results: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
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

        The results are only saved if the analysis has been run (_has_analysis_been_run flag is
        checked).

        Raises:
            RuntimeError: If the :meth:'run_analysis' method has not been called yet.
        """
        if not self._has_analysis_been_run:
            raise RuntimeError("'run_analysis' method must be called before 'save_analysis'.")
        file_name_prefix = os.path.join(self.output_directory,
                                        f"{self.output_filename_prefix}_analysis-"
                                        f"{self.analysis_props['MethodName']}")
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
            bp_val = calc_bp_from_mrtm_original_fit(fit_results)
            k2_val = calc_k2prime_from_mrtm_original_fit(fit_results)
        elif self.method == 'mrtm':
            bp_val = calc_bp_from_mrtm_2003_fit(fit_results)
            k2_val = calc_k2prime_from_mrtm_2003_fit(fit_results)
        else:
            bp_val = calc_bp_from_mrtm2_2003_fit(fit_results)
            k2_val = None
        self.analysis_props["k2Prime"] = k2_val.round(5)
        self.analysis_props["BP"] = bp_val.round(5)
        self.analysis_props["RawFits"] = list(fit_results.round(5))

        ref_tac_times, _ = safe_load_tac(filename=self.ref_tac_path)
        t_thresh_index = get_index_from_threshold(times_in_minutes=ref_tac_times,
                                                  t_thresh_in_minutes=t_thresh_in_mins)
        self.analysis_props['ThresholdTime'] = t_thresh_in_mins
        self.analysis_props['StartFrameTime'] = ref_tac_times[t_thresh_index]
        self.analysis_props['EndFrameTime'] = ref_tac_times[-1]
        self.analysis_props['NumberOfPointsFit'] = len(ref_tac_times[t_thresh_index:])

    def _calc_frtm_or_srtm_fit_props(self,
                                     fit_results: tuple[np.ndarray, np.ndarray],
                                     k2_prime: float):
        r"""
        Internal function used to calculate additional fitting properties for 'frtm' and 'srtm'
        type analyses.

        This method is used internally within :meth:`calculate_fit_properties`.

        Args:
            fit_results (tuple[np.ndarray, np.ndarray]): Tuple containing the fit parameters and
                their corresponding fit covariances.

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

        Returns a dictionary with keys: 'R1', 'k2', and 'BP' and the corresponding values from
        ``param_fits``.

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

        Returns a dictionary with keys: 'R1', 'k2', 'k3', and 'k4' and the corresponding values from
        ``param_fits``.

        Args:
            param_fits (np.ndarray): array containing the fit parameters.

        Returns:
            dict: Dictionary of fit parameters and their corresponding values.
        """
        if reduced:
            return {name: val for name, val in zip(['R1', 'k3', 'k4'], param_fits)}
        else:
            return {name: val for name, val in zip(['R1', 'k2', 'k3', 'k4'], param_fits)}
