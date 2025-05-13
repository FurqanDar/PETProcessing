"""
This module implements PCA (Principal Component Analysis)-guided methods for generating
Image-Derived Input Functions (IDIF) from 4D-PET data. IDIFs are used in PET
(Positron Emission Tomography) imaging workflows for kinetic modeling and quantification.
The methods provided in this module focus on data-driven approaches based on PCA, offering
tools for selecting relevant voxels and refining input functions.

The module is designed for flexibility, offering tools for to apply PCA-based analyses over
4D-PET data and masks. The classes are structured to allow reuse and extension for new functionalities.

Classes
-------
- :class:`PCAGuidedIdifBase`: An abstract base class that supports data preparation, PCA operations,
  and IDIF computation from PCA projections.
- :class:`PCAGuidedTopVoxelsIDIF`: Implements IDIF generation based on selecting the top PCA component
  voxels.
- :class:`PCAGuidedIdifFitterBase`: A base class for deriving optimal voxel fit masks using minimization
  routines and customizable terms (e.g., smoothness, noise).
- :class:`PCAGuidedIdifFitter`: A concrete implementation of :class:`PCAGuidedIdifFitterBase`, providing
  default term definitions.

Dependencies
------------
- :mod:`numpy`: For numerical operations and array manipulations.
- :mod:`ants`: For reading and handling medical image data.
- :mod:`lmfit`: Used for robust fitting and parameter optimization.
- :mod:`sklearn`: For PCA implementation.
- Other project-specific utilities:
    - :class:`~.ScanTimingInfo`: Handles timing information from NIfTI images.
    - :func:`~.temporal_pca_analysis_of_image_over_mask`: Performs PCA on dynamic image data over a given mask.
    - :func:`~.extract_roi_voxel_tacs_from_image_using_mask`: Extracts voxel TACS (Time-Activity Curves) from the input image.

Notes
-----
- The classes contain abstract (``NotImplementedError``) methods, intended to be overridden by derived
  classes as per the specific use case. These serve as stubs for users wishing to extend the module.
- While the module supports large dynamic datasets, care should be taken with memory usage when
  handling large voxel masks or PCA decomposition.

TODO:
    -   Write a tutorial to to use the class to generate an IDIF and use the IF to perform kinetic analysis

"""

import numpy as np
import ants
import lmfit
from sklearn.decomposition import PCA
from lmfit import Minimizer
from lmfit.minimizer import MinimizerResult

from ..preproc.regional_tac_extraction import extract_roi_voxel_tacs_from_image_using_mask as extract_masked_voxels
from ..utils.scan_timing import ScanTimingInfo
from ..utils.data_driven_image_analyses import temporal_pca_analysis_of_image_over_mask as temporal_pca_over_mask
from ..utils.constants import CONVERT_kBq_to_mCi_
from ..utils.time_activity_curve import TimeActivityCurve


class PCAGuidedIdifBase(object):
    """A base class for PCA-guided Image-Derived Input Function (IDIF) generation.

    This class provides core functionality for processing image data, performing PCA, and
    generating IDIF-related metrics such as mean voxel time-activity curves (TACs) and
    PCA-based projections. The class is abstract and intended to be extended with specific
    methods for voxel selection or further analysis.

    Attributes:
        image_path (str): Path to the input dynamic image file.
        mask_path (str): Path to the mask image file used to extract regions of interest.
        output_tac_path (str): Path for saving the output TAC file.
        num_components (int): Number of PCA components to compute.
        verbose (bool): Flag to enable verbose output for diagnostic purposes.
        tac_times_in_mins (np.ndarray): Array containing the mid-time points of each scan
            frame in minutes, extracted using `ScanTimingInfo`.
        idif_vals (np.ndarray): Array of IDIF values calculated from selected voxels.
        idif_errs (np.ndarray): Array of standard deviations of the IDIF values.
        prj_idif_vals (np.ndarray): PCA-projected IDIF values.
        prj_idif_errs (np.ndarray): PCA-projected IDIF value standard deviations.
        pca_obj (PCA or None): PCA object used to perform decomposition, initialized
            during processing.
        pca_fit (np.ndarray or None): Array of PCA-projected voxel representations.
        mask_voxel_tacs (np.ndarray): Time-activity curves (TACs) extracted from the mask
            region in the image.
        mask_avg (np.ndarray): Voxel-wise mean activity within the mask.
        mask_std (np.ndarray): Voxel-wise standard deviation of activity within the mask.
        mask_peak_arg (int): Index of the peak voxel activity in the time-activity curve.
        mask_peak_val (float): Value of the peak voxel activity.
        selected_voxels_mask (np.ndarray or None): Boolean mask of selected voxels after analysis.
        selected_voxels_tacs (np.ndarray or float): TACs for the selected voxels.
        selected_voxels_prj_tacs (np.ndarray or float): PCA-projected TACs for the selected voxels.
        analysis_has_run (bool): Indicates whether the analysis (typically :meth:`run`) has been executed.

    See Also:
        - :class:`~.PCAGuidedTopVoxelsIDIF`
        - :class:`~.PCAGuidedIdifFitter`

    """
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_tac_path: str,
                 num_pca_components: int,
                 verbose: bool):
        r"""Initializes the base class for PCA-guided IDIF generation.

        This constructor sets up the necessary attributes for processing image and mask data,
        performing PCA decomposition, and preparing for downstream TAC and IDIF calculations.

        Args:
            input_image_path (str): The file path for the dynamic 4D image used for generating the IDIF.
            mask_image_path (str): The file path for the mask image used to extract voxel data.
            output_tac_path (str): The file path where the resulting TAC data will be saved.
            num_pca_components (int): The number of PCA components to calculate.
            verbose (bool): If True, enables verbose output for intermediate steps and diagnostics.

        Raises:
            FileNotFoundError: If either the input image file or mask file cannot be found.
        """
        self.image_path: str = input_image_path
        self.mask_path: str = mask_image_path
        self.output_tac_path: str = output_tac_path
        self.num_components: int = num_pca_components
        self.verbose: bool = verbose

        self.tac_times_in_mins: np.ndarray = ScanTimingInfo.from_nifti(image_path=self.image_path).center_in_mins
        self.idif_vals: np.ndarray = np.zeros_like(self.tac_times_in_mins)
        self.idif_errs: np.ndarray = np.zeros_like(self.tac_times_in_mins)
        self.prj_idif_vals: np.ndarray = np.zeros_like(self.tac_times_in_mins)
        self.prj_idif_errs: np.ndarray = np.zeros_like(self.tac_times_in_mins)

        self.pca_obj: PCA | None = None
        self.pca_fit: np.ndarray | None = None

        self.mask_voxel_tacs = extract_masked_voxels(input_image=ants.image_read(self.image_path),
                                                     mask_image=ants.image_read(self.mask_path),
                                                     verbose=self.verbose)

        self.mask_avg = np.mean(self.mask_voxel_tacs, axis=0)
        self.mask_std = np.std(self.mask_voxel_tacs, axis=0)
        self.mask_peak_arg = np.argmax(self.mask_avg)
        self.mask_peak_val = self.mask_avg[self.mask_peak_arg]

        self.perform_temporal_pca()

        self.selected_voxels_mask: np.ndarray | None = None
        self.selected_voxels_tacs: np.ndarray | float = None
        self.selected_voxels_prj_tacs: np.ndarray | float = None

        self.analysis_has_run: bool = False

    def perform_temporal_pca(self):
        """Performs PCA decomposition on the dynamic image data within the specified mask.

        This method applies temporal PCA analysis on the input dynamic image constrained to the
        region defined by the mask. The resulting PCA object and fitted data are stored as
        attributes for subsequent processing. Uses
        :func:`~.extract_roi_voxel_tacs_from_image_using_mask`

        Attributes Updated:
            - `pca_obj`
            - `pca_fit`

        Raises:
            ValueError: If PCA analysis fails during execution.
        """
        self.pca_obj, self.pca_fit = temporal_pca_over_mask(input_image=ants.image_read(self.image_path),
                                                            mask_image=ants.image_read(self.mask_path),
                                                            num_components=self.num_components)

    def rescale_tacs(self, rescale_constant: float = CONVERT_kBq_to_mCi_) -> None:
        r"""Rescales the time-activity curves (TACs) and associated data by a constant factor.

        This method uniformly rescales voxel-level TACs, IDIF values, PCA outputs, and other
        TAC-associated metrics using the specified constant. The default value corresponds to
        a pre-defined scaling factor to convert units between kilobecquerels and millicuries.

        Args:
            rescale_constant (float): The constant factor used for rescaling.
                Must be greater than zero. Default is 37000.0.

        Attributes Updated:
            - `mask_voxel_tacs`
            - `mask_avg`
            - `mask_std`
            - `mask_peak_val`
            - `idif_vals`
            - `idif_errs`
            - `prj_idif_vals`
            - `prj_idif_errs`
            - `selected_voxels_tacs`
            - `selected_voxels_prj_tacs`

        Raises:
            AssertionError: If the `rescale_constant` is not greater than 0.
        """
        assert rescale_constant > 0.0, "rescale_constant must be > 0.0"

        self.mask_voxel_tacs /= rescale_constant
        self.mask_avg /= rescale_constant
        self.mask_std /= rescale_constant
        self.mask_peak_val /= rescale_constant
        self.idif_vals /= rescale_constant
        self.idif_errs /= rescale_constant
        self.prj_idif_vals /= rescale_constant
        self.prj_idif_errs /= rescale_constant

        if self.selected_voxels_tacs is not None:
            self.selected_voxels_tacs /= rescale_constant
        if self.selected_voxels_prj_tacs is not None:
            self.selected_voxels_prj_tacs /= rescale_constant

        return None

    def save(self):
        r"""Saves the computed IDIF and related metrics to a file.

        This method exports the time-activity curves (TACs) and other analysis results
        into a tab-delimited text file. The file will include time points, IDIF values,
        errors, PCA-projected metrics, and mask-derived averages, along with their
        associated standard deviations.

        The output file is saved at the path specified in the `output_tac_path` attribute.

        Raises:
            AssertionError: If the analysis has not been run prior to calling this method.

        Output Format:
            The saved text file will have the following tab-delimited columns:
              - time: Time in minutes.
              - idif: IDIF values.
              - d_idif: Errors (standard deviations) for IDIF.
              - prj_idif: PCA-projected IDIF values.
              - d_prj_idif: Errors (standard deviations) for PCA-projected IDIF values.
              - mask: Mean voxel activity in the mask.
              - d_mask: Standard deviation of voxel activity in the mask.


        """
        assert self.analysis_has_run is not None, "The .run() has not been called yet."
        out_arr = np.asarray([self.tac_times_in_mins,
                              self.idif_vals, self.idif_errs,
                              self.prj_idif_vals, self.prj_idif_errs,
                              self.mask_avg, self.mask_std])
        out_head = ['time', 'idif', 'd_idif', 'prj_idif', 'd_prj_idif', 'mask', 'd_mask']
        np.savetxt(fname=self.output_tac_path, X=out_arr.T,
                   fmt='%.6e', delimiter='\t', comments='',
                   header='\t'.join(out_head))

    def calculate_tacs_from_mask(self) -> None:
        """Calculates Time-Activity Curves (TACs) and IDIF-related metrics from the selected voxel mask.

         This method processes the data from selected voxels within a previously defined mask and
         computes key metrics such as mean IDIF values, standard deviations, PCA-projected values,
         and their respective errors. It utilizes PCA transformations to refine the IDIF and ensure
         consistency with the voxel data.

         Side Effects:
             idif_vals (np.ndarray): Mean IDIF values calculated from the selected voxel TACs.
             idif_errs (np.ndarray): Standard deviations of the IDIF values derived from selected voxels.
             prj_idif_vals (np.ndarray): Mean PCA-projected IDIF values, with negative values set to zero.
             prj_idif_errs (np.ndarray): Standard deviations for the PCA-projected IDIF values.

         Raises:
             AssertionError: If the analysis has not been previously performed (i.e., `.run()` was not called).

         Notes:
             - Negative PCA-projected IDIF values are set to zero since negative values are non-physical.

         """
        assert self.analysis_has_run, "The .run() has not been called yet."
        self.selected_voxels_tacs = self.mask_voxel_tacs[self.selected_voxels_mask]
        self.idif_vals = np.mean(self.selected_voxels_tacs, axis=0)
        self.idif_errs = np.std(self.selected_voxels_tacs, axis=0)
        self.selected_voxels_tacs = self.pca_obj.inverse_transform(self.pca_fit[self.selected_voxels_mask])
        self.prj_idif_vals = np.mean(self.selected_voxels_tacs, axis=0)
        self.prj_idif_vals[self.prj_idif_vals < 0] = 0.
        self.prj_idif_errs = np.std(self.selected_voxels_tacs, axis=0)

    def run(self, *args, **kwargs):
        """Abstract method to be implemented by concrete subclasses.

        Args:
            *args:
            **kwargs:

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Abstract method to be implemented by concrete subclasses.

        Should usually run the `.run()` method and then the `.save()` method to allow for easily running and saving
        results.

        Args:
            *args:
            **kwargs:

        """
        raise NotImplementedError

    @property
    def idif_tac(self):
        """Get the calculated IDIF TAC


        Returns:
            TimeActivityCurve: TAC object with (times, idif_vals)

        See Also:
            :class:`~.TimeActivityCurve`
        """
        return TimeActivityCurve(self.tac_times_in_mins, self.idif_vals)

    @property
    def idif_tac_werr(self):
        """Get the calculated IDIF TAC with the standard deviations


        Returns:
            TimeActivityCurve: TAC object with (times, idif_vals, idif_stderrs)

        See Also:
            :class:`~.TimeActivityCurve`
        """
        return TimeActivityCurve(self.tac_times_in_mins, self.idif_vals, self.idif_errs)

    @property
    def prj_idif_tac(self):
        """Get the calculated PCA-projected IDIF TAC


        Returns:
            TimeActivityCurve: TAC object with (times, projected_idif_vals)

        See Also:
            :class:`~.TimeActivityCurve`
        """
        return TimeActivityCurve(self.tac_times_in_mins, self.prj_idif_vals)

    @property
    def prj_idif_tac_werr(self):
        """Get the calculated PCA-projected IDIF TAC with the standard deviations


        Returns:
            TimeActivityCurve: TAC object with (times, projected_idif_vals, projected_idif_stderrs)

        See Also:
            :class:`~.TimeActivityCurve`
        """
        return TimeActivityCurve(self.tac_times_in_mins, self.prj_idif_vals, self.prj_idif_errs)


class PCAGuidedTopVoxelsIDIF(PCAGuidedIdifBase):
    """Class for calculating a PCA-guided IDIF by averaging over the top voxels of a selected principal component (PC).

    This class extends :class:`~.PCAGuidedIdifBase` and specializes in selecting the most contributing
    voxels for a specific PCA component to refine the IDIF estimation. The user must specify
    the PCA component and the number of voxels to be analyzed for the calculation.

    Example:

    .. code-block:: python

        from petpal.input_function.pca_guided_idif import PCAGuidedTopVoxelsIDIF

        ## Initializing the fitting object
        pca_idif_top = PCAGuidedTopVoxelsIDIF(input_image_path='/path/to/4d/pet/image.nii.gz',
                                              mask_image_path='/path/to/arterial/mask.nii.gz',
                                              output_tac_path='/path/to/save/tac.tsv',
                                              num_pca_components=3,
                                              verbose=True)

        ## Sorting the voxels with respect to the 1st PC, and averaging over 50 voxels
        pca_idif_top.run(selected_component=0,
                         num_of_voxels=50)

        ## Saving the IDIF TACs to disk
        pca_idif_top.save()

    """
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_tac_path: str,
                 num_pca_components: int,
                 verbose: bool):
        """Initializes the PCA-guided Top Voxels IDIF class.

        Args:
            input_image_path (str): Path to the dynamic input image used to calculate the IDIF.
            mask_image_path (str): Path to the mask image used for voxel selection.
            output_tac_path (str): Destination path where the output TAC data will be saved.
            num_pca_components (int): Number of PCA components to compute.
            verbose (bool): If True, enables detailed diagnostic output.

        Side Effects:
            Initializes instance-specific attributes (`num_of_voxels` and `selected_component`)
            as `None` until provided during a call to `run`.
        """
        PCAGuidedIdifBase.__init__(self,
                                   input_image_path=input_image_path,
                                   mask_image_path=mask_image_path,
                                   output_tac_path=output_tac_path,
                                   num_pca_components=num_pca_components,
                                   verbose=verbose)
        self.num_of_voxels: int | None = None
        self.selected_component: int | None = None

    @staticmethod
    def calculate_top_pc_voxels_mask(pca_obj: PCA,
                                     pca_fit: np.ndarray,
                                     pca_component: int,
                                     number_of_voxels: int) -> np.ndarray:
        r"""Determines the top contributing voxels for a specified PCA component.

        Args:
            pca_obj (PCA): A scikit-learn PCA object used to compute the principal components.
            pca_fit (np.ndarray): The array of PCA projections for each voxel.
            pca_component (int): Index of the PCA component for voxel selection (0-based).
            number_of_voxels (int): The number of top-contributing voxels to include.

        Returns:
            np.ndarray: A Boolean mask indicating the voxels selected for the specified PCA component.

        Raises:
            AssertionError: If the `pca_component` index is invalid (e.g., less than 0 or greater
                            than or equal to the number of PCA components in `pca_obj`).

        Notes:
            Voxels are ranked and selected based on a descending sort of their contributions
            to the specified PCA component.
        """

        assert pca_obj.n_components > pca_component >= 0, "PCA component index must be >= 0 and less than the number of total components."
        pc_comp_argsort = np.argsort(pca_fit[:, pca_component])[::-1]
        return pc_comp_argsort[:number_of_voxels]

    def run(self, selected_component: int, num_of_voxels: int) -> None:
        """Executes the PCA-guided IDIF analysis by selecting voxels and calculating TACs.

        Args:
            selected_component (int): The PCA component index to guide voxel selection (0-based).
            num_of_voxels (int): The number of top-contributing voxels to include in the analysis.

        Side Effects:
            - Sets `selected_component` to the chosen PCA component index.
            - Sets `num_of_voxels` to the number of selected voxels.
            - Updates `selected_voxels_mask` with a mask indicating top-contributing voxels.
            - Marks `analysis_has_run` as True after successful execution.
            - Updates IDIF-related metrics (e.g., `idif_vals`, `prj_idif_vals`) by invoking
              `calculate_tacs_from_mask`.

        Raises:
            AssertionError: If `num_of_voxels` is less than 3.

        Notes:
            This method must be called before saving or further analyzing the results, as it sets
            the necessary state attributes.
        """
        assert num_of_voxels > 2, "num_of_voxels must be greater than 2."
        self.selected_component = selected_component
        self.num_of_voxels = num_of_voxels
        self.selected_voxels_mask = self.calculate_top_pc_voxels_mask(pca_obj=self.pca_obj,
                                                                      pca_fit=self.pca_fit,
                                                                      pca_component=self.selected_component,
                                                                      number_of_voxels=self.num_of_voxels)
        self.analysis_has_run = True
        self.calculate_tacs_from_mask()

    def __call__(self, selected_component: int, num_of_voxels: int) -> None:
        r"""Callable interface for running the analysis and saving the results.

        This method provides a convenient way to execute the analysis and save the resultant
        TAC data to the specified output file in a single call.

        Args:
            selected_component (int): The PCA component index to guide voxel selection (0-based).
            num_of_voxels (int): The number of top-contributing voxels to include in the analysis.

        Side Effects:
            - Runs the full analysis using the `run` method.
            - Saves the resulting TAC data to the file specified in `output_tac_path`.

        Notes:
            Combines the functionality of `run` and `save` for streamlined usage.
        """
        self.run(selected_component=selected_component, num_of_voxels=num_of_voxels)
        self.save()


class PCAGuidedIdifFitterBase(PCAGuidedIdifBase):
    """Base class for calculating the PCA-guided IDIF by fitting to find the best voxels

    This class extends :class:`~.PCAGuidedIdifBase` and provides a framework for applying optimizations
    and heuristics (e.g., voxel count, peak sharpness, and smoothness) to filter and refine the
    targeted voxel selection for IDIF estimation.

    Notes:
        This class provides a foundation for subclasses to customize voxel filtering and
        optimization strategies by overriding the static methods for term functions.

    See :class:`~.PCAGuidedIdifFitter` for a concrete implementation example.

    """
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_tac_path: str,
                 num_pca_components: int,
                 pca_comp_filter_min_value: float,
                 pca_comp_threshold: float,
                 verbose: bool):
        """Initializes the PCA-guided IDIF fitter base-class.

        Args:
            input_image_path (str): Path to the dynamic input image.
            mask_image_path (str): Path to the mask image used to select the region of interest.
            output_tac_path (str): Path where the output TAC data will be saved.
            num_pca_components (int): Number of PCA components to compute.
            pca_comp_filter_min_value (float): Minimum value for filtering PCA components.
            pca_comp_threshold (float): Threshold for filtering PCA components.
            verbose (bool): Whether to enable detailed diagnostic output.

        Side Effects:
            - Initializes several fitting-related attributes (e.g., `alpha`, `beta`, `_fitting_params`).
            - Calculates PCA filtering flags and signs based on the passed thresholds.
        """
        PCAGuidedIdifBase.__init__(self,
                                   input_image_path=input_image_path,
                                   mask_image_path=mask_image_path,
                                   output_tac_path=output_tac_path,
                                   num_pca_components=num_pca_components,
                                   verbose=verbose)
        self.mask_peak_val += self.mask_std[self.mask_peak_arg] * 3.
        self.alpha: float | None = None
        self.beta: float | None = None

        self.pca_filter_flags: np.ndarray | None = None
        self.filter_signs: np.ndarray | None = None

        self._fitting_params: lmfit.Parameters | None = None
        self.fitting_obj: Minimizer | None = None
        self.fit_result: MinimizerResult | None = None
        self.result_params: lmfit.Parameters | None = None
        self.fit_quantiles: np.ndarray | None = None

        self._pca_comp_filter_min_val = pca_comp_filter_min_value
        self._pca_comp_filter_threshold = pca_comp_threshold
        self.calculate_filter_flags_and_signs(comp_min_val=self.pca_comp_filter_min_val,
                                              threshold=self.pca_comp_filter_flag_threshold)
        self._fitting_params = self._generate_quantile_params(num_components=self.num_components)

    @property
    def pca_comp_filter_flag_threshold(self) -> float:
        return self._pca_comp_filter_threshold

    @pca_comp_filter_flag_threshold.setter
    def pca_comp_filter_flag_threshold(self, val: float) -> None:
        self._pca_comp_filter_threshold = val
        self.calculate_filter_flags_and_signs(comp_min_val=self.pca_comp_filter_min_val, threshold=val)

    @property
    def pca_comp_filter_min_val(self) -> float:
        return self._pca_comp_filter_min_val

    @pca_comp_filter_min_val.setter
    def pca_comp_filter_min_val(self, val: float):
        self._pca_comp_filter_min_val = val
        self.calculate_filter_flags_and_signs(comp_min_val=val, threshold=self.pca_comp_filter_flag_threshold)

    @staticmethod
    def get_pca_component_filter_flags(pca_components: np.ndarray,
                                       comp_min_val: float = 0.0,
                                       threshold: float = 0.1) -> np.ndarray[bool]:
        """Generates filtering flags for PCA components based on their contribution.

        Given the `comp_min_val`, we check how many of the time-points of the PCA components are
        above the threshold, then `threshold` determines if the component contributes with ``>`` (if
        the fraction of higher points is above the threshold) or ``<`` (if the fraction of higher points is
        below the threshold).

        Args:
            pca_components (np.ndarray): Array of PCA components for evaluation.
            comp_min_val (float): Minimum value for PCA components to contribute positively.
            threshold (float): Threshold for rejecting components with insufficient contributions.

        Returns:
            np.ndarray[bool]: Boolean flags indicating whether a PCA filters less than, or greater than.
        """
        pca_components_positive_pts: np.ndarray = np.mean(pca_components > comp_min_val, axis=1)
        pca_components_filter_flags = ~(pca_components_positive_pts > threshold)
        return pca_components_filter_flags

    @staticmethod
    def get_pca_filter_signs_from_flags(pca_component_filter_flags: np.ndarray[bool]) -> list[str]:
        """Derives signs ('>' or '<') for PCA components based on their filtering flags.

        See Also:
            :meth:`get_pca_component_filter_flags`

        Args:
            pca_component_filter_flags (np.ndarray[bool]): Filtering flags for PCA components.

        Returns:
            list[str]: List of signs corresponding to the filter flags.
        """
        return ['>' if sgn else '<' for sgn in ~pca_component_filter_flags]

    @staticmethod
    def _voxel_term_func(voxel_nums: float) -> float:
        """Abstract method for computing the voxel-number term for the optimization function.

        Args:
            voxel_nums (float): Number of valid voxels contributing to the TACs.

        Returns:
            float: Calculated voxel term.
        """
        raise NotImplementedError

    @staticmethod
    def _noise_term_func(tac_stderrs: np.ndarray[float]) -> float:
        """Abstract method for computing the noise term for the optimization function.

        Args:
            tac_stderrs (np.ndarray[float]): Standard deviations of TACs from voxel data.

        Returns:
            float: Calculated noise term.
        """
        raise NotImplementedError

    @staticmethod
    def _smoothness_term_func(tac_values: np.ndarray[float]) -> float:
        """Abstract method for computing the smoothness term for the optimization function.

        Args:
            tac_values (np.ndarray[float]): Mean values of TACs from voxel data.

        Returns:
            float: Calculated smoothness term.
        """
        raise NotImplementedError

    @staticmethod
    def _peak_term_func(tac_peak_ratio: float) -> float:
        """Abstract method for computing the peak term for the optimization function.

        Args:
            tac_peak_ratio (float): Ratio of the voxel TAC peak to the peak of the mask average

        Returns:
            float: Calculated peak term.
        """
        raise NotImplementedError

    @staticmethod
    def _generate_quantile_params(num_components: int = 3,
                                  value: float = 0.5,
                                  lower: float = 1e-4,
                                  upper: float = 0.999) -> lmfit.Parameters:
        """Generates initial fitting quantile parameters for determining voxel filters.

        Args:
            num_components (int): The number of PCA components to generate parameters for.
            value (float): Default value for each quantile parameter (e.g., median = 0.5).
            lower (float): Lower bound for the quantile parameter (default is 1e-4).
            upper (float): Upper bound for the quantile parameter (default is 0.999).

        Returns:
            lmfit.Parameters: A parameter set with quantile values for each PCA component.
        """
        tmp_dict = {'value': value, 'min': lower, 'max': upper}
        return lmfit.create_params(**{f'pc{i}': tmp_dict for i in range(num_components)})

    @staticmethod
    def calculate_voxel_mask_from_quantiles(params: lmfit.Parameters,
                                            pca_values_per_voxel: np.ndarray,
                                            quantile_flags: np.ndarray[bool]) -> np.ndarray[bool]:
        """Calculates a mask to identify selected voxels based on quantile thresholds.

        Given the passed in quantile `params`, we check if the value of the principal component for a
        voxel is above the quantile threshold, then `quantile_flags` is used to flip the signs of
        the components which should be lower than. For each voxel, the boolean product is calculated to
        determine if the voxel should be a part of the mask or not. Only voxels where every PC is above (or below)
        the quantile thresholds in `params` are included in the mask.

        Args:
            params (lmfit.Parameters): Quantile parameters for each PCA component.
            pca_values_per_voxel (np.ndarray): PCA values for every voxel.
            quantile_flags (np.ndarray[bool]): Filtering flags for quantile-based masking.

        Returns:
            np.ndarray[bool]: Mask identifying selected voxels passing the quantile criteria.
        """
        voxel_mask = np.ones(len(pca_values_per_voxel), dtype=bool)
        quantile_values = params.valuesdict().values()
        for pca_component, quantile, flag in zip(pca_values_per_voxel.T, quantile_values, quantile_flags):
            voxel_mask *= (pca_component > np.quantile(pca_component, quantile)) ^ flag
        return voxel_mask

    def calculate_filter_flags_and_signs(self, comp_min_val: float, threshold: float):
        """Updates PCA filter flags and their respective signs based on component values.

        Args:
            comp_min_val (float): Minimum acceptable value for filtering PCA components.
            threshold (float): Threshold percentage for filtering components.

        Side Effects:
            - Updates the `pca_filter_flags` attribute with new flags for indicating filtered components.
            - Updates the `filter_signs` attribute with the corresponding value comparison signs
              based on the flags.
        """
        self.pca_filter_flags = self.get_pca_component_filter_flags(pca_components=self.pca_obj.components_,
                                                                    comp_min_val=comp_min_val,
                                                                    threshold=threshold)
        self.filter_signs = self.get_pca_filter_signs_from_flags(pca_component_filter_flags=self.pca_filter_flags)

    def residual(self,
                 params: lmfit.Parameters,
                 pca_values_per_voxel: np.ndarray[float],
                 quantile_flags: np.ndarray[bool],
                 voxel_tacs: np.ndarray,
                 alpha: float,
                 beta: float) -> float:
        r"""Calculates the residual (objective function) for optimization.

        The residual has the following terms:

        .. math::

            \mathcal{L} = \mathrm{VoxelTerm} + \mathrm{NoiseTerm} + (\alpha\cdot\mathrm{PeakTerm}) + (\beta\cdot\mathrm{SmoothnessTerm})


        Args:
            params (lmfit.Parameters): Parameters for the optimization.
            pca_values_per_voxel (np.ndarray[float]): PCA values for each voxel.
            quantile_flags (np.ndarray[bool]): Flags indicating quantile filtering.
            voxel_tacs (np.ndarray): TACs for valid voxels.
            alpha (float): Weight for the peak term.
            beta (float): Weight for the smoothness term.

        Returns:
            float: Residual value.
        """
        voxel_mask = self.calculate_voxel_mask_from_quantiles(params, pca_values_per_voxel, quantile_flags)
        valid_voxels_number = np.sum(voxel_mask)
        masked_voxels = voxel_tacs[voxel_mask]

        tacs_avg = np.mean(masked_voxels, axis=0) if valid_voxels_number > 1 else self.mask_avg
        tacs_std = np.std(masked_voxels, axis=0) if valid_voxels_number > 1 else self.mask_std

        voxel_term = self._voxel_term_func(voxel_nums=valid_voxels_number)
        noise_term = self._noise_term_func(tac_stderrs=tacs_std)

        peak_ratio = tacs_avg[self.mask_peak_arg] / self.mask_peak_val
        peak_term = alpha * self._peak_term_func(tac_peak_ratio=peak_ratio) if alpha != 0.0 else 0.0
        smth_term = beta * self._smoothness_term_func(tac_values=tacs_avg) if beta != 0.0 else 0.0

        return voxel_term + noise_term + peak_term + smth_term

    def run(self,
            alpha: float, beta: float,
            method: str = 'ampgo', **method_kwargs):
        """Runs the PCA-guided IDIF fitting process using optimization.

        Args:
            alpha (float): Weight for the peak term in the optimization.
            beta (float): Weight for the smoothness term in the optimization.
            method (str): Optimization method to use (default is 'ampgo').
            **method_kwargs: Additional keyword arguments for the optimizer.

        Side Effects:
            - Updates optimization-related attributes (e.g., `fit_result`, `fit_quantiles`).
            - Updates `selected_voxels_mask` with the best voxel subset.
            - Marks `analysis_has_run` as True.

        Notes:
            The `run` method uses the residual function as the target for minimization.
        """
        self.alpha = alpha
        self.beta = beta
        self.fitting_obj = lmfit.Minimizer(userfcn=self.residual,
                                           params=self._fitting_params,
                                           fcn_args=(self.pca_fit, self.pca_filter_flags, self.mask_voxel_tacs,
                                                     alpha, beta))
        self.fit_result = self.fitting_obj.minimize(method=method, **method_kwargs)
        self.result_params = self.fit_result.params
        self.fit_quantiles = np.asarray(list(self.fit_result.params.valuesdict().values()))
        self.selected_voxels_mask = self.calculate_voxel_mask_from_quantiles(params=self.fit_result.params,
                                                                             pca_values_per_voxel=self.pca_fit,
                                                                             quantile_flags=self.pca_filter_flags, )
        self.analysis_has_run = True
        self.calculate_tacs_from_mask()

    def __call__(self, alpha: float, beta: float, method: str, **meth_kwargs) -> None:
        """Callable interface for running the fitting process and saving results.

        Args:
            alpha (float): Weight for the peak term in the optimization.
            beta (float): Weight for the smoothness term in the optimization.
            method (str): Optimization method to use.
            **method_kwargs: Additional keyword arguments for the optimizer.

        Side Effects:
            - Executes the analysis and saves the resulting TAC data to the output file.
        """
        self.run(alpha=alpha, beta=beta, method=method, **meth_kwargs)
        self.save()


class PCAGuidedIdifFitter(PCAGuidedIdifFitterBase):
    """Class to calculate the PCA-guided Image-Derived Input Function (IDIF) by fitting to find the best voxels.

    This class extends :class:`~.PCAGuidedIdifFitterBase` and implements domain-specific functions to guide the
    optimization process, including terms for voxel count, noise reduction, smoothness enforcement,
    and peak emphasis. The optimization aims to refine the voxel selection for IDIF estimation using
    the provided fitting terms.

    Example:

        .. code-block:: python

            from petpal.input_function.pca_guided_idif import PCAGuidedIdifFitter

            ## Initializing the fitting object
            pca_idif_fit = PCAGuidedIdifFitter(input_image_path='/path/to/4d/pet/image.nii.gz',
                                               mask_image_path='/path/to/arterial/mask.nii.gz',
                                               output_tac_path='/path/to/save/tac.tsv',
                                               num_pca_components=3,
                                               pca_comp_filter_min_value=0.0,
                                               pca_comp_threshold=0.1,
                                               verbose=True)

            ## Running the fitting with the dual_annealing minimization method
            pca_idif_fit.run(alpha=2.5,
                             beta=0.0,
                             method='dual_annealing')

            ## Saving the IDIF TACs to disk
            pca_idif_fit.save()



    """
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_tac_path: str,
                 num_pca_components: int,
                 pca_comp_filter_min_value: float = 0.0,
                 pca_comp_threshold: float = 0.1,
                 verbose: bool = False):
        """Initializes the PCA-guided IDIF fitter.

        Args:
            input_image_path (str): Path to the input dynamic image used for IDIF estimation.
            mask_image_path (str): Path to the mask image for selecting the region of interest.
            output_tac_path (str): Path where the calculated TAC data will be saved.
            num_pca_components (int): Number of PCA components to compute for voxel selection.
            pca_comp_filter_min_value (float, optional): Minimum value for filtering PCA components (default: 0.0).
            pca_comp_threshold (float, optional): Threshold for filtering PCA components (default: 0.1).
            verbose (bool, optional): Enables detailed diagnostic output if set to `True` (default: `False`).

        Side Effects:
            - Calls the base class constructor to initialize parameters and fit-related attributes.

        """
        PCAGuidedIdifFitterBase.__init__(self,
                                         input_image_path=input_image_path,
                                         mask_image_path=mask_image_path,
                                         output_tac_path=output_tac_path,
                                         num_pca_components=num_pca_components,
                                         pca_comp_filter_min_value=pca_comp_filter_min_value,
                                         pca_comp_threshold=pca_comp_threshold,
                                         verbose=verbose)

    @staticmethod
    def _voxel_term_func(voxel_nums: float) -> float:
        r"""A softplus function to encourage optimization toward selecting more voxels.

        This term penalizes scenarios where fewer voxels are selected by incrementally rewarding
        configurations with larger voxel numbers.

        Notes:
            Applies the softplus function defined as :math:`\ln\left(1+e^{v/6}\right)`
            where :math:`v` is the number of voxels.

        Args:
            voxel_nums (float): The number of voxels contributing to the TAC.

        Returns:
            float: The calculated voxel term value.

        """
        return np.log1p(np.exp(-voxel_nums / 6.0))

    @staticmethod
    def _noise_term_func(tac_stderrs: np.ndarray[float]) -> float:
        r"""Root-mean-square (RMS) noise term to favor voxels with reduced per-frame variance.

        This term penalizes higher noise levels in the selected voxels' TAC data, rewarding smoother
        per-frame signal averages.

        Notes:
            The noise term is computed as :math:`\sqrt{\frac{1}{n}\sum_{i}^{n}{\sigma_i}^2}`, where :math:`\sigma_i`
            is the per-frame standard deviation, and :math:`n` is the number of frames.

        Args:
            tac_stderrs (np.ndarray[float]): The array of standard errors of the mean (SEM) for the TAC.

        Returns:
            float: The calculated noise term value.

        """
        return np.sqrt(np.mean(tac_stderrs ** 2))

    @staticmethod
    def _smoothness_term_func(tac_values: np.ndarray[float]) -> float:
        r"""Smoothness term enforcing total variation consistency on the TAC values.

         This term penalizes abrupt changes in frame-to-frame TAC values by calculating the L1-norm
         of differences in TAC values. It helps ensure smoother TACs from the selected voxels.

         Notes:
             Defined as the sum of absolute differences in normalized TAC values:
             :math:`\frac{\beta}{c_\mathrm{max}}\sum_{i}^{n}{\left|c_{i+1} - c_{i}\right|}` where :math:`\beta`
             is a tuning parameter, and :math:`n` is the number of frames.

         Args:
             tac_values (np.ndarray[float]): The TAC values obtained from the selected voxels.

         Returns:
             float: The calculated smoothness term value.

         """
        return np.sum(np.abs(np.diff(tac_values, prepend=tac_values[0]) / np.max(tac_values)))

    @staticmethod
    def _peak_term_func(tac_peak_ratio: float) -> float:
        r"""
        Peak term driving the TAC peak value to be sufficiently high.

        This term rewards configurations where the TAC has a pronounced peak relative to its
        baseline, as higher peaks often correlate with improved IDIF estimates.

        Notes:
            Applies the softplus function defined as :math:`\alpha \times\ln\left(1+e^{-\frac{3}{2}r}\right)` where
            :math:`\alpha` is the tuning parameter, and :math:`r` is the ratio of the value of the computed TAC
            to the value of the average TAC of the mask.

        Args:
            tac_peak_ratio (float): The ratio of the TAC peak value to a reference value.

        Returns:
            float: The calculated peak term value.

        """
        return np.log1p(np.exp(-tac_peak_ratio * 1.5))
