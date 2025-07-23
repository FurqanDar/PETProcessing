"""Tools for calculating weights for application to kinetic models."""
import numpy as np

from ..utils.time_activity_curve import TimeActivityCurve
from ..utils.image_io import get_half_life_from_nifti
from ..utils.scan_timing import ScanTimingInfo

class TacWeight:
    """Determine weighting scheme for Time Activity Curves. Includes options for constant,
    calculated, or preset weighting.    
    """
    def __init__(self,
                 time_activity_curve: TimeActivityCurve,
                 input_image_path: str=None):
        """Initialize TacWeight with provided arguments.

        Args:
            time_activity_curve (TimeActivityCurve): The time activity curve on which weights are
                applied.
            input_image_path (str): Path to the PET image used to create the TAC supplied to
                object. Used only to retrieve scan timing and half life information for calculated
                TAC weights. Default None.
        """
        self.time_activity_curve = time_activity_curve
        self.input_image_path = input_image_path


    def weight_tac_simple(self,
                          tac_durations_in_minutes: np.ndarray,
                          tac_vals: np.ndarray) -> np.ndarray:
        """Weight a Time Activity Curve (TAC) based on variance. This function applies the simple
        frame time length to activity ratio found in
        http://www.turkupetcentre.net/petanalysis/model_weighting.html.

        Args:
            tac_durations_in_minutes (np.ndarray): Duration of each frame in the TAC in minutes.
            tac_vals (np.ndarray): Activity values for each frame in the TAC.

        Returns:
            tac_weights (np.ndarray): Weights to be applied to the TAC during fitting process.
        """
        tac_weights = tac_durations_in_minutes/tac_vals
        tac_vals_where_zero = np.where(tac_vals==0)
        tac_weights[tac_vals_where_zero] = 0
        return tac_weights


    def weight_tac_decay(self,
                         tac_durations_in_minutes: np.ndarray,
                         tac_vals: np.ndarray,
                         tac_times_in_minutes: np.ndarray,
                         half_life: float) -> np.ndarray:
        """Weight a Time Activity Curve (TAC) based on variance. This function applies the simple
        frame time length to activity ratio found in
        http://www.turkupetcentre.net/petanalysis/model_weighting.html with an extra factor for
        decay correction.

        Args:
            tac_durations_in_minutes (np.ndarray): Duration of each frame in the TAC in minutes.
            tac_vals (np.ndarray): Activity values for each frame in the TAC.
            tac_times_in_minutes (np.ndarray): Frame times for the TAC in minutes.
            half_life (np.ndarray): Half life of the radioisotope in seconds.

        Returns:
            tac_weights (np.ndarray): Weights to be applied to the TAC during fitting process.
        """
        decay_factor = np.exp(-np.log(2) / (half_life / 60) * tac_times_in_minutes)
        tac_weights = tac_durations_in_minutes * decay_factor / tac_vals
        tac_vals_where_zero = np.where(tac_vals==0)
        tac_weights[tac_vals_where_zero] = 0
        return tac_weights


    def convert_weights_to_sigma(self, tac_weights: np.ndarray) -> np.ndarray:
        r"""Convert TAC weights to sigma (standard deviation) values. Calculated as
        :math:`\sigma=w^{-1/2}`. Returns zero as the sigma value if the weight at that time point is
        zero.

        Args:
            tac_weights (np.ndarray): Weights calculated using :meth:`weight_tac_simple` or
            `weight_tac_decay`.

        Returns:
            tac_sigma (np.ndarray): Array of sigmas calculated from the weights.
        """
        tac_sigma = np.power(tac_weights,-1/2)
        tac_weights_where_zero = np.where(tac_weights==0)
        tac_sigma[tac_weights_where_zero] = np.inf
        return tac_sigma


    @staticmethod
    def convert_sigma_to_weights(tac_uncertainty: np.ndarray) -> np.ndarray:
        r"""Convert TAC sigma (standard deviation) to weights. Calculated as
        :math:`w=\sigma^{-2}`. Returns zero as the sigma value if the sigma at that time point is
        inf.

        Args:
            tac_weights (np.ndarray): Weights calculated using :meth:`weight_tac_simple` or
            `weight_tac_decay`.

        Returns:
            tac_sigma (np.ndarray): Array of sigmas calculated from the weights.
        """
        tac_weight = np.power(tac_uncertainty,-2)
        tac_sigma_where_inf = np.where(tac_uncertainty==np.inf)
        tac_weight[tac_sigma_where_inf] = 0
        return tac_weight


    @property
    def constant_weights(self):
        """Get constant weights for the TAC.
        """
        weights = np.ones_like(self.time_activity_curve.activity)
        return weights


    @property
    def provided_weights(self):
        """Get user provided weights for the TAC.
        """
        sigma = self.time_activity_curve.uncertainty
        weights = self.convert_sigma_to_weights(tac_uncertainty=sigma)
        return weights


    @property
    def half_life(self):
        """The half life in seconds of the radiotracer used in the analysis."""
        return get_half_life_from_nifti(image_path=self.input_image_path)


    @property
    def scan_timing(self):
        """The scan timing for the input image used in the analysis."""
        return ScanTimingInfo.from_nifti(image_path=self.input_image_path)


    def validate_scan_timing(self):
        """Validate the input image successfully retrieved scan_timing property for calculating
        TAC weights.
        
        Raises:
            TypeError: When property self.scan_timing returns a TypeError on attempt to access."""
        try:
            self.scan_timing
        except TypeError as exc:
            raise TypeError("Could not read scan timing from input image. Make sure to provide "
                            "input_image_path when using calculated weights. input_image_path "
                            f"currently set to: {self.input_image_path}.") from exc


    @property
    def calculated_weights(self):
        """The calculated weights for the TAC.
        """
        self.validate_scan_timing()
        weights = self.weight_tac_decay(tac_durations_in_minutes=self.scan_timing.duration_in_mins,
                                        tac_vals=self.time_activity_curve.activity,
                                        tac_times_in_minutes=self.scan_timing.center_in_mins,
                                        half_life=self.half_life)
        return weights


    def __call__(self, weight_method: str) -> np.ndarray:
        """Get the TAC weights corresponding to the identified method.
        
        Args:
            weight_methood (str): TAC weight type to apply to the model.

        Returns:
            weights (np.ndarray): The weight applied to each time frame in the model.

        Raises:
            ValueError: If `weight_method` is not one of: 'constant', 'calculated', or 'provided'.
        """
        match weight_method:
            case 'constant':
                return self.constant_weights
            case 'provided':
                return self.provided_weights
            case 'calculated':
                return self.calculated_weights
            case _:
                raise ValueError("weight_method must be one of: 'constant','calculated', "
                                f"'provided'. Got {weight_method}.")
