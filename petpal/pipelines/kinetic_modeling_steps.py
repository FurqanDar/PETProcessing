from typing import Union
from .steps_base import *
from ..kinetic_modeling import parametric_images
from ..kinetic_modeling import tac_fitting
from ..kinetic_modeling import rtm_analysis as pet_rtms
from ..kinetic_modeling import graphical_analysis as pet_grph
from .preproc_steps import TACsFromSegmentationStep, ResampleBloodTACStep, PreprocStepType, ImageToImageStep
from ..utils.bids_utils import parse_path_to_get_subject_and_session_id, gen_bids_like_dir_path

class TACAnalysisStepMixin(StepsAPI):
    """
    A mixin class for TAC analysis steps.

    This class manages initialization and properties related to TAC paths, directories, and output
    configurations. It provides methods to infer output paths and prefixes based on input TAC paths.

    Attributes:
        input_tac_path (str): Path to the input TAC file. Doubles as the path to a reference tac if appropriate.
        roi_tacs_dir (str): Directory containing the ROI TAC files to be analyzed.
        output_directory (str): Directory where output files will be saved.
        output_prefix (str): Prefix for the output files.
        is_ref_tac_based_model (bool): Indicates if the model is reference TAC-based.
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 is_ref_tac_based_model: bool,
                 **kwargs):
        """
        Initializes the TACAnalysisStepMixin with specified parameters.

        Args:
            input_tac_path (str): Path to the input TAC file.
            roi_tacs_dir (str): Directory containing the ROI TAC files.
            output_directory (str): Directory where output files will be saved.
            output_prefix (str): Prefix for the output files.
            is_ref_tac_based_model (bool): Indicates if the model is reference TAC-based.
            **kwargs: Additional keyword arguments to be included in initialization.
        """
        common_init_kwargs = dict(roi_tacs_dir=roi_tacs_dir, output_directory=output_directory,
                                  output_filename_prefix=output_prefix, )
        if is_ref_tac_based_model:
            self.init_kwargs = dict(ref_tac_path=input_tac_path, **common_init_kwargs, **kwargs)
        else:
            self.init_kwargs = dict(input_tac_path=input_tac_path, **common_init_kwargs, **kwargs)
        self._tacs_dir = roi_tacs_dir
        self._input_tac_path = input_tac_path
        self._output_directory = output_directory
        self._output_prefix = output_prefix
    
    @property
    def input_tac_path(self) -> str:
        """
        Gets the path to the input TAC file.

        Returns:
            str: Path to the input TAC file.
        """
        return self._input_tac_path
    
    @input_tac_path.setter
    def input_tac_path(self, input_tac_path: str):
        """
        Sets the path to the input TAC file.

        Args:
            input_tac_path (str): Path to the input TAC file.
        """
        self._input_tac_path = input_tac_path
        self.init_kwargs['input_tac_path'] = input_tac_path
    
    @property
    def reference_tac_path(self) -> str:
        """
        Gets the path to the reference TAC file.

        Returns:
            str: Path to the reference TAC file.
        """
        return self.input_tac_path
    
    @reference_tac_path.setter
    def reference_tac_path(self, ref_tac_path: str):
        """
        Sets the path to the reference TAC file.

        Args:
            ref_tac_path (str): Path to the reference TAC file.
        """
        self._input_tac_path = ref_tac_path
        self.init_kwargs['ref_tac_path'] = ref_tac_path
    
    @property
    def tacs_dir(self) -> str:
        """
        Gets the directory containing the TAC files to be analyzed.

        Returns:
            str: Directory containing the TAC files to be analyzed.
        """
        return self._tacs_dir
    
    @tacs_dir.setter
    def tacs_dir(self, tacs_dir: str):
        """
        Sets the directory containing the TAC files to be analyzed.

        Args:
            tacs_dir (str): Directory containing the TAC files to be analyzed.
        """
        self._tacs_dir = tacs_dir
    
    @property
    def roi_tacs_dir(self) -> str:
        """
        Gets the directory containing the ROI TAC files to be analyzed.

        Returns:
            str: Directory containing the ROI TAC files to be analyzed.
        """
        return self.tacs_dir
    
    @roi_tacs_dir.setter
    def roi_tacs_dir(self, roi_tacs_dir: str):
        """
        Sets the directory containing the ROI TAC files to be analyzed.

        Args:
            roi_tacs_dir (str): Directory containing the ROI TAC files to be analyzed.
        """
        self.tacs_dir = roi_tacs_dir
        self.init_kwargs['roi_tacs_dir'] = roi_tacs_dir
    
    @property
    def output_directory(self) -> str:
        """
        Gets the directory where output files will be saved.

        Returns:
            str: Directory where output files will be saved.
        """
        return self._output_directory
    
    @output_directory.setter
    def output_directory(self, output_directory: str):
        """
        Sets the directory where output files will be saved.

        Args:
            output_directory (str): Directory where output files will be saved.
        """
        self._output_directory = output_directory
        self.init_kwargs['output_directory'] = output_directory
    
    @property
    def output_prefix(self) -> str:
        """
        Gets the prefix for the output files. Usually something like ``sub-XXXX_ses-XX`` if BIDS compliant output.

        Returns:
            str: Prefix for the output files.
        """
        return self._output_prefix
    
    @output_prefix.setter
    def output_prefix(self, output_prefix: str):
        """
        Sets the prefix for the output files.

        Args:
            output_prefix (str): Prefix for the output files.
        """
        self._output_prefix = output_prefix
        self.init_kwargs['output_filename_prefix'] = output_prefix
    
    @property
    def out_path_and_prefix(self):
        """
        Gets the tuple of output directory and prefix.

        Returns:
            tuple: A tuple containing output directory and prefix.
        """
        return self._output_directory, self._output_prefix
    
    @out_path_and_prefix.setter
    def out_path_and_prefix(self, out_dir_and_prefix):
        """
        Sets the output directory and prefix based on a tuple.

        Args:
            out_dir_and_prefix (tuple): A tuple containing output directory and prefix.

        Raises:
            ValueError: If the tuple does not contain exactly two items.
        """
        try:
            out_dir, out_prefix = out_dir_and_prefix
        except ValueError:
            raise ValueError("Pass a tuple with two items: `(out_dir, out_prefix)`")
        else:
            self.output_directory = out_dir
            self.output_prefix = out_prefix
    
    def infer_prefix_from_input_tac_path(self):
        """
        Infers the output prefix based on the input/reference TAC file path. Gets the subject and session from the
        path if possible.
        
        See Also:
            :func:`parse_path_to_get_subject_and_session_id<petpal.utils.bids_utils.parse_path_to_get_subject_and_session_id>`
        """
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_tac_path)
        self.output_prefix = f'sub-{sub_id}_ses-{ses_id}'
    
    def infer_output_directory_from_input_tac_path(self, out_dir: str, der_type: str = 'km'):
        """
        Infers the output directory based on the input/reference TAC file path. Gets the subject and session from the
        path is possible.

        Args:
            out_dir (str): Root output directory.
            der_type (str): Type of derivatives. Defaults to 'km'.
            
        See Also:
            :func:`parse_path_to_get_subject_and_session_id<petpal.utils.bids_utils.parse_path_to_get_subject_and_session_id>`
            :func:`gen_bids_like_dir_path<petpal.utils.bids_utils.gen_bids_like_dir_path>`
        """
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_tac_path)
        outpath = gen_bids_like_dir_path(sub_id=sub_id, ses_id=ses_id, modality=der_type, sup_dir=out_dir)
        self.output_directory = outpath
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type, suffix=None, ext=None, **extra_desc):
        """
        Infers the output directory and prefix based on the input/reference TAC file path. Gets the subject and session
        from the path is possible.

        Args:
            out_dir (str): Root output directory.
            der_type (str): Type of derivatives.
            suffix (str, optional): Suffix for the output files.
            ext (str, optional): Extension for the output files.
            **extra_desc: Additional descriptive parameters.
        """
        self.infer_prefix_from_input_tac_path()
        self.infer_output_directory_from_input_tac_path(out_dir=out_dir, der_type=der_type)
    
    def set_input_as_output_from(self, sending_step: PreprocStepType) -> None:
        """
        Sets the input parameters based on the output from a specified sending step.

        Args:
            sending_step (PreprocStepType): The step from which to derive the input parameters.
        """
        if isinstance(sending_step, TACsFromSegmentationStep):
            self.roi_tacs_dir = sending_step.out_tacs_dir
        elif isinstance(sending_step, ResampleBloodTACStep):
            self.input_tac_path = sending_step.resampled_tac_path
        else:
            super().set_input_as_output_from(sending_step)


class GraphicalAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    """
    A step for performing graphical analysis on TACs using various methods. Uses
    :class:`MultiTACGraphicalAnalysis<petpal.kinetic_modeling.graphical_analysis.MultiTACGraphicalAnalysis>`.

    This class sets up graphical analysis, initializes required paths and parameters,
    and provides class methods for creating default steps with common graphical analysis
    methods like Patlak, Logan, and Alternative Logan.

    Attributes:
        input_tac_path (str): Path to the input TAC file.
        roi_tacs_dir (str): Directory containing the ROI TAC files.
        output_directory (str): Directory where output files will be saved.
        output_prefix (str): Prefix for the output files.
        method (str): Graphical analysis method.
        fit_threshold_in_mins (float): Threshold in minutes for fitting. Defaults to 30.0.
        image_rescale (float): Scale for the image values. Defaults to 1.0 / 37000.0.
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 method: str,
                 fit_threshold_in_mins: float = 30.0):
        """
        Initializes the GraphicalAnalysisStep with specified parameters.

        Args:
            input_tac_path (str): Path to the input TAC file.
            roi_tacs_dir (str): Directory containing the ROI TAC files.
            output_directory (str): Directory where output files will be saved.
            output_prefix (str): Prefix for the output files.
            method (str): Graphical analysis method.
            fit_threshold_in_mins (float, optional): Threshold in minutes for fitting. Defaults to 30.0.
            
        """
        TACAnalysisStepMixin.__init__(self, input_tac_path=input_tac_path, roi_tacs_dir=roi_tacs_dir,
                                      output_directory=output_directory, output_prefix=output_prefix,
                                      is_ref_tac_based_model=False, method=method,
                                      fit_thresh_in_mins=fit_threshold_in_mins)
        ObjectBasedStep.__init__(self, name=f'roi_{method}_fit', class_type=pet_grph.MultiTACGraphicalAnalysis,
                                 init_kwargs=self.init_kwargs, call_kwargs=dict())
    
    def __repr__(self):
        """
        Provides an unambiguous string representation of the ResampleBloodTACStep instance.

        Returns:
            str: A string representation showing how the instance can be recreated.
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(']
        
        in_kwargs = ArgsDict(dict(input_tac_path=self.input_tac_path, roi_tacs_dir=self.roi_tacs_dir,
                                  output_directory=self.output_directory, output_prefix=self.output_prefix,
                                  method=self.init_kwargs['method'],
                                  fit_threshold_in_mins=self.init_kwargs['fit_thresh_in_mins'],
                                  image_rescale=self.init_kwargs['image_rescale'], ))
        
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    @classmethod
    def default_patlak(cls):
        """
        Creates a default instance for Patlak graphical analysis of ROI TACs in a directory using
        :class:`MultiTACGraphicalAnalysis<petpal.kinetic_modeling.graphical_analysis.MultiTACGraphicalAnalysis>`.
        All paths are set to empty strings.

        Returns:
            GraphicalAnalysisStep: A new instance for Patlak graphical analysis.
        """
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='', output_prefix='', method='patlak', )
    
    @classmethod
    def default_logan(cls):
        """
        Creates a default instance for Logan graphical analysis of ROI TACs in a directory using
        :class:`MultiTACGraphicalAnalysis<petpal.kinetic_modeling.graphical_analysis.MultiTACGraphicalAnalysis>`.
        All paths are set to empty strings.

        Returns:
            GraphicalAnalysisStep: A new instance for Logan graphical analysis.
        """
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='', output_prefix='', method='logan', )
    
    @classmethod
    def default_alt_logan(cls):
        """
        Creates a default instance for Alt-Logan (New Plot) graphical analysis of ROI TACs in a directory using
        :class:`MultiTACGraphicalAnalysis<petpal.kinetic_modeling.graphical_analysis.MultiTACGraphicalAnalysis>`.
        All paths are set to empty strings.

        Returns:
            GraphicalAnalysisStep: A new instance for Alt-Logan (New Plot) graphical analysis.
        """
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='', output_prefix='', method='alt_logan', )


class TCMFittingAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    """
    A step for fitting compartment models to TAC data using
    :class:`MultiTACTCMAnalsyis<petpal.kinetic_modeling.tac_fitting.MultiTACTCMAnalsyis>`.

    This class sets up the compartment model fitting, initializes required paths and parameters,
    and provides class methods for creating default steps with common compartment models like 1TCM,
    Serial 2TCM, and Irreversible 2TCM.

    Attributes:
        input_tac_path (str): Path to the input TAC file.
        roi_tacs_dir (str): Directory containing the ROI TAC files.
        output_directory (str): Directory where output files will be saved.
        output_prefix (str): Prefix for the output files.
        compartment_model (str): Compartment model to be used for fitting.
        
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 compartment_model='2tcm-k4zer0',
                 **kwargs):
        """
        Initializes the TCMFittingAnalysisStep with specified parameters.

        Args:
            input_tac_path (str): Path to the input TAC file.
            roi_tacs_dir (str): Directory containing the ROI TAC files.
            output_directory (str): Directory where output files will be saved.
            output_prefix (str): Prefix for the output files.
            compartment_model (str, optional): Compartment model to be used for fitting. Defaults to '2tcm-k4zer0'.
            **kwargs: Additional keyword arguments to be included in initialization for :class:`MultiTACTCMAnalsyis<petpal.kinetic_modeling.tac_fitting.MultiTACTCMAnalsyis>`.
        """
        TACAnalysisStepMixin.__init__(self, input_tac_path=input_tac_path, roi_tacs_dir=roi_tacs_dir,
                                      output_directory=output_directory, output_prefix=output_prefix,
                                      is_ref_tac_based_model=False, compartment_model=compartment_model, **kwargs)
        
        ObjectBasedStep.__init__(self, name=f'roi_{compartment_model}_fit', class_type=tac_fitting.MultiTACTCMAnalsyis,
                                 init_kwargs=self.init_kwargs, call_kwargs=dict())
    
    def __repr__(self):
        """
        Provides an unambiguous string representation of the ResampleBloodTACStep instance.

        Returns:
            str: A string representation showing how the instance can be recreated.
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(']
        
        in_kwargs = ArgsDict(dict(input_tac_path=self.input_tac_path, roi_tacs_dir=self.roi_tacs_dir,
                                  output_directory=self.output_directory, output_prefix=self.output_prefix,
                                  compartment_model=self.init_kwargs['compartment_model']))
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        
        for arg_name in list(self.init_kwargs)[5:]:
            info_str.append(f'{arg_name}={repr(self.init_kwargs[arg_name])},')
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    @classmethod
    def default_1tcm(cls, **kwargs):
        """
        Creates a default instance for 1TCM compartment model fitting using
        :class:`MultiTACTCMAnalsyis<petpal.kinetic_modeling.tac_fitting.MultiTACTCMAnalsyis>`.
        All paths are set to empty-strings.

        Args:
            **kwargs: Additional keyword arguments to be included in the initialization.

        Returns:
            TCMFittingAnalysisStep: A new instance for 1TCM compartment model fitting.
        """
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='', output_prefix='', compartment_model='1tcm',
                   **kwargs)
    
    @classmethod
    def default_serial2tcm(cls, **kwargs):
        """
        Creates a default instance for serial-2TCM compartment model fitting using
        :class:`MultiTACTCMAnalsyis<petpal.kinetic_modeling.tac_fitting.MultiTACTCMAnalsyis>`.
        All paths are set to empty-strings.

        Args:
            **kwargs: Additional keyword arguments to be included in the initialization.

        Returns:
            TCMFittingAnalysisStep: A new instance for serial-2TCM compartment model fitting.
        """
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='', output_prefix='',
                   compartment_model='serial-2tcm', **kwargs)
    
    @classmethod
    def default_irreversible_2tcm(cls, **kwargs):
        """
        Creates a default instance for irreversible-2TCM compartment model fitting using
        :class:`MultiTACTCMAnalsyis<petpal.kinetic_modeling.tac_fitting.MultiTACTCMAnalsyis>`.
        All paths are set to empty-strings.

        Args:
            **kwargs: Additional keyword arguments to be included in the initialization.

        Returns:
            TCMFittingAnalysisStep: A new instance for irreversible-2TCM compartment model fitting.
        """
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='', output_prefix='',
                   compartment_model='2tcm-k4zero', **kwargs)


class RTMFittingAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    """
    A step for fitting reference tissue models (RTMs) to TAC data using
    :class:`MultiTACRTMAnalysis<petpal.kinetic_modeling.rtm_analysis.MultiTACRTMAnalysis>`.

    This class sets up RTM fitting, initializes required paths and parameters.

    Attributes:
        ref_tac_path (str): Path to the reference TAC file.
        roi_tacs_dir (str): Directory containing the ROI TAC files.
        output_directory (str): Directory where output files will be saved.
        output_prefix (str): Prefix for the output files.
        rtm_model (str): Reference tissue model to be used for fitting.
        bounds: Bounds for the fitting parameters. Defaults to None.
        k2_prime: K2' value for the fitting. Defaults to None.
        fit_threshold_in_mins (float): Threshold in minutes for fitting. Defaults to 30.0.
    """
    def __init__(self,
                 ref_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 rtm_model: str,
                 bounds=None,
                 k2_prime=None,
                 fit_threshold_in_mins: float = 30.0, ):
        r"""
        Initializes the RTMFittingAnalysisStep with specified parameters.

        Args:
            ref_tac_path (str): Path to the reference TAC file.
            roi_tacs_dir (str): Directory containing the ROI TAC files.
            output_directory (str): Directory where output files will be saved.
            output_prefix (str): Prefix for the output files.
            rtm_model (str): Reference tissue model to be used for fitting.
            bounds (optional): Bounds for the fitting parameters. Defaults to None.
            k2_prime (optional): :math:`k_{2}^\prime` value for the fitting. Defaults to None.
            fit_threshold_in_mins (float, optional): Threshold in minutes for fitting. Defaults to 30.0.
        """
        TACAnalysisStepMixin.__init__(self, input_tac_path=ref_tac_path, roi_tacs_dir=roi_tacs_dir,
                                      output_directory=output_directory, output_prefix=output_prefix,
                                      is_ref_tac_based_model=True, method=rtm_model, )
        ObjectBasedStep.__init__(self, name=f'roi_{rtm_model}_fit', class_type=pet_rtms.MultiTACRTMAnalysis,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict(bounds=bounds, t_thresh_in_mins=fit_threshold_in_mins,
                                                  k2_prime=k2_prime))
    
    def __repr__(self):
        """
        Provides an unambiguous string representation of the ResampleBloodTACStep instance.

        Returns:
            str: A string representation showing how the instance can be recreated.
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(']
        
        in_kwargs = ArgsDict(dict(ref_tac_path=self.reference_tac_path, roi_tacs_dir=self.roi_tacs_dir,
                                  output_directory=self.output_directory, output_prefix=self.output_prefix,
                                  rtm_model=self.init_kwargs['method'], bounds=self.call_kwargs['bounds'],
                                  k2_prime=self.call_kwargs['k2_prime'],
                                  fit_threshold_in_mins=self.call_kwargs['t_thresh_in_mins']))
        
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)


class ParametricGraphicalAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    """
    A step for performing parametric graphical analysis on TACs using various methods using
    :class:`GraphicalAnalysisParametricImages<petpal.kinetic_modeling.parametric_images.GraphicalAnalysisParametricImages>`

    This class sets up parametric graphical analysis, initializes required paths and parameters,
    and provides class methods for creating default steps with common graphical analysis methods
    like Patlak, Logan, and Alternative Logan.

    Attributes:
        input_tac_path (str): Path to the input TAC file.
        input_image_path (str): Path to the input image file.
        output_directory (str): Directory where output files will be saved.
        output_prefix (str): Prefix for the output files.
        method (str): Graphical analysis method.
        fit_threshold_in_mins (float): Threshold in minutes for fitting. Defaults to 30.0.
    """
    def __init__(self,
                 input_tac_path: str,
                 input_image_path: str,
                 output_directory: str,
                 output_prefix: str,
                 method: str,
                 fit_threshold_in_mins: float = 30.0,
                 image_rescale:float=1.0/37000.):
        """
        Initializes the ParametricGraphicalAnalysisStep with specified parameters.

        Args:
            input_tac_path (str): Path to the input TAC file.
            input_image_path (str): Path to the input image file.
            output_directory (str): Directory where output files will be saved.
            output_prefix (str): Prefix for the output files.
            method (str): Graphical analysis method.
            fit_threshold_in_mins (float, optional): Threshold in minutes for fitting. Defaults to 30.0
            image_rescale (float, optional): Image rescale factor. Defaults to 1.0/37000.0
        """
        TACAnalysisStepMixin.__init__(self, input_tac_path=input_tac_path, pet4D_img_path=input_image_path,
                                      roi_tacs_dir='', output_directory=output_directory, output_prefix=output_prefix,
                                      is_ref_tac_based_model=False, )
        del self.init_kwargs['roi_tacs_dir']
        
        ObjectBasedStep.__init__(self, name=f'parametric_{method}_fit',
                                 class_type=parametric_images.GraphicalAnalysisParametricImage,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict(method_name=method,
                                                  t_thresh_in_mins=fit_threshold_in_mins,
                                                  image_scale=image_rescale))
        self._input_image_path = input_image_path
    
    def __repr__(self):
        """
        Provides an unambiguous string representation of the ResampleBloodTACStep instance.

        Returns:
            str: A string representation showing how the instance can be recreated.
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(']
        
        in_kwargs = ArgsDict(dict(input_tac_path=self.input_tac_path, input_image_path=self.input_image_path,
                                  output_directory=self.output_directory, output_prefix=self.output_prefix,
                                  method=self.call_kwargs['method_name'],
                                  fit_threshold_in_mins=self.call_kwargs['t_thresh_in_mins'], ))
        
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    @property
    def input_image_path(self) -> str:
        """
        Returns the path to the input image.

        Returns:
            str: Path to the input image.
        """
        return self._input_image_path
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        """
        Sets the path to the input image and updates the initialization arguments.

        Args:
            input_image_path (str): Path to the input image.
        """
        self._input_image_path = input_image_path
        self.init_kwargs['pet4D_img_path'] = input_image_path
    
    def set_input_as_output_from(self, sending_step: PreprocStepType) -> None:
        """
        Sets the input paths based on the outputs of a sending preprocessing step.
        
        - Given a :class:`ResampleBloodTACStep<petpal.pipelines.preproc_steps.ResampleBloodTACStep>`,
          we get the input TAC path.
        - Given a :class:`ImageToImageStep<petpal.pipelines.preproc_steps.ImageToImageStep>`,
          we get the output image path.

        Args:
            sending_step (PreprocStepType): The preprocessing step which provides the input paths.
        """
        if isinstance(sending_step, ResampleBloodTACStep):
            self.input_tac_path = sending_step.resampled_tac_path
        elif isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            super().set_input_as_output_from(sending_step)
    
    @classmethod
    def default_patlak(cls):
        """
        Creates a default instance for Patlak parametric graphical analysis using
        :class:`GraphicalAnalysisParametricImages<petpal.kinetic_modeling.parametric_images.GraphicalAnalysisParametricImages>`.
        All non-method arguments are set to empty-strings.
        
        Returns:
            ParametricGraphicalAnalysisStep: A new instance for Patlak parametric graphical analysis.
        """
        return cls(input_tac_path='', input_image_path='', output_directory='', output_prefix='', method='patlak')
    
    @classmethod
    def default_logan(cls):
        """
        Creates a default instance for Logan parametric graphical analysis using
        :class:`GraphicalAnalysisParametricImages<petpal.kinetic_modeling.parametric_images.GraphicalAnalysisParametricImages>`.
        All non-method arguments are set to empty-strings.

        Returns:
            ParametricGraphicalAnalysisStep: A new instance for Logan parametric graphical analysis.
        """
        return cls(input_tac_path='', input_image_path='', output_directory='', output_prefix='', method='logan')
    
    @classmethod
    def default_alt_logan(cls):
        """
        Creates a default instance for Alt-Logan parametric graphical analysis using
        :class:`GraphicalAnalysisParametricImages<petpal.kinetic_modeling.parametric_images.GraphicalAnalysisParametricImages>`.
        All non-method arguments are set to empty-strings.

        Returns:
            ParametricGraphicalAnalysisStep: A new instance for Alt-Logan parametric graphical analysis.
        """
        return cls(input_tac_path='', input_image_path='', output_directory='', output_prefix='', method='alt_logan')
    
KMStepType = Union[GraphicalAnalysisStep,
                   TCMFittingAnalysisStep,
                   ParametricGraphicalAnalysisStep,
                   RTMFittingAnalysisStep]