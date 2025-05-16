import warnings
import copy
from typing import Union

from ..preproc.regional_tac_extraction import write_tacs

from .steps_base import *
from ..preproc.image_operations_4d import SimpleAutoImageCropper, rescale_image
from ..preproc.register import register_pet, warp_pet_to_atlas
from ..preproc.motion_corr import (motion_corr_frames_above_mean_value,
                                   windowed_motion_corr_to_target)
from ..input_function import blood_input
from ..utils.bids_utils import parse_path_to_get_subject_and_session_id, snake_to_camel_case, gen_bids_like_dir_path, gen_bids_like_filepath
from ..utils.image_io import safe_copy_meta
from ..utils.decorators import ANTsImageToANTsImage

class TACsFromSegmentationStep(FunctionBasedStep):
    """
    A step in a processing pipeline for generating Time Activity Curves (TACs) from segmented images.

    This class is specialized for handling the input and output paths related to TAC generation,
    extending the :class:`FunctionBasedStep<petpal.pipelines.steps_base.FunctionBasedStep>` with specific properties
    and methods for TACs. The class uses :func:`write_tacs<petpal.preproc.regional_tac_extraction.write_tacs>` which
    uses segmentation information to generate ROI TACs, and write them to disk.

    Attributes:
        input_image_path (str): Path to the input image.
        segmentation_image_path (str): Path to the segmentation image.
        segmentation_label_map_path (str): Path to the segmentation label map.
        out_tacs_dir (str): Directory where the output TACs will be saved.
        out_tacs_prefix (str): Prefix for the output TACs.
        verbose (bool): Verbosity flag, default is False.

    """
    def __init__(self,
                 input_image_path: str,
                 segmentation_image_path: str,
                 segmentation_label_map_path: str,
                 out_tacs_dir: str,
                 out_tacs_prefix: str,
                 verbose=False) -> None:
        """
        Initializes a TACsFromSegmentationStep with specified parameters.

        Args:
            input_image_path (str): Path to the input image.
            segmentation_image_path (str): Path to the segmentation image.
            segmentation_label_map_path (str): Path to the segmentation label map.
            out_tacs_dir (str): Directory where the output TACs will be saved.
            out_tacs_prefix (str): Prefix for the output TACs.
            verbose (bool): Verbosity flag, default is False.
        """
        super().__init__(name='write_roi_tacs', function=write_tacs, input_image_path=input_image_path,
                         segmentation_image_path=segmentation_image_path, label_map_path=segmentation_label_map_path,
                         out_tac_dir=out_tacs_dir, out_tac_prefix=out_tacs_prefix,
                         verbose=verbose, )
        self._input_image = input_image_path
        self._segmentation_image = segmentation_image_path
        self._segmentation_label_map = segmentation_label_map_path
        self._out_tacs_path = out_tacs_dir
        self._out_tacs_prefix = out_tacs_prefix
        self.verbose = verbose
    
    def __repr__(self):
        """
        Provides an unambiguous string representation of the TACsFromSegmentationStep instance.

        Returns:
            str: A string representation showing how the instance can be recreated.
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(']
        
        in_kwargs = ArgsDict(
            dict(input_image_path=self.input_image_path, segmentation_image_path=self.segmentation_image_path,
                 segmentation_label_map_path=self.segmentation_label_map_path, out_tacs_dir=self.out_tacs_dir,
                 out_tacs_prefix=self.out_tacs_prefix, verbose=self.verbose))
        
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    @property
    def segmentation_image_path(self):
        """
        Gets the path to the segmentation image.

        Returns:
            str: The path to the segmentation image.
        """
        return self._segmentation_image
    
    @segmentation_image_path.setter
    def segmentation_image_path(self, segmentation_image_path: str):
        """
        Sets the path to the segmentation image and updates the function arguments.

        Args:
            segmentation_image_path (str): The new path to the segmentation image.
        """
        self._segmentation_image = segmentation_image_path
        self.kwargs['segmentation_image_path'] = segmentation_image_path
    
    @property
    def segmentation_label_map_path(self):
        """
        Gets the path to the segmentation label map.

        Returns:
            str: The path to the segmentation label map.
        """
        return self._segmentation_label_map
    
    @segmentation_label_map_path.setter
    def segmentation_label_map_path(self, segmentation_label_map_path: str):
        """
        Sets the path to the segmentation label map and updates the function arguments.

        Args:
            segmentation_label_map_path (str): The new path to the segmentation label map.
        """
        self._segmentation_label_map = segmentation_label_map_path
        self.kwargs['label_map_path'] = segmentation_label_map_path
    
    @property
    def out_tacs_dir(self):
        """
        Gets the directory where the output TACs will be saved.

        Returns:
            str: The output directory path.
        """
        return self._out_tacs_path
    
    @out_tacs_dir.setter
    def out_tacs_dir(self, out_tacs_path: str):
        """
        Sets the directory where the output TACs will be saved and updates the function arguments.

        Args:
            out_tacs_path (str): The new output directory path.
        """
        self.kwargs['out_tac_dir'] = out_tacs_path
        self._out_tacs_path = out_tacs_path
    
    @property
    def out_tacs_prefix(self):
        """
        Gets the prefix for the output TACs.

        Returns:
            str: The prefix for the output TACs.
        """
        return self._out_tacs_prefix
    
    @out_tacs_prefix.setter
    def out_tacs_prefix(self, out_tacs_prefix: str):
        """
        Sets the prefix for the output TACs and updates the function arguments.

        Args:
            out_tacs_prefix (str): The new prefix for the output TACs.
        """
        self.kwargs['out_tac_prefix'] = out_tacs_prefix
        self._out_tacs_prefix = out_tacs_prefix
    
    @property
    def out_path_and_prefix(self):
        """
        Gets the output directory path and prefix as a tuple.

        Returns:
            tuple: A tuple containing the output directory path and prefix.
        """
        return self._out_tacs_path, self._out_tacs_prefix
    
    @out_path_and_prefix.setter
    def out_path_and_prefix(self, out_dir_and_prefix: str):
        """
        Sets both the output directory path and prefix.

        Args:
            out_dir_and_prefix (tuple): A tuple containing the output directory and prefix.

        Raises:
            ValueError: If the provided value is not a tuple with two items.
        """
        try:
            out_dir, out_prefix = out_dir_and_prefix
        except ValueError:
            raise ValueError("Pass a tuple with two items: `(out_dir, out_prefix)`")
        else:
            self.out_tacs_dir = out_dir
            self.out_tacs_prefix = out_prefix
    
    @property
    def input_image_path(self):
        """
        Gets the path to the input image.

        Returns:
            str: The path to the input image.
        """
        return self._input_image
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        """
        Sets the path to the input image and updates the function arguments.

        Args:
            input_image_path (str): The new path to the input image.
        """
        self.kwargs['input_image_path'] = input_image_path
        self._input_image = input_image_path
    
    def set_input_as_output_from(self, *sending_steps):
        """
        Sets the input image path based on the output from a specified sending step.

        Args:
            sending_step: The step from which to derive the input image path.
        """
        assert len(sending_steps) == 1, "TACsFromSegmentationStep can only receive a single sending step."
        for sending_step in sending_steps:
            if isinstance(sending_step, ImageToImageStep):
                self.input_image_path = sending_step.output_image_path
            else:
                super().set_input_as_output_from(sending_step)
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type: str, suffix: str=None, ext: str=None, **extra_desc):
        """
        Infers output directory and prefix for TACs based on the input image path.

        Args:
            out_dir (str): Directory where the outputs will be saved.
            der_type (str): Type of derivatives.
            suffix (str, optional): Suffix for the output files. Defaults to None.
            ext (str, optional): Extension for the output files. Defaults to None.
            **extra_desc: Additional descriptive parameters.
        """
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_image_path)
        outpath = gen_bids_like_dir_path(sub_id=sub_id, ses_id=ses_id, sup_dir=out_dir, modality='tacs')
        self.out_tacs_dir = outpath
        step_name_in_camel_case = snake_to_camel_case(self.name)
        self.out_tacs_prefix = f'sub-{sub_id}_ses-{ses_id}_desc-{step_name_in_camel_case}'
    
    @classmethod
    def default_write_tacs_from_segmentation_rois(cls, **overrides):
        """
        Provides a class method to create an instance with default parameters. All paths
        are set to empty strings, and `verbose=False`.

        Returns:
            TACsFromSegmentationStep: A new instance with default parameters.
        """

        defaults = dict(input_image_path='', segmentation_image_path='', segmentation_label_map_path='',
                        out_tacs_dir='', out_tacs_prefix='', verbose=False)
        override_dict = defaults | overrides

        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f'Invalid override: {err}. Using default instance instead.', stacklevel=2)
            return cls(**defaults)
    

class ResampleBloodTACStep(FunctionBasedStep):
    """
    A step in a processing pipeline for resampling blood Time Activity Curves (TACs) based on PET image timings.

    This class facilitates the resampling of blood TACs to match the scanner's time frames, providing properties and
    methods for handling the input and output paths, as well as the resampling thresholds. This class uses
    :func:`resample_blood_data_on_scanner_times<petpal.input_function.blood_input.resample_blood_data_on_scanner_times>`
    to perform the resampling.

    Attributes:
        raw_blood_tac_path (str): Path to the input raw blood TAC file.
        input_image_path (str): Path to the 4D-PET image file.
        resampled_tac_path (str): Path where the resampled TAC will be saved.
        lin_fit_thresh_in_mins (float): Threshold in minutes for linear fitting.

    """
    def __init__(self,
                 input_raw_blood_tac_path: str,
                 input_image_path: str,
                 out_tac_path: str,
                 lin_fit_thresh_in_mins=30.0,
                 rescale_constant:float = 37000.0):
        """
        Initializes a ResampleBloodTACStep with specified parameters.

        Args:
            input_raw_blood_tac_path (str): Path to the input raw blood TAC file.
            input_image_path (str): Path to the PET image file.
            out_tac_path (str): Path where the resampled TAC will be saved.
            lin_fit_thresh_in_mins (float): Threshold in minutes for linear fitting.
        """
        super().__init__(name='resample_PTAC_on_scanner',
                         function=blood_input.resample_blood_data_on_scanner_times,
                         blood_tac_path=input_raw_blood_tac_path,
                         reference_4dpet_img_path=input_image_path,
                         out_tac_path=out_tac_path,
                         lin_fit_thresh_in_mins=lin_fit_thresh_in_mins,
                         rescale_constant=rescale_constant)
        self._raw_blood_tac_path = input_raw_blood_tac_path
        self._input_image_path = input_image_path
        self._resampled_tac_path = out_tac_path
        self.lin_fit_thresh_in_mins = lin_fit_thresh_in_mins
        self.rescale_constant = rescale_constant
    
    def __repr__(self):
        """
        Provides an unambiguous string representation of the ResampleBloodTACStep instance.

        Returns:
            str: A string representation showing how the instance can be recreated.
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(']
        
        in_kwargs = ArgsDict(
                dict(input_raw_blood_tac_path=self.raw_blood_tac_path, input_image_path=self.input_image_path,
                     out_tac_path=self.resampled_tac_path, lin_fit_thresh_in_mins=self.lin_fit_thresh_in_mins))
        
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    @property
    def raw_blood_tac_path(self):
        """
        Gets the path to the input raw blood TAC file.

        Returns:
            str: The path to the input raw blood TAC file.
        """
        return self._raw_blood_tac_path
    
    @raw_blood_tac_path.setter
    def raw_blood_tac_path(self, raw_blood_tac_path):
        """
        Sets the path to the input raw blood TAC file and updates the function arguments.

        Args:
            raw_blood_tac_path (str): The new path to the raw blood TAC file.
        """
        self.kwargs['blood_tac_path'] = raw_blood_tac_path
        self._raw_blood_tac_path = raw_blood_tac_path
    
    @property
    def input_image_path(self):
        """
        Gets the path to the PET image file.

        Returns:
            str: The path to the PET image file.
        """
        return self._input_image_path
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        """
        Sets the path to the PET image file and updates the function arguments.

        Args:
            input_image_path (str): The new path to the PET image file.
        """
        self.kwargs['reference_4dpet_img_path'] = input_image_path
        self._input_image_path = input_image_path
    
    @property
    def resampled_tac_path(self):
        """
        Gets the path where the resampled TAC will be saved.

        Returns:
            str: The path where the resampled TAC will be saved.
        """
        return self._resampled_tac_path
    
    @resampled_tac_path.setter
    def resampled_tac_path(self, resampled_tac_path):
        """
        Sets the path where the resampled TAC will be saved and updates the function arguments.

        Args:
            resampled_tac_path (str): The new path where the resampled TAC will be saved.
        """
        self.kwargs['out_tac_path'] = resampled_tac_path
        self._resampled_tac_path = resampled_tac_path
    
    def set_input_as_output_from(self, *sending_steps):
        """
        Sets the input image path based on the output from a specified sending step.

        Args:
            sending_step: The step from which to derive the input image path.
        """
        assert len(sending_steps) == 1, "ResampleBloodTACStep should only receive a single sending step."
        if isinstance(sending_steps[0], ImageToImageStep):
            self.input_image_path = sending_steps[0].output_image_path
        else:
            super().set_input_as_output_from(sending_steps[0])
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type, suffix='blood', ext='.tsv', **extra_desc):
        """
        Infers the output file path for resampled TAC based on the input raw blood TAC path.

        Args:
            out_dir (str): Directory where the outputs will be saved.
            der_type: Type of derivatives.
            suffix (str, optional): Suffix for the output files. Defaults to 'blood'.
            ext (str, optional): Extension for the output files. Defaults to '.tsv'.
            **extra_desc: Additional descriptive parameters.
        """
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.raw_blood_tac_path)
        filepath = gen_bids_like_filepath(sub_id=sub_id, ses_id=ses_id, bids_dir=out_dir, modality='preproc',
                                          suffix=suffix, ext=ext, desc='OnScannerFrameTimes')
        self.resampled_tac_path = filepath
    
    @classmethod
    def default_resample_blood_tac_on_scanner_times(cls):
        """
        Provides a class method to create an instance with default parameters. All paths are set to empty-strings
        and the fitting threshold is set to 30 minutes.

        Returns:
            ResampleBloodTACStep: A new instance with default parameters.
        """
        return cls(input_raw_blood_tac_path='', input_image_path='', out_tac_path='', lin_fit_thresh_in_mins=30.0)


class ImageToImageStep(FunctionBasedStep):
    """
    A step in a processing pipeline for processing and transforming image files. This class handles input and output
    image paths, executes image transformation functions, and provides methods for setting inputs from other steps
    and inferring output paths.
    
    .. attention::
    
       The passed function must have the following arguments order:
       ``func(input_image, output_image, *args, **kwargs)`` where ``input_image`` and ``output_image``
       can be named something else. The first argument must be an input image path, and the second argument
       must be an output image path.

    Attributes:
        input_image_path (str): Path to the input image file.
        output_image_path (str): Path to the output image file.
        
    See Also:
        - :meth:`default_threshold_cropping`
        - :meth:`default_moco_frames_above_mean`
        - :meth:`default_register_pet_to_t1`

    """
    def __init__(self,
                 name: str,
                 function: Callable,
                 input_image_path: str,
                 output_image_path: str,
                 *args,
                 **kwargs) -> None:
        """
        Initializes an ImageToImageStep with specified parameters.

        Args:
            name (str): The name of the step.
            function (Callable): The function to execute.
            input_image_path (str): Path to the input image file.
            output_image_path (str): Path to the output image file.
            *args: Additional positional arguments for the transformation function.
            **kwargs: Additional keyword arguments for the transformation function.
        
        Notes:
            The passed function must have the following arguments order:
            ``func(input_image, output_image, *args, **kwargs)`` where ``input_image`` and ``output_image``
            can be named something else. The first argument must be an input image path, and the second argument
            must be an output image path.
        
        """
        super().__init__(name, function, *(input_image_path, output_image_path, *args), **kwargs)
        self.input_image_path = copy.copy(self.args[0])
        self.output_image_path = copy.copy(self.args[1])
        self.args = self.args[2:]
    
    def execute(self, copy_meta_file: bool = True) -> None:
        """
        Executes the function and optionally copies meta-data information using
        :func:`safe_copy_meta<petpal.utils.image_io.safe_copy_meta>`.

        Args:
            copy_meta_file (bool): Whether to copy meta information from input to output image. Defaults to True.
            
        Notes:
            Function must have the following arguments order: ``(input_image_path, output_image_path, *args, **kwargs)``
            where ``input_image`` and ``output_image`` are abritrary names.
            
        """
        print(f"(Info): Executing {self.name}")
        self.function(self.input_image_path, self.output_image_path, *self.args, **self.kwargs)
        if copy_meta_file:
            safe_copy_meta(input_image_path=self.input_image_path, out_image_path=self.output_image_path)
        print(f"(Info): Finished {self.name}")
    
    def __str__(self):
        """
        Provides a string representation of the ImageToImageStep instance.

        Returns:
            str: A string representation of the instance.
        """
        io_dict = ArgsDict({'input_image_path': self.input_image_path,
                            'output_image_path': self.output_image_path
                            })
        sup_str_list = super().__str__().split('\n')
        args_ind = sup_str_list.index("Arguments Passed:")
        sup_str_list.insert(args_ind, f"Input & Output Paths:\n{io_dict}")
        def_args_ind = sup_str_list.index("Default Arguments:")
        sup_str_list.pop(def_args_ind + 1)
        if not hasattr(self.function, '__wrapped__'):
            sup_str_list.pop(def_args_ind + 1)
        
        return "\n".join(sup_str_list)
    
    def set_input_as_output_from(self, *sending_steps) -> None:
        """
        Sets the input image path based on the output from a specified sending step.

        Args:
            sending_step (FunctionBasedStep): The step from which to derive the input image path.
        """
        assert len(sending_steps) == 1, "ImageToImageStep must have 1 sending step."
        if isinstance(sending_steps[0], ImageToImageStep):
            self.input_image_path = sending_steps[0].output_image_path
        else:
            super().set_input_as_output_from(sending_steps[0])
    
    def can_potentially_run(self):
        """
        Checks if the step can potentially run based on input and output paths. Checks if all path related
        arguments are non-empty strings.

        Returns:
            bool: True if the step can potentially run, False otherwise.
        """
        input_img_non_empty_str = False if self.input_image_path == '' else True
        output_img_non_empty_str = False if self.output_image_path == '' else True
        return super().can_potentially_run() and input_img_non_empty_str and output_img_non_empty_str
    
    def infer_outputs_from_inputs(self,
                                  out_dir: str,
                                  der_type='preproc',
                                  suffix: str = 'pet',
                                  ext: str = '.nii.gz',
                                  **extra_desc):
        """
        Infers the output file path based on the input image path and other parameters.

        Args:
            out_dir (str): Directory where the outputs will be saved.
            der_type (str): Type of derivatives. Defaults to 'preproc'.
            suffix (str, optional): Suffix for the output files. Defaults to 'pet'.
            ext (str, optional): Extension for the output files. Defaults to '.nii.gz'.
            **extra_desc: Additional descriptive parameters.
        """
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_image_path)
        step_name_in_camel_case = snake_to_camel_case(self.name)
        filepath = gen_bids_like_filepath(sub_id=sub_id, ses_id=ses_id, suffix=suffix, bids_dir=out_dir,
                                          modality=der_type, ext=ext, desc=step_name_in_camel_case, **extra_desc)
        self.output_image_path = filepath
    
    @classmethod
    def default_threshold_cropping(cls, name: str = 'thresh_crop',  **overrides):
        """
        Creates a default instance for threshold cropping using :class:`SimpleAutoImageCropper<petpal.preproc.image_operations_4d.SimpleAutoImageCropper>`.
        All paths are empty-strings.

        Args:
            name (str): Name of the step. Defaults to 'thresh_crop'.
            **overrides: Override default parameters.

        Returns:
            ImageToImageStep: A new instance for threshold cropping.
        """
        defaults = dict(name=name, function=SimpleAutoImageCropper, input_image_path='',
                        output_image_path='', )
        override_dict = defaults | overrides
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)
    
    @classmethod
    def default_moco_frames_above_mean(cls, name: str = 'moco_frames_above_mean', verbose=False, **overrides):
        """
        Creates a default instance for motion correction frames above mean value using
        :func:`motion_corr_frames_above_mean_value<petpal.preproc.motion_corr.motion_corr_frames_above_mean_value>`.
        All paths are empty-strings.

        Args:
            name (str): Name of the step. Defaults to 'moco_frames_above_mean'
            verbose (bool): Whether to run in verbose mode.
            **overrides: Override default parameters.

        Returns:
            ImageToImageStep: A new instance for motion correction frames above mean value.
        """
        defaults = dict(name=name, function=motion_corr_frames_above_mean_value,
                        input_image_path='', output_image_path='', motion_target_option='mean_image', verbose=verbose,
                        half_life=None, )
        override_dict = defaults | overrides
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)

    @classmethod
    def default_windowed_moco(cls, name: str = 'windowed_moco', verbose=False, **overrides):
        """
        Creates a default instance for motion correction of frames using a windowed strategy.
        See :func:`windowed_motion_corr_to_target<petpal.preproc.motion_corr.windowed_motion_corr_to_target>`
        for more details. All paths are empty-strings.

        Args:
            name (str): Name of the step. Defaults to 'windowed_moco'.
            verbose:
            **overrides:

        Returns:
            ImageToImageStep: A new instance for windowed motion correction of frames.
        """
        defaults = dict(name=name, function=windowed_motion_corr_to_target,
                        input_image_path='', output_image_path='',
                        motion_target_option='weighted_series_sum', w_size=60.0,
                        verbose=verbose)
        override_dict = defaults | overrides
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)

    @classmethod
    def default_register_pet_to_t1(cls, name:str = 'register_pet_to_t1', reference_image_path='', half_life:float=None, verbose=False, **overrides):
        """
        Creates a default instance for registering PET to T1 image using :func:`register_pet<petpal.preproc.register.register_pet>`.
        All paths are empty-strings.

        Args:
            name (str): Name of the step. Defaults to 'register_pet_to_t1'
            reference_image_path (str): Path to the reference image.
            half_life (float): Half-life value, in seconds, for the radiotracer. Used to
                generate a weighted_series_sum image.
            verbose (bool): Whether to run in verbose mode.
            **overrides: Override default parameters.

        Returns:
            ImageToImageStep: A new instance for registering PET to T1 image.

        """
        defaults = dict(name=name, function=register_pet, input_image_path='', output_image_path='',
                        reference_image_path=reference_image_path, motion_target_option='weighted_series_sum',
                        verbose=verbose, half_life=half_life)
        override_dict = defaults | overrides
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)

    @classmethod
    def default_rescale_image(cls, name: str = 'rescale_image', **overrides):
        r"""Creates a default step-instance for rescaling an image using :class:`rescale_image<petpal.preproc.image_operations_4d.rescale_image>`.

        The defaults for this step will divide the input image by 37000.0 which is usually done to go
        from kBq/ml to nCi/ml.

        Notes:
             The function :class:`rescale_image<petpal.preproc.image_operations_4d.rescale_image>` is wrapped using
             :func:`ANTsImageToANTsImage<petpal.utils.decorators.ANTsImageToANTsImage>` since steps require input and output
             paths.

        Args:
            name (str): Name of the step. Defaults to 'rescale_image'.
            **overrides:

        Returns:
            ImageToImageStep: A new step instance for rescaling the input image.
        """
        defaults = dict(name=name, function=ANTsImageToANTsImage(rescale_image),
                        input_image_path='', output_image_path='',
                        rescale_constant=37000.0, op='/')
        override_dict = defaults | overrides

        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)

    @classmethod
    def default_warp_pet_to_atlas(cls, name: str = 'warp_pet_to_atlas', anat_image_path: str = '', atlas_image_path: str = '', **overrides):
        """Creates a default step instance for warping a PET image to an atlas using :func:`~petpal.preproc.register.warp_pet_to_atlas`.

        Note:
            The defaults for this step do not include the *anat_image_path* or *atlas_image_path*, as these will depend
            on the pipeline construction. These paths will need to be set before a pipeline will be able to run.

        Args:
            name (str): Name of the step. Defaults to 'warp_pet_to_atlas'.
            anat_image_path (str): Path to anatomical image used to compute transform to atlas space. Defaults to an
                empty string ''.
            atlas_image_path (str): Path to atlas image to which pet will be warped
                (indirectly, via the anatomical image).
            **overrides: Override default parameters

        Returns:
            ImageToImageStep:
                A new step instance for warping the pet image to atlas space.
        """
        defaults = dict(name=name, function=ANTsImageToANTsImage(warp_pet_to_atlas),
                        input_image_path='', output_image_path='', anat_image_path=anat_image_path,
                        atlas_image_path=atlas_image_path, )
        override_dict = defaults | overrides

        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)

class ImagePairToArrayStep(FunctionBasedStep):
    """
    A step in a processing pipeline for transforming two input image files into an output array.

    This class extends the :class:`FunctionBasedStep<petpal.pipelines.steps_base.FunctionBasedStep>` and is designed
    for tasks that require combining or processing information from two image inputs and outputing an array e.g.
    getting the mean activity from a 4D-PET image given a voxel mask image. It handles input image paths, output
    array paths, and provides methods for setting inputs from connected pipeline steps or for inferring output
    array paths.

    .. attention::

       The passed function must have the following arguments order:
       ``func(input_image, second_image, output_array, *args, **kwargs)`` where ``input_image`` and ``output_image``
       can be named something else. The first argument must be an input image path, the second argument must be the
       path to the second image, and the third argument must be an output array path.

    Attributes:
        input_image_path (str): Path to the first input image file.
        second_image_path (str): Path to the second input image file.
        output_array_path (str): Path to the output array file.

    """
    def __init__(self,
                 name: str,
                 function: Callable,
                 input_image_path: str,
                 second_image_path: str,
                 output_array_path: str,
                 *args,
                 **kwargs):
        """
        Initializes an ImagePairToArrayStep with specified parameters.

        Args:
            name (str): The name of the step.
            function (Callable): The function that uses the two input images and outputs an array.
            input_image_path (str): Path to the first input image file.
            second_image_path (str): Path to the second input image file.
            output_array_path (str): Path to the output array file.
            *args: Additional positional arguments for the processing function.
            **kwargs: Additional keyword arguments for the processing function.

        Notes:
            The passed function (``func``) must have the following arguments order:
            ``func(input_image, second_image, output_array, *args, **kwargs)`` where ``input_image``, ``second_image``,
            and ``output_array`` can be named something else. The first argument must be an input image path,
            the second argument must be the path to the second image, and the third argument must be an output
            array path.

        """
        super().__init__(name, function,
                         *(input_image_path, output_array_path,
                           second_image_path, *args),
                         **kwargs)
        self.input_image_path = copy.copy(self.args[0])
        self.second_image_path = copy.copy(self.args[1])
        self.output_array_path = copy.copy(self.args[2])
        self.args = copy.copy(self.args[3:])

    def execute(self):
        """
        Executes the function to process the two input images into an output array.

        The specified function will be called with the paths to the two input images, the output array,
        any additional arguments, and keyword arguments.

        """
        print(f"(Info): Executing {self.name}")
        self.function(self.input_image_path,
                      self.second_image_path,
                      self.output_array_path,
                      *self.args,
                      **self.kwargs)
        print(f"(Info): Finished {self.name}")

    def set_input_as_output_from(self, *sending_steps) -> None:
        """
        Sets the input image paths based on the output paths from other steps in the pipeline.
        The first sending step will set the input image path, and the second sending step will
        set the second image path.

        Args:
            sending_steps (tuple[FunctionBasedStep]): Two pipeline steps whose outputs will be used
                as the input image path and second image input path.

        Raises:
            AssertionError: If the number of provided sending steps is not exactly two.
        """
        assert len(sending_steps) == 2, "ImagePairToArrayStep must have 2 sending ImageToImageStep steps."
        if isinstance(sending_steps[0], ImageToImageStep):
            self.input_image_path = sending_steps[0].output_image_path
        else:
            super().set_input_as_output_from(sending_steps[0])
        if isinstance(sending_steps[1], ImageToImageStep):
            self.second_image_path = sending_steps[1].output_image_path
        else:
            super().set_input_as_output_from(sending_steps[1])

    def infer_outputs_from_inputs(self,
                                  out_dir: str,
                                  der_type: str = 'tacs',
                                  suffix: str = 'tac',
                                  ext: str = '.tsv',
                                  **extra_desc):
        """
        Infers the output array path based on the inputs and specified parameters.

        This method generates a BIDS-like derivatives filepath for the output based on the subject and
        session IDs extracted from the input image path.

        Args:
            out_dir (str): Directory where the output array will be saved.
            der_type (str, optional): Type of derivative. Will set the sub-directory in `out_dir`. Defaults to 'tacs'.
            suffix (str, optional): Suffix for the output filename. Defaults to 'tac'.
            ext (str, optional): File extension for the output file. Defaults to '.tsv'.
            **extra_desc: Additional descriptive parameters for the output filename.
        """
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_image_path)
        step_name_in_camel_case = snake_to_camel_case(self.name)
        filepath = gen_bids_like_filepath(sub_id=sub_id, ses_id=ses_id, suffix=suffix, bids_dir=out_dir,
                                          modality=der_type, ext=ext, desc=step_name_in_camel_case, **extra_desc)
        self.output_array_path = filepath


PreprocStepType = Union[TACsFromSegmentationStep, ResampleBloodTACStep, ImageToImageStep, ImagePairToArrayStep]