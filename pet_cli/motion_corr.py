"""
Provides methods to motion correct 4D PET data. Includes method
:meth:`determine_motion_target`, which produces a flexible target based on the
4D input data to optimize contrast when computing motion correction or
registration.
"""
import os
import re
import tempfile
from typing import Union
import ants
import nibabel
import numpy as np
from . import image_io, image_operations_4d

weighted_series_sum = image_operations_4d.weighted_series_sum


def determine_motion_target(motion_target_option: Union[str,tuple,list],
                            input_image_4d_path: str=None,
                            half_life: float=None) -> str:
    """
    Produce a motion target given the ``motion_target_option`` from a method
    running registrations on PET, i.e. :meth:`motion_correction` or
    :meth:`register_pet`.

    The motion target option can be a string or a tuple. If it is a string,
    then if this string is a file, use the file as the motion target.

    If it is the option ``weighted_series_sum``, then run
    :meth:`weighted_series_sum` and return the output path.

    If it is a tuple, run a weighted sum on the PET series on a range of 
    frames. The elements of the tuple are treated as times in seconds, counted
    from the time of the first frame, i.e. (0,300) would average all frames 
    from the first to the frame 300 seconds later. If the two elements are the
    same, returns the one frame closest to the entered time.

    Args:
        motion_target_option (str | tuple | list): Determines how the method behaves,
            according to the above description. Can be a file, a method
            ('weighted_series_sum'), or a tuple range e.g. (0,600).
        input_image_4d_path (str): Path to the PET image. This is intended to
            be supplied by the parent method employing this function. Default
            value None.
        half_life (float): Half life of the radiotracer used in the image
            located at ``input_image_4d_path``. Only used if a calculation is
            performed.
    
    Returns:
        out_image_file (str): File to use as a target to compute
            transformations on.

    Raises:
        ValueError: If ``motion_target_option`` does not match an acceptable option.
        TypeError: If start and end time are incompatible with ``float`` type.
    """
    if isinstance(motion_target_option,str):
        if os.path.exists(motion_target_option):
            return motion_target_option

        elif motion_target_option=='weighted_series_sum':        
            out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
            weighted_series_sum(input_image_4d_path=input_image_4d_path,
                                out_image_path=out_image_file,
                                half_life=half_life,
                                verbose=False)
            return out_image_file

        else:
            raise ValueError("motion_target_option did not match a file or 'weighted_series_sum'")

    elif isinstance(motion_target_option,tuple) or isinstance(motion_target_option,list):

        start_time = motion_target_option[0]
        end_time = motion_target_option[1]

        try:
            float(start_time)
            float(end_time)
        except:
            raise TypeError('Start time and end time of calculation must be '
                            'able to be cast into float! Provided values are '
                            f"{start_time} and {end_time}.")

        out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
        weighted_series_sum(input_image_4d_path=input_image_4d_path,
                            out_image_path=out_image_file,
                            half_life=half_life,
                            verbose=False,
                            start_time=float(start_time),
                            end_time=float(end_time))

        return out_image_file

    else:
        raise ValueError('motion_target_option did not match str or tuple type.')


def motion_corr(input_image_4d_path: str,
                motion_target_option: Union[str,tuple],
                out_image_path: str,
                verbose: bool,
                type_of_transform: str='DenseRigid',
                half_life: float=None,
                **kwargs) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Correct PET image series for inter-frame motion. Runs rigid motion
    correction module from Advanced Normalisation Tools (ANTs) with default
    inputs.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be motion corrected.
        motion_target_option (str | tuple): Target image for computing
            transformation. See :meth:`determine_motion_target`.
        out_image_path (str): Path to a .nii or .nii.gz file to which the
            motion corrected PET series is written.
        verbose (bool): Set to ``True`` to output processing information.
        type_of_transform (str): Type of transform to perform on the PET image,
            must be one of antspy's transformation types, i.e. 'DenseRigid' or
            'Translation'. Any transformation type that uses >6 degrees of
            freedom is not recommended, use with caution. See 
            :py:func:`ants.registration`.
        half_life (float): Half life of the PET radioisotope in seconds. Used
            for certain settings of ``motion_target_option``.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`ants.motion_correction`.

    Returns:
        pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy
            array.
        pet_moco_params (list[str]): List of ANTS registration files applied to
            each frame.
        pet_moco_fd (list[float]): List of framewise displacement measure
            corresponding to each frame transform.
    """
    pet_ants = ants.image_read(input_image_4d_path)
    motion_target_image_path = determine_motion_target(motion_target_option=motion_target_option,
                                                       input_image_4d_path=input_image_4d_path,
                                                       half_life=half_life)

    motion_target_image = ants.image_read(motion_target_image_path)
    pet_moco_ants_dict = ants.motion_correction(image=pet_ants,
                                                fixed=motion_target_image,
                                                type_of_transform=type_of_transform,
                                                **kwargs)
    if verbose:
        print('(ImageOps4D): motion correction finished.')

    pet_moco_ants = pet_moco_ants_dict['motion_corrected']
    pet_moco_params = pet_moco_ants_dict['motion_parameters']
    pet_moco_fd = pet_moco_ants_dict['FD']
    pet_moco_np = pet_moco_ants.numpy()
    pet_moco_nibabel = ants.to_nibabel(pet_moco_ants)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(
        input_image_4d_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)

    nibabel.save(pet_moco_nibabel, out_image_path)
    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")
    return pet_moco_np, pet_moco_params, pet_moco_fd
