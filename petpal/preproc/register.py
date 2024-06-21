"""
Provides tools to register PET images to anatomical or atlas space. Wrapper for
ANTs and FSL registration software.
"""
import re
from typing import Union
import fsl.wrappers
import ants
from ..utils import image_io
from . import motion_corr

determine_motion_target = motion_corr.determine_motion_target

def register_pet(input_reg_image_path: str,
                 reference_image_path: str,
                 motion_target_option: Union[str,tuple],
                 out_image_path: str,
                 verbose: bool,
                 type_of_transform: str='DenseRigid',
                 half_life: str=None,
                 **kwargs):
    """
    Computes and runs rigid registration of 4D PET image series to 3D anatomical image, typically
    a T1 MRI. Runs rigid registration module from Advanced Normalisation Tools (ANTs) with  default
    inputs. Will upsample PET image to the resolution of anatomical imaging.

    Args:
        input_reg_image_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be registered to anatomical space.
        reference_image_path (str): Path to a .nii or .nii.gz file containing a 3D
            anatomical image to which PET image is registered.
        motion_target_option (str | tuple): Target image for computing
            transformation. See :meth:`determine_motion_target`.
        type_of_transform (str): Type of transform to perform on the PET image, must be one of antspy's
            transformation types, i.e. 'DenseRigid' or 'Translation'. Any transformation type that uses
            >6 degrees of freedom is not recommended, use with caution. See :py:func:`ants.registration`.
        out_image_path (str): Path to a .nii or .nii.gz file to which the registered PET series
            is written.
        verbose (bool): Set to ``True`` to output processing information.
        kwargs (keyword arguments): Additional arguments passed to :py:func:`ants.registration`.
    """
    motion_target = determine_motion_target(motion_target_option=motion_target_option,
                                                input_image_4d_path=input_reg_image_path,
                                                half_life=half_life)
    motion_target_image = ants.image_read(motion_target)
    mri_image = ants.image_read(reference_image_path)
    pet_image_ants = ants.image_read(input_reg_image_path)
    xfm_output = ants.registration(moving=motion_target_image,
                                   fixed=mri_image,
                                   type_of_transform=type_of_transform,
                                   write_composite_transform=True,
                                   **kwargs)
    if verbose:
        print(f'Registration computed transforming image {motion_target} to '
              f'{reference_image_path} space')

    if pet_image_ants.dimension==4:
        dim = 3
    else:
        dim = 0

    xfm_apply = ants.apply_transforms(moving=pet_image_ants,
                                      fixed=mri_image,
                                      transformlist=xfm_output['fwdtransforms'],
                                      imagetype=dim)
    if verbose:
        print(f'Registration applied to {input_reg_image_path}')

    ants.image_write(xfm_apply, out_image_path)
    if verbose:
        print(f'Transformed image saved to {out_image_path}')

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_reg_image_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)


def warp_pet_atlas(input_image_path: str,
                   anat_image_path: str,
                   atlas_image_path: str,
                   out_image_path: str,
                   verbose: bool,
                   type_of_transform: str='SyN',
                   **kwargs):
    """
    Compute and apply a warp on a 3D or 4D image in anatomical space
    to atlas space using ANTs.

    Args:
        input_image_path (str): Image to be registered to atlas. Must be in
            anatomical space. May be 3D or 4D.
        anat_image_path (str): Image used to compute registration to atlas space.
        atlas_image_path (str): Atlas to which input image is warped.
        out_image_path (str): Path to which warped image is saved.
        type_of_transform (str): Type of non-linear transform applied to input 
            image using :py:func:`ants.registration`.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`ants.registration`.
    
    Returns:
        xfm_to_apply (list[str]): The computed transforms, saved to a temp dir.
    """
    anat_image_ants = ants.image_read(anat_image_path)
    atlas_image_ants = ants.image_read(atlas_image_path)

    anat_atlas_xfm = ants.registration(fixed=atlas_image_ants,
                                       moving=anat_image_ants,
                                       type_of_transform=type_of_transform,
                                       write_composite_transform=True,
                                       **kwargs)
    xfm_to_apply = anat_atlas_xfm['fwdtransforms']
    if verbose:
        print(f'Xfms located at: {xfm_to_apply}')

    pet_image_ants = ants.image_read(input_image_path)

    if pet_image_ants.dimension==4:
        dim = 3
    else:
        dim = 0

    pet_atlas_xfm = ants.apply_transforms(fixed=atlas_image_ants,
                                          moving=pet_image_ants,
                                          transformlist=xfm_to_apply,verbose=True,
                                          imagetype=dim)

    if verbose:
        print('Computed transform, saving to file.')
    ants.image_write(pet_atlas_xfm,out_image_path)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)

    return xfm_to_apply

def apply_xfm_ants(input_image_path: str,
                   ref_image_path: str,
                   out_image_path: str,
                   xfm_paths: list[str]):
    """
    Applies existing transforms in ANTs or ITK format to an input image, onto
    a reference image. This is useful for applying the same transform on
    different images to atlas space, for example.

    Args:
        input_image_path (str): Path to image on which transform is applied.
        ref_image_path (str): Path to image to which transform is applied.
        out_image_path (str): Path to which the transformed image is saved.
        xfm_paths (list[str]): List of transforms to apply to image. Must be in
            ANTs or ITK format, and can be affine matrix or warp coefficients.
    """
    pet_image_ants = ants.image_read(input_image_path)
    ref_image_ants = ants.image_read(ref_image_path)

    if pet_image_ants.dimension==4:
        dim = 3
    else:
        dim = 0

    xfm_image = ants.apply_transforms(fixed=ref_image_ants,
                                      moving=pet_image_ants,
                                      transformlist=xfm_paths,
                                      imagetype=dim-1)

    ants.image_write(xfm_image,out_image_path)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)


def apply_xfm_fsl(input_image_path: str,
                  ref_image_path: str,
                  out_image_path: str,
                  warp_path: str=None,
                  premat_path: str=None,
                  postmat_path: str=None,
                  **kwargs):
    """
    Applies existing transforms in FSL format to an input image, onto a
    reference image. This is useful for applying the same transform on
    different images to atlas space, for example.

    .. important::
        Requires installation of ``FSL``, and environment variables ``FSLDIR`` and
        ``FSLOUTPUTTYPE`` set appropriately in the shell.

    Args:
        input_image_path (str): Path to image on which transform is applied.
        ref_image_path (str): Path to image to which transform is applied.
        out_image_path (str): Path to which the transformed image is saved.
        warp_path (str): Path to FSL warp file.
        premat_path (str): Path to FSL ``premat`` matrix file.
        postmat_path (str): Path to FSL ``postmat`` matrix file.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`fsl.wrappers.applywarp`.
    """
    fsl.wrappers.applywarp(src=input_image_path,
                           ref=ref_image_path,
                           out=out_image_path,
                           warp=warp_path,
                           premat=premat_path,
                           postmat=postmat_path,
                           **kwargs)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)