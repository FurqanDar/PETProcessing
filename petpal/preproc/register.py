"""
Provides tools to register PET images to anatomical or atlas space. Wrapper for
ANTs and FSL registration software.
"""
import os.path
from typing import Union
import logging
import numpy as np
import fsl.wrappers
import ants
import nibabel
from nibabel.processing import resample_from_to
from ..utils import image_io
from .motion_corr import determine_motion_target

logger = logging.getLogger(__name__)

def register_pet_to_anat(input_image_path: str,
                         reference_image_path: str,
                         motion_target_option: Union[str, tuple],
                         out_image_path: str,
                         verbose: bool = False,
                         type_of_transform: str = 'DenseRigid',
                         half_life: float = None,
                         **kwargs):
    """
    Computes and runs rigid registration of 4D PET image series to 3D anatomical image, typically a T1 MRI.

    Runs rigid registration module from Advanced Normalisation Tools (ANTs) with  default
    inputs. Will upsample PET image to the resolution of anatomical imaging.

    Args:
        input_image_path (str): Path to a .nii or .nii.gz file containing a 4D
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
        half_life (float): Half-life of PET tracer in seconds.
        kwargs (keyword arguments): Additional arguments passed to :py:func:`ants.registration`.
    """
    if verbose:
        logger.setLevel(logging.INFO)

    motion_target = determine_motion_target(motion_target_option=motion_target_option,
                                            input_image_4d_path=input_image_path,
                                            half_life=half_life)
    motion_target_image = ants.image_read(motion_target)
    mri_image = ants.image_read(reference_image_path)
    pet_image_ants = ants.image_read(input_image_path)

    transforms = ants.registration(moving=motion_target_image,
                                   fixed=mri_image,
                                   type_of_transform=type_of_transform,
                                   write_composite_transform=True,
                                   **kwargs)

    if pet_image_ants.dimension == 4:
        dim = 3
    else:
        dim = 0

    registered_image = ants.apply_transforms(moving=pet_image_ants,
                                             fixed=mri_image,
                                             transformlist=transforms['fwdtransforms'],
                                             interpolator='linear',
                                             imagetype=dim)
    logger.info(f'Registration applied to {input_image_path}')

    ants.image_write(registered_image, out_image_path)
    logger.info(f'Transformed image saved to {out_image_path}')

    image_io.safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)

def register_anat_to_pet(input_image_path: str,
                         reference_image_path: str,
                         motion_target_option: Union[str, tuple],
                         out_image_path: str,
                         verbose: bool = False,
                         type_of_transform: str = 'DenseRigid',
                         half_life: float = None,
                         input_seg_path: str = None,
                         out_seg_path: str = None,
                         **kwargs):
    """
    Computes and runs rigid registration of 3D anatomical image (typically a T1 MRI) to 3D Space of 4D PET image series.

    Runs rigid registration module from Advanced Normalisation Tools (ANTs) with default inputs. Will downsample
    anatomical image to the resolution of anatomical imaging. Allows for simultaneous registration of a segmentation
    image if it is in the same space as the anatomical image.

    Args:
        input_image_path (str): Path to a .nii or .nii.gz file containing a 3D
            anatomical image to be registered to PET space.
        reference_image_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to which anatomical image is registered.
        motion_target_option (str | tuple): Target image for computing
            transformation. See :meth:`determine_motion_target`.
        type_of_transform (str): Type of transform to perform on the PET image, must be one of antspy's
            transformation types, i.e. 'DenseRigid' or 'Translation'. Any transformation type that uses
            >6 degrees of freedom is not recommended, use with caution. See :py:func:`ants.registration`.
        out_image_path (str): Path to a .nii or .nii.gz file to which the registered PET series
            is written.
        verbose (bool): Set to ``True`` to output processing information.
        half_life (float): Half-life of PET tracer in seconds.
        input_seg_path (str): Path to a .nii or .nii.gz file containing a 3D segmentation image to be
            registered as well.
        out_seg_path (str): Path to a .nii or .nii.gz file to which the registered segmentation is written.
        kwargs (keyword arguments): Additional arguments passed to :py:func:`ants.registration`.
    """
    if verbose:
        logger.setLevel(logging.INFO)

    if bool(out_seg_path) != bool(input_seg_path): # logical XOR; either both or neither paths should be passed
        raise ValueError("If in_seg_path is specified, out_seg_path must also be specified.")

    motion_target = determine_motion_target(motion_target_option=motion_target_option,
                                            input_image_4d_path=reference_image_path,
                                            half_life=half_life)
    motion_target_image = ants.image_read(motion_target)
    mri_image_ants = ants.image_read(input_image_path)

    transforms = ants.registration(moving=mri_image_ants,
                                   fixed=motion_target_image,
                                   type_of_transform=type_of_transform,
                                   write_composite_transform=True,
                                   **kwargs)

    transform = transforms['fwdtransforms']

    registered_image = ants.apply_transforms(moving=mri_image_ants,
                                             fixed=motion_target_image,
                                             transformlist=transform,
                                             interpolator='linear',
                                             imagetype=0)
    logger.info(f'Registration applied to {input_image_path}')

    ants.image_write(registered_image, out_image_path)
    logger.info(f'Transformed image saved to {out_image_path}')

    image_io.safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)

    if input_seg_path is not None:
        segmentation_image_ants = ants.image_read(input_seg_path)
        registered_segmentation = ants.apply_transforms(moving=segmentation_image_ants,
                                                        fixed=motion_target_image,
                                                        transformlist=transform,
                                                        interpolator='genericLabel',
                                                        imagetype=0)
        logger.info(f'Registration applied to {input_seg_path}')
        ants.image_write(registered_segmentation, out_seg_path)
        logger.info(f'Transformed segmentation saved to {out_seg_path}')

def warp_pet_atlas(input_image_path: str,
                   anat_image_path: str,
                   atlas_image_path: str,
                   out_image_path: str,
                   verbose: bool,
                   type_of_transform: str = 'SyN',
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
        verbose (bool): Set to ``True`` to output processing information.
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

    if pet_image_ants.dimension == 4:
        dim = 3
    else:
        dim = 0

    pet_atlas_xfm = ants.apply_transforms(fixed=atlas_image_ants,
                                          moving=pet_image_ants,
                                          transformlist=xfm_to_apply,
                                          verbose=True,
                                          imagetype=dim)

    if verbose:
        print('Computed transform, saving to file.')
    ants.image_write(pet_atlas_xfm, out_image_path)

    image_io.safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)

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

    if pet_image_ants.dimension == 4:
        dim = 3
    else:
        dim = 0

    xfm_image = ants.apply_transforms(fixed=ref_image_ants,
                                      moving=pet_image_ants,
                                      transformlist=xfm_paths,
                                      imagetype=dim - 1)

    ants.image_write(xfm_image, out_image_path)

    image_io.safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)


def apply_xfm_fsl(input_image_path: str,
                  ref_image_path: str,
                  out_image_path: str,
                  warp_path: str = None,
                  premat_path: str = '',
                  postmat_path: str = '',
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
    if premat_path == '' and postmat_path == '':
        fsl.wrappers.applywarp(src=input_image_path,
                               ref=ref_image_path,
                               out=out_image_path,
                               warp=warp_path,
                               **kwargs)
    elif premat_path == '' and postmat_path != '':
        fsl.wrappers.applywarp(src=input_image_path,
                               ref=ref_image_path,
                               out=out_image_path,
                               warp=warp_path,
                               postmat=postmat_path,
                               **kwargs)
    elif premat_path != '' and postmat_path == '':
        fsl.wrappers.applywarp(src=input_image_path,
                               ref=ref_image_path,
                               out=out_image_path,
                               warp=warp_path,
                               premat=premat_path,
                               **kwargs)
    else:
        fsl.wrappers.applywarp(src=input_image_path,
                               ref=ref_image_path,
                               out=out_image_path,
                               warp=warp_path,
                               premat=premat_path,
                               postmat=postmat_path,
                               **kwargs)

    image_io.safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)


def resample_nii_4dfp(input_image_path: str,
                      resampled_image_path: str,
                      mpr_image_path: str,
                      out_image_path: str):
    """
    Resample and rearrange a 3D PET image to mpr space. This function mimics the functionality
    of converting an image from Nifti to 4dfp and back into Nifti, without using 4dfp tools.

    The typical use case would be for multi-modal studies where getting PET data aligned using
    exactly the same transformation procedures as BOLD data is critical. If, say, transformations
    have been computed from MPRAGE to atlas space using 4dfp tools, but the MPRAGE has been
    modified by converting from Nifti to 4dfp and back into Nifti, then the same conversion must be
    applied to the PET data aligned to anatomical space. This function accurately replicates that
    conversion.

    Args:
        input_image_path (str): Path to PET image on which transform is applied.
        resampled_image_path (str): Path to image with sampling needed for output. Often `rawavg.mgz` in FreeSurfer directory.
        mpr_image_path (str): Path to mpr (MPRAGE) image the PET will be transformed to.
        out_image_path (str): Path to which the transformed image is saved.
    """
    input_image = nibabel.load(input_image_path)
    resampled_image = nibabel.load(resampled_image_path)
    mpr_image = nibabel.load(mpr_image_path)
    input_image_resampled = resample_from_to(
        from_img=input_image,
        to_vox_map=(resampled_image.shape[:3], resampled_image.affine)
    )
    image_resampled_array = input_image_resampled.get_fdata()
    resampled_swapped = np.swapaxes(np.swapaxes(image_resampled_array, 0, 2), 1, 2)
    resampled_swapped_flipped = np.flip(np.flip(np.flip(resampled_swapped, 1), 2), 0)
    input_on_mpr = nibabel.nifti1.Nifti1Image(
        dataobj=resampled_swapped_flipped,
        affine=mpr_image.affine,
        header=mpr_image.header
    )
    nibabel.save(input_on_mpr, out_image_path)
    image_io.safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)
