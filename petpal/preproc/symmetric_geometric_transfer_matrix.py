import numpy as np
from typing import Tuple, Union
from scipy.ndimage import gaussian_filter
import nibabel as nib


def sgtm(pet_nifti: nib.Nifti1Image,
         roi_nifti: nib.Nifti1Image,
         fwhm: Union[float, Tuple[float, float, float]],
         zeroth_roi: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply Symmetric Geometric Transfer Matrix (SGTM) method for Partial Volume Correction (PVC) to PET images based
    on ROI labels.

    This method involves using a matrix-based approach to adjust the PET signal intensities for the effects of
    partial volume averaging.

    Args:
        pet_nifti (nib.Nifti1Image): The 3D PET image Nifti1 object.
        roi_nifti (nib.Nifti1Image): The 3D ROI image, Nifti1 object, should have the same dimensions as `pet_nifti`.
        fwhm (Union[float, Tuple[float, float, float]]): Full width at half maximum of the Gaussian blurring kernel for each dimension.
        zeroth_roi (bool): If False, ignores the zero label in calculations, often used to exclude background or non-ROI regions.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - np.ndarray: Array of unique ROI labels.
            - np.ndarray: Corrected PET values after applying PVC.
            - float: Condition number of the omega matrix, indicating the numerical stability of the inversion.

    Raises:
        AssertionError: If `pet_nifti` and `roi_nifti` do not have the same dimensions.

    Examples:
        .. code-block:: python

            pet_nifti = nib.load('path_to_pet_image.nii')
            roi_nifti = nib.load('path_to_roi_image.nii')
            fwhm = (8.0, 8.0, 8.0)  # or fwhm = 8.0
            labels, corrected_values, cond_number = sgtm(pet_nifti, roi_nifti, fwhm)

    Notes:
        The SGTM method uses the matrix :math:`\Omega` (omega), defined as:

        .. math::
        
            \Omega = V^T V

        where :math:`V` is the matrix obtained by applying Gaussian filtering to each ROI, converting each ROI into a
        vector. The element :math:`\Omega_{ij}` of the matrix :math:`\Omega` is the dot product of vectors
        corresponding to the i-th and j-th ROIs, representing the spatial overlap between these ROIs after blurring.

        The vector :math:`t` is calculated as:

        .. math::
        
            t = V^T p

        where :math:`p` is the vectorized PET image. The corrected values, :math:`t_{corrected}`, are then obtained
        by solving the linear system:

        .. math::
        
            \Omega t_{corrected} = t

        This provides the estimated activity concentrations corrected for partial volume effects in each ROI.
    """
    pet_3d = pet_nifti.get_fdata()
    roi_3d = roi_nifti.get_fdata()
    assert pet_3d.shape == roi_3d.shape, "PET and ROI images must be the same dimensions"

    resolution = pet_nifti.header.get_zooms()[:3]
    if isinstance(fwhm, float):
        sigma = [(fwhm / 2.355) / res for res in resolution]
    else:
        sigma = [(fwhm_i / 2.355) / res_i for fwhm_i, res_i in zip(fwhm, resolution)]

    unique_labels = np.unique(roi_3d)
    if not zeroth_roi:
        unique_labels = unique_labels[unique_labels != 0]

    flattened_size = pet_3d.size
    voxel_by_roi_matrix = np.zeros((flattened_size, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        masked_roi = (roi_3d == label).astype(float)
        blurred_roi = gaussian_filter(masked_roi, sigma=sigma)
        voxel_by_roi_matrix[:, i] = blurred_roi.ravel()

    omega = voxel_by_roi_matrix.T @ voxel_by_roi_matrix

    t_vector = voxel_by_roi_matrix.T @ pet_3d.ravel()
    t_corrected = np.linalg.solve(omega, t_vector)
    condition_number = np.linalg.cond(omega)

    return unique_labels, t_corrected, condition_number
