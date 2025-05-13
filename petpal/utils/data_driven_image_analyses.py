"""
Data driven image analyses, including PCA analysis and input function extraction.
"""
import ants
import numpy as np
from sklearn.decomposition import PCA

from ..preproc.regional_tac_extraction import extract_roi_voxel_tacs_from_image_using_mask
from .useful_functions import check_physical_space_for_ants_image_pair
from .scan_timing import ScanTimingInfo


def temporal_pca_analysis_of_image_over_mask(input_image: ants.core.ANTsImage,
                                             mask_image: ants.core.ANTsImage,
                                             num_components: int = 3,
                                             svd_solver: str = 'full',
                                             whiten: bool = True,
                                             **sklearn_pca_kwargs) -> tuple[PCA, np.ndarray]:
    """
    Perform temporal PCA (Principal Component Analysis) on a 4D PET image using a specified mask.

    This function analyzes the temporal dynamics of voxel time-activity curves (TACs) extracted
    from the region specified by the input mask. PCA reduces the dimensionality of the dataset,
    extracting a set of temporal features ranked by their contribution to variability across
    voxels within the specified region. This approach effectively identifies the most prominent
    temporal patterns in the PET image's dynamic data.

    .. important::
        Ensure that the input and mask images are in the same coordinate space
        (aligned and registered) to avoid errors or mismatched results.

    Args:
        input_image (ants.core.ANTsImage):
            The 4D PET image for temporal PCA analysis. The image must have 3 spatial dimensions
            and one temporal dimension.
        mask_image (ants.core.ANTsImage):
            A binary mask representing the region of interest (ROI) in the input image.
            Only voxels within this mask are included in the PCA computation. The mask and input
            image must have the same physical dimensions.
        num_components (int, optional):
            The number of principal components to compute. Defaults to 3.
        svd_solver (str, optional): The type of SVD solver to use for PCA. Defaults to `"full"`.
        whiten (bool, optional):
            If True, the output features are normalized to have unit variance. This is useful
            for further statistical analysis of the temporal components. Defaults to True.
        **sklearn_pca_kwargs:
            Additional keyword arguments to customize the PCA behavior using scikit-learn's
            `PCA` class (e.g., `random_state`, `tol`, etc.).

    Returns:
        tuple[PCA, np.ndarray]: Fitted PCA model object and voxel TACs projected onto the principal components.

    Example:

        .. code-block:: python

            import ants
            from petpal.utils.data_driven_image_analyses import temporal_pca_analysis_of_image_over_mask as tpca_func

            pca_obj, pca_projections = tpca_func(input_image=ants.image_read('/path/to/4D/PET.nii.gz'),
                                                 mask_image=ants.image_read('/path/to/aligned/mask.nii.gz'),
                                                 num_components=3)

    Raises:
        AssertionError: The input image is not 4D.
        AssertionError: The input image and mask image are not in the same physical space or dimensions.

    Notes:
        - Ensure that the input image and mask image are pre-registered and aligned, with identical
          spatial dimensions.
        - The PCA algorithm identifies patterns across the temporal dynamics while minimizing loss
          of information.
        - Whitening normalizes the data, which can be important if further statistical operations
          (e.g., clustering) are planned.

    See Also:
        - :func:`extract_roi_voxel_tacs_from_image_using_mask<petpal.preproc.regional_tac_extraction.extract_roi_voxel_tacs_from_image_using_mask>`: Extracts voxel-level TACs for regions of interest
          in the input image, used in TAC calculations here.
        - :class:`sklearn.decomposition.PCA`: Underlying PCA implementation used in this function.
    """
    assert len(input_image.shape) == 4, "Input image must be 4D."
    assert check_physical_space_for_ants_image_pair(input_image, mask_image), (
        "Images must have the same physical dimensions.")
    mask_voxels = extract_roi_voxel_tacs_from_image_using_mask(input_image=input_image,
                                                               mask_image=mask_image)
    pca_obj = PCA(n_components=num_components, svd_solver=svd_solver, whiten=whiten, **sklearn_pca_kwargs)
    pca_obj.fit(mask_voxels)
    return pca_obj, np.ascontiguousarray(pca_obj.fit_transform(mask_voxels), float)


def extract_temporal_pca_components_of_image_over_mask(input_image: ants.core.ANTsImage,
                                                       mask_image: ants.core.ANTsImage,
                                                       num_components: int = 3,
                                                       svd_solver: str = 'full',
                                                       whiten: bool = True,
                                                       **sklearn_pca_kwargs) -> np.ndarray:
    """
    Extract principal components from temporal PCA on a 4D PET image using a specified mask.

    This function performs temporal Principal Component Analysis (PCA) on a 4D input image,
    focusing on the region defined by the mask. It returns only the PCA components
    (i.e., the eigenvectors that describe temporal variation) from the fitted PCA model.
    These components might help you identify the primary kinetic components over the mask.

    .. important::
        Ensure that the input and mask images are in the same coordinate space
        (aligned and registered) to avoid errors or mismatched results.

    Args:
        input_image (ants.core.ANTsImage):
            The 4D PET image for temporal PCA analysis. The image should have 3 spatial
            dimensions and one temporal dimension.
        mask_image (ants.core.ANTsImage):
            A binary mask defining the region of interest (ROI) in the input image.
            Only voxels within this mask will be considered during PCA computation.
            The mask and the input image must share the same physical and spatial dimensions.
        num_components (int, optional):
            The number of principal components to compute. Defaults to 3.
        svd_solver (str, optional):
            The type of SVD solver to use for PCA. Defaults to `"full"`.
        whiten (bool, optional):
            If set to `True`, the resulting components are scaled to have unit variance.
            Defaults to `True`.
        **sklearn_pca_kwargs:
            Additional keyword arguments for customizing PCA behavior via scikit-learn's
            `PCA` class, such as `random_state`, `tol`, etc.

    Returns:
        np.ndarray:
            A 2D array of shape `(num_components, num_timepoints)` representing the PCA
            components of temporal features. Each row corresponds to a principal component
            in the temporal domain.

    Example:

        .. code-block:: python

            import ants
            from petpal.utils.data_driven_image_analyses import extract_temporal_pca_components_of_image_over_mask as ext_tpca_comps_func

            pca_comps = ext_tpca_comps_func(input_image=ants.image_read('/path/to/4D/PET.nii.gz'),
                                            mask_image=ants.image_read('/path/to/aligned/mask.nii.gz'),
                                            num_components=3)


    Raises:
        AssertionError:
            Raised if the input image is not 4D, or if the mask is not in the same
            physical space as the input image.
        ValueError:
            Raised if the number of components exceeds the valid limits given the input
            data size.

    Notes:
        - This function is a higher-level utility that focuses solely on extracting
          PCA components for temporal dynamics.

    See Also:
        - :func:`temporal_pca_analysis_of_image_over_mask`: Performs full temporal PCA
          and returns both the fitted PCA model and the projections of voxel time-activity
          curves.
        - :class:`sklearn.decomposition.PCA`: Core class for performing Principal
          Component Analysis in scikit-learn.

    """
    pca_obj, _ = temporal_pca_analysis_of_image_over_mask(input_image=input_image,
                                                          mask_image=mask_image,
                                                          num_components=num_components,
                                                          svd_solver=svd_solver,
                                                          whiten=whiten,
                                                          **sklearn_pca_kwargs)
    return pca_obj.components_


def extract_temporal_pca_projection_of_image_over_mask(input_image: ants.core.ANTsImage,
                                                       mask_image: ants.core.ANTsImage,
                                                       num_components: int = 3,
                                                       svd_solver: str = 'full',
                                                       whiten: bool = True,
                                                       **sklearn_pca_kwargs) -> np.ndarray:
    """
    Compute and extract the temporal PCA projections for a masked region in a 4D PET image.

    This function carries out temporal Principal Component Analysis (PCA) on a 4D input image,
    focusing on the region defined by the mask. It returns the projections (i.e., the representation
    of each voxel in the reduced-dimensional space of principal components).

    .. important::
        Ensure that the input and mask images are in the same coordinate space
        (aligned and registered) to avoid errors or mismatched results.

    Args:
        input_image (ants.core.ANTsImage):
            A 4D PET image for temporal PCA analysis. The image should have 3 spatial
            dimensions and one temporal dimension.
        mask_image (ants.core.ANTsImage):
            A binary mask specifying the region of interest (ROI) to include in the PCA analysis.
            The mask must have the same physical and spatial dimensions as the input image.
        num_components (int, optional):
            The number of principal components to compute. Determines the dimensionality of
            the reduced output space. Defaults to 3.
        svd_solver (str, optional):
            Determines the solver used for Singular Value Decomposition (SVD) during PCA.
            Defaults to `"full"`.
        whiten (bool, optional):
            If set to `True`, components are scaled to have unit variance. This is often
            useful when the output is used as input to further statistical or machine learning models.
            Defaults to `True`.
        **sklearn_pca_kwargs:
            Additional keyword arguments for scikit-learn's `PCA` class, such as specification
            of `random_state`, `tol`, etc.

    Returns:
        np.ndarray:
            A 2D array of shape `(num_voxels, num_components)` containing the temporal PCA
            projections for each voxel within the mask. Each row represents the reduced-dimensional
            representation of a voxel's TAC within the selected components.

    Example:

        .. code-block:: python

            import ants
            from petpal.utils.data_driven_image_analyses import extract_temporal_pca_projection_of_image_over_mask as ext_tpca_proj_func

            pca_proj = ext_tpca_proj_func(input_image=ants.image_read('/path/to/4D/PET.nii.gz'),
                                          mask_image=ants.image_read('/path/to/aligned/mask.nii.gz'),
                                          num_components=3)

    Raises:
        AssertionError:
            Raised if the `input_image` is not 4D, or if the physical dimensions of the
            `mask_image` and `input_image` are not identical.
        ValueError:
            Raised if the number of PCA components exceeds the number of voxels included
            in the mask.

    Notes:
        - The projections represent the linear combination of the temporal principal components
          for each voxel, condensed into a reduced space.

    See Also:
        - :func:`temporal_pca_analysis_of_image_over_mask`: Performs the underlying PCA
          calculation and provides both PCA projections and the PCA model.
        - :class:`sklearn.decomposition.PCA`: Core implementation of the PCA used in this function.
    """
    _, transformed_voxels = temporal_pca_analysis_of_image_over_mask(input_image=input_image,
                                                                     mask_image=mask_image,
                                                                     num_components=num_components,
                                                                     svd_solver=svd_solver,
                                                                     whiten=whiten,
                                                                     **sklearn_pca_kwargs)
    return transformed_voxels


def extract_temporal_pca_quantile_thresholded_tacs_of_image_using_mask(input_image: ants.core.ANTsImage,
                                                                       mask_image: ants.core.ANTsImage,
                                                                       num_components: int = 3,
                                                                       threshold_components: list[int] | None = None,
                                                                       quantiles: list[float] | None = None,
                                                                       direction: str = '>',
                                                                       **sklearn_pca_kwargs) -> np.ndarray:
    """
    Extract quantile-thresholded time-activity curve (TAC) values for temporal PCA components from a 4D PET image.

    This function computes time-activity curve (TAC) statistics for masked regions in a 4D PET image, based on
    Principal Component Analysis (PCA). It thresholds the PCA projections (for provided selected components) using
    specified quantiles, and calculates the mean and standard deviation of the TACs for voxels exceeding the
    thresholds. The output includes both the TAC mean and standard deviation across all components and quantile
    thresholds.

    .. important::
        Ensure that the input and mask images are in the same coordinate space
        (aligned and registered) to avoid errors or mismatched results.

    Args:
        input_image (ants.core.ANTsImage):
            A 4D PET image.
        mask_image (ants.core.ANTsImage):
            A binary mask specifying the region of interest (ROI) in the image. Only voxels within this mask
            are considered for the PCA analysis and TAC computation. This image is assumed to be aligned
            to the input image.
        num_components (int, optional):
            The number of PCA components to use for reducing the dataset's dimensionality. Defaults to 3.
        threshold_components (list[int] | None, optional):
            A list of component indices (from 0 to `num_components - 1`) to use for thresholding.
            Thresholding is applied to these PCA components using the given quantiles. If not set, defaults to `[0]`.
        quantiles (list[float], optional):
            An array of quantiles (values between 0 and 1) to compute thresholds for the PCA projections.
            Only voxels with PCA projection values above the corresponding quantile thresholds are included
            when calculating TAC statistics. Defaults to `[0.5, 0.75, 0.9, 0.975]`.
        direction (str, optional):
            An optional argument to switch from ``>`` to ``<``. Defaults to ``>``.
        **sklearn_pca_kwargs:
            Additional keyword arguments passed to customize scikit-learn's PCA class (e.g., `svd_solver`,
            `random_state`, etc.).

    Returns:
        np.ndarray:
            A 2-element array containing `out_vals` (*np.ndarray*),  a 3D array of shape
            `(num_components, num_quantiles, num_timepoints)`, containing the mean TAC values of
            voxels exceeding the quantile thresholds for each component, and `out_stds` (*np.ndarray*),
            a 3D array of the same shape as `out_vals`, containing the standard deviation of the TAC
            values under the same conditions.

    Example:

        .. code-block:: python

            import ants
            from petpal.utils.data_driven_image_analyses import extract_temporal_pca_quantile_thresholded_tacs_of_image_using_mask as ext_pca_q_func

            tac_means, tac_stds = ext_pca_q_func(input_image=ants.image_read('/path/to/4D/PET.nii.gz'),
                                                 mask_image=ants.image_read('/path/to/aligned/mask.nii.gz'),
                                                 num_components=3,
                                                 threshold_components=[0,1,2],
                                                 direction='>', # set to '<' if we want the opposite direction
                                                 quantiles=np.array([0.5, 0.75, 0.9, 0.975]))

    Raises:
        AssertionError: `threshold_components` contains invalid indices (e.g., negative or greater than `num_components - 1`).
        AssertionError: Quantiles are not within the [0, 1] range.

    Notes:
        - Thresholding PCA projections using quantiles isolates specific temporal dynamics for analysis,
          which can provide insight into dominant trends in the time-activity data.
        - Voxels not exceeding the quantile thresholds are excluded from the mean and standard deviation
          calculations.

    See Also:
        - :func:`extract_temporal_pca_projection_of_image_over_mask`: Computes PCA projections used as input
          for thresholding in this function.
        - :func:`extract_roi_voxel_tacs_from_image_using_mask<petpal.preproc.regional_tac_extraction.extract_roi_voxel_tacs_from_image_using_mask>`: Extracts voxel-level TACs for regions of interest
          in the input image, used in TAC calculations here.
        - :class:`sklearn.decomposition.PCA`: Core PCA implementation used in this analysis.
    """

    if threshold_components is None:
        threshold_components = [i for i in range(num_components)]

    if quantiles is None:
        quantiles = [0.5, 0.75, 0.9, 0.975]

    assert min(threshold_components) >= 0, "Threshold components must be integers >= 0."
    assert max(threshold_components) < num_components, "Threshold components must be < num_components."
    assert np.min(quantiles) >= 0, "Quantiles must be >= 0."
    assert np.max(quantiles) <= 1, "Quantiles must be <= 1."

    voxels_pca_projs = extract_temporal_pca_projection_of_image_over_mask(input_image=input_image,
                                                                          mask_image=mask_image,
                                                                          num_components=num_components,
                                                                          **sklearn_pca_kwargs)
    voxels_pca_projs = voxels_pca_projs[:, threshold_components]

    roi_voxels = extract_roi_voxel_tacs_from_image_using_mask(input_image=input_image, mask_image=mask_image)

    num_components = len(threshold_components)
    num_quantiles = len(quantiles)
    num_frames = roi_voxels.shape[1]

    tacs_mean = np.zeros((num_components, num_quantiles, num_frames), float)
    tacs_std = np.zeros_like(tacs_mean)


    for comp_id, comp_pc_vals in enumerate(voxels_pca_projs.T):
        thresh_vals_for_comp = np.quantile(comp_pc_vals, quantiles)
        for thresh_id, a_thresh in enumerate(thresh_vals_for_comp):
            valid_voxels_mask = comp_pc_vals < a_thresh if direction == '<' else comp_pc_vals > a_thresh
            tmp_tacs = roi_voxels[valid_voxels_mask, :]
            tacs_mean[comp_id, thresh_id, :] = np.nanmean(tmp_tacs, axis=0)
            tacs_std[comp_id, thresh_id, :] = np.nanstd(tmp_tacs, axis=0)

    return np.asarray([tacs_mean, tacs_std])





def generate_temporal_pca_quantile_threshold_tacs_of_image_over_mask(input_image_path: str,
                                                                     mask_image_path: str,
                                                                     output_arrays_path: str | None,
                                                                     num_components: int,
                                                                     threshold_components: list | None,
                                                                     quantiles: list | None,
                                                                     direction: str = '>',
                                                                     **sklearn_pca_kwargs
                                                                     ):
    """
    Extract and compute time-activity curves (TACs) for an image using components of PCA projections
    and quantile thresholds over a specified mask.

    .. important::
        Ensure that the input and mask images are in the same coordinate space
        (aligned and registered) to avoid errors or mismatched results.

    This function processes a 4D-PET image (`input_image`) with an accompanying mask,
    computes PCA projections on the image data within the mask, thresholds these projections 
    using specified quantiles, and extracts TAC statistics (mean and standard deviation) 
    for each component-quantile pair.

    The results can optionally be saved to a file, and the TAC means and standard deviations 
    are returned as a NumPy array. The file written to disk has a different format than the array output
    of this function. The written file will have the frame reference times as the first column. The rest of
    the columns will come in pairs of mean and stderr of the TACs, iterated over the quantiles and then
    the threshold components.

    Args:
        input_image_path (str): Path to the input image file (in NIfTI format or similar).
        mask_image_path (str): Path to the mask image file (same dimensions as `input_image`).
        output_arrays_path (str | None):
            Path to save the output TAC arrays. If `None`, the arrays are not saved.
        num_components (int): Number of PCA components to compute.
        threshold_components (list[int] | None):
            Indices of PCA components to apply thresholds on. Defaults to `[0, 1]` if `None`.
        quantiles (list[float] | None):
            Array of quantile thresholds for filtering the PCA projections.
            Defaults to `[0.5, 0.75, 0.9, 0.975]` if `None`.
        direction (str):
            Comparison direction for the thresholds. Defaults to '>' (projection values
            greater than the quantile threshold). Other options may include '<'.
        **sklearn_pca_kwargs:
            Additional keyword arguments to pass to the PCA implementation (from scikit-learn), if applicable.

    Returns:
        np.ndarray:
            A NumPy array containing the mean and standard deviation TACs for each component-quantile pair,
            with shape `(2, num_components, num_quantiles, num_frames)`.

    Example:

        .. code-block:: python

                from petpal.utils.data_driven_image_analyses import generate_temporal_pca_quantile_threshold_tacs_of_image_over_mask


                # Input paths
                input_image_path = "example_input_image.nii"
                mask_image_path = "example_mask_image.nii"
                output_arrays_path = "output_tacs.txt"

                # Parameters
                num_components = 3
                threshold_components = [0, 1]
                quantiles = [0.5, 0.75, 0.95]
                direction = '<'

                # Run function
                tpca_tacs_func = generate_temporal_pca_quantile_threshold_tacs_of_image_over_mask
                tac_means, tac_stds = tpca_tacs_func(input_image_path=input_image_path,
                                                     mask_image_path=mask_image_path,
                                                     output_arrays_path=output_arrays_path,
                                                     num_components=num_components,
                                                     threshold_components=threshold_components,
                                                     quantiles=quantiles,
                                                     direction=direction)
                # Output shape
                print(tac_means.shape)

    Raises:
        AssertionError: If the shapes of the computed TAC mean and standard deviation arrays do not match.

    See Also:
        * :class:`sklearn.decomposition.PCA`: Core PCA implementation used in this analysis.
        * :func:`extract_temporal_pca_quantile_thresholded_tacs_of_image_using_mask`: computes PCA projections
        * :func:`_generate_quantiled_multi_tacs_header`: formats the output array header
        * :func:`_gen_reshaped_quantiled_tacs`: reshapes the means and stds for writing to disk

    """
    if threshold_components is None:
        threshold_components = [0, 1]

    if quantiles is None:
        quantiles = [0.5, 0.75, 0.9, 0.975]

    input_image = ants.image_read(input_image_path)
    mask_image = ants.image_read(mask_image_path)

    tacs_ext_func = extract_temporal_pca_quantile_thresholded_tacs_of_image_using_mask
    tacs_mean, tacs_std = tacs_ext_func(input_image=input_image,
                                        mask_image=mask_image,
                                        num_components=num_components,
                                        threshold_components=threshold_components,
                                        quantiles=quantiles,
                                        direction=direction,
                                        **sklearn_pca_kwargs
                                        )

    if output_arrays_path is not None:
        image_timing_info = ScanTimingInfo.from_nifti(image_path=input_image_path)
        tac_ref_times = image_timing_info.center_in_mins
        out_array = _gen_reshaped_quantiled_tacs(times=tac_ref_times,
                                                 tacs_mean=tacs_mean,
                                                 tacs_std=tacs_std)
        out_header = _generate_quantiled_multi_tacs_header(threshold_components=threshold_components,
                                                           quantiles=quantiles,
                                                           direction=direction)
        np.savetxt(fname=output_arrays_path,
                   X=out_array.T,
                   fmt='%.6e',
                   delimiter='\t',
                   header=out_header,
                   comments='')

    return np.asarray([tacs_mean, tacs_std])


def _generate_quantiled_multi_tacs_header(threshold_components: list, quantiles: list, direction: str):
    """
    Generate the header for outputting multiple TACs generated by the quantile threshold function.

    First column is assumed to be time in minutes, the second column is the mean tac values
    of the first component and first quantile, the third column is the standard deviation,
    and the rest of pairs of columns go over all the quantiles and then components in order.

    Args:
        threshold_components (list[int]): Indices of PCA components for thresholding.
        quantiles (list[float]): List of quantile threshold values.
        direction (str): '<' or '>'.

    Returns:
        str: A single string representing the tab-delimited header.
    """
    header = ['time(mins)']

    for a_comp in threshold_components:
        for a_quan in quantiles:
            header.append(f'(c:{a_comp};q{direction}{a_quan:.3f})_mean')
            header.append(f'(c:{a_comp};q{direction}{a_quan:.3f})_std')

    return "\t".join(header)


def _gen_reshaped_quantiled_tacs(times: np.ndarray[float],
                                 tacs_mean: np.ndarray[float],
                                 tacs_std: np.ndarray[float]) -> np.ndarray:
    """
    Generate the aggregated multiple-tacs array for quantile threshold obtained tacs.

    First column is assumed to be time in minutes, the second column is the mean tac values
    of the first component and first quantile, the third column is the standard deviation,
    and the rest of pairs of columns go over all the quantiles and then components in order.

    Args:
        times (np.ndarray[float]): The frame times corresponding to the TACs.
        tacs_mean (np.ndarray[float]):
            The mean tac values of the TACs over the threshold components and quantiles.
        tacs_std (np.ndarray[float]):
            The stderr value of the TACs over the threshold components and quantiles.
    Returns:
        np.ndarray:
            Array containing the aggregated multiple-tacs array. The first column is the time,
            the rest of the column pairs correspond to the threshold component and the quantile.
    """
    assert tacs_mean.shape == tacs_std.shape, (f"The TACs mean and stderrs must have the same shape."
                                               f" mean:{tacs_mean.shape}, std:{tacs_std.shape}")
    num_components, num_quantiles, num_frames = tacs_mean.shape
    num_tacs = num_components * num_quantiles
    num_columns = 2 * num_tacs + 1
    out_tacs_array = np.zeros(shape=(num_columns, num_frames), dtype=float)

    out_tacs_array[0] = times
    out_tacs_array[1::2] = tacs_mean.reshape(num_tacs, num_frames)
    out_tacs_array[2::2] = tacs_std.reshape(num_tacs, num_frames)

    return out_tacs_array
