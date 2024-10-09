""" Provides functions for denoising PET images. """
import logging
import os
import math

import numpy as np
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from scipy.ndimage import convolve, binary_fill_holes

from ..utils.image_io import ImageIO
from ..preproc.image_operations_4d import threshold_binary

logger = logging.getLogger(__name__)


class Denoiser:
    """Wrapper class for handling inputs, outputs, and logging for denoising."""

    def __init__(self,
                 path_to_pet: str,
                 path_to_mri: str,
                 path_to_segmentation: str):
        try:
            self.pet_data, self.mri_data, self.segmentation_data = self._prepare_inputs(path_to_pet=path_to_pet,
                                                                                        path_to_mri=path_to_mri,
                                                                                        path_to_freesurfer_segmentation=path_to_segmentation)
        except OSError as e:
            raise OSError(e)
        except Exception as e:
            raise Exception(e)

    def run(self):
        """Denoise Image"""
        pass

    @staticmethod
    def _prepare_inputs(path_to_pet: str,
                        path_to_mri: str,
                        path_to_freesurfer_segmentation: str) -> (np.ndarray, np.ndarray, np.ndarray):
        """Read images from files into ndarrays, and ensure all images have the same dimensions as PET."""

        images_loaded = []
        images_failed_to_load = []
        errors = []
        image_loader = ImageIO()

        # Verify that all files can be loaded and saved as ndarrays.
        for path in [path_to_pet, path_to_mri, path_to_freesurfer_segmentation]:
            try:
                images_loaded.append(image_loader.load_nii(path))
            except (FileNotFoundError, OSError) as e:
                images_failed_to_load.append(path)
                errors.append(e)

        # Log errors if any images couldn't be loaded
        if len(images_failed_to_load) > 0:
            raise OSError(
                f'{len(images_failed_to_load)} images could not be loaded. See errors below.\n{print(errors)}')

        # Extract ndarrays from each image.
        pet_data = image_loader.extract_image_from_nii_as_numpy(images_loaded[0])
        mri_data = image_loader.extract_image_from_nii_as_numpy(images_loaded[1])
        segmentation_data = image_loader.extract_image_from_nii_as_numpy(images_loaded[2])
        pet_data_3d_shape = pet_data.shape[:-1]

        if pet_data.ndim != 4:
            raise Exception(
                f'PET data has {pet_data.ndim} dimensions, but 4 is expected. Ensure that you are loading a '
                f'4DPET dataset, not a single frame')

        if mri_data.shape != pet_data_3d_shape or segmentation_data.shape != pet_data_3d_shape:
            raise Exception(f'MRI and/or Segmentation has different dimensions from 3D PET image:\n'
                            f'PET Frame Shape: {pet_data_3d_shape}\n'
                            f'Segmentation Shape: {segmentation_data.shape}\n'
                            f'MRI Shape: {mri_data.shape}.\n'
                            f'Ensure that all non-PET data is registered to PET space')

        return pet_data, mri_data, segmentation_data


def apply_3_tier_k_means_clustering(flattened_feature_data: np.ndarray,
                                    num_clusters: list[int],
                                    **kwargs) -> (np.ndarray, np.ndarray):
    """Separate data into num_clusters clusters using Lloyd's algorithm implemented in sklearn.

    This function performs k-means clustering "recursively" on feature data from a (PET) image. The input data should be 2D,
    where one dimension corresponds to all the voxels in a single 3D PET Frame, and the other dimension corresponds to
    the feature values for those voxels. Example features include Temporal PCA components from PET, T1 or T2 MRI
    intensity, freesurfer segmentation. Note that MRI and segmentation data must be registered to native PET space.
    This input data is clustered with k-means 3 successive times, with each cluster from the first tier being passed
    into the second tier, and so on for the third tier. The final number of clusters is considered to be the product of
    num_cluster's elements, since the final tier's cluster outputs are returned.

    Args:
        flattened_feature_data (np.ndarray): Feature data from PET (and other sources) to cluster. Must be 2D, where one
            dimension corresponds to all the voxels in a single 3D PET Frame, and the other dimension corresponds to the
            feature values for those voxels.
        num_clusters (list[int]): Number of clusters to use in each tier of k_means clustering. num_clusters must have
            a length of 3, where the value at the first index is the number of clusters in the first tier, and so on.
        **kwargs: Additional keyword arguments passed to the `sklearn.cluster.k_means` method.

    Returns:
        Tuple[np.ndarray, np.ndarray]: First array contains the feature centroids for each cluster, and the second
        contains the cluster labels for each "voxel" in the input data. See sklearn.cluster.k_means documentation for
        more details.

    """

    # Verify format of inputs
    if len(num_clusters) != 3:
        raise IndexError(
            'num_clusters must be a list of length 3, where num_clusters[0] is the number of clusters at the top-level,'
            ' num_clusters[1] is the number of clusters to separate each of the top-level clusters into, and so on.')

    if flattened_feature_data.ndim != 2:
        raise IndexError(
            'flattened_feature_data input MUST be a 2-D numpy array, where the first dimension corresponds to the samples, '
            'and the second dimension corresponds to the features')

    # Dimensions will be (# of clusters, # of features)
    centroids = np.zeros(shape=(np.prod(num_clusters), flattened_feature_data.shape[1]))
    _, cluster_ids, _ = k_means(X=flattened_feature_data,
                                n_clusters=num_clusters[0],
                                **kwargs)

    cluster_ids_2 = np.zeros(shape=cluster_ids.shape)
    for cluster in range(num_clusters[0]):
        logger.debug(f'Top-Level Cluster ID: {cluster}')
        cluster_data = flattened_feature_data[cluster_ids == cluster, :]
        logger.debug(f'{cluster_data}\n{cluster_data.shape}')
        _, cluster_ids_temp, _ = k_means(X=cluster_data,
                                         n_clusters=num_clusters[1],
                                         **kwargs)
        logger.debug(f'cluster_ids_temp\n{cluster_ids_temp}\n{cluster_ids_temp.shape}')
        cluster_ids_2[cluster_ids == cluster] = cluster_ids[cluster_ids == cluster] * num_clusters[1] + cluster_ids_temp

    cluster_ids_3 = np.zeros(shape=cluster_ids.shape)
    for cluster in range(num_clusters[0] * num_clusters[1]):
        logger.debug(f'Mid-Level Cluster ID: {cluster}')
        cluster_data = flattened_feature_data[cluster_ids_2 == cluster, :]
        centroids_temp, cluster_ids_temp, _ = k_means(X=cluster_data,
                                                      n_clusters=num_clusters[2],
                                                      **kwargs)
        cluster_ids_3[cluster_ids_2 == cluster] = cluster_ids_temp + num_clusters[2] * cluster
        logger.debug(f'Centroids for cluster {cluster}\n{centroids_temp}\n{centroids_temp.shape}')
        for sub_cluster in range(num_clusters[2]):
            centroids[cluster * num_clusters[2] + sub_cluster, :] = centroids_temp[sub_cluster]

    cluster_ids = cluster_ids + cluster_ids_3

    return centroids, cluster_ids


def head_mask(pet_data: np.ndarray,
              thresh: float = 500.0) -> np.ndarray:
    """Function to extract 3D head mask PET data using basic morphological methods"""

    mean_slice = np.mean(pet_data, axis=3)
    thresholded_data = threshold_binary(input_image_numpy=mean_slice, lower_bound=thresh)
    kernel = np.ones(shape=(3, 3, 3))
    neighbor_count = convolve(thresholded_data, kernel, mode='constant')
    thresholded_data[neighbor_count < 14] = 0
    mask_image = binary_fill_holes(thresholded_data)

    return mask_image


def flatten_pet_spatially(pet_data: np.ndarray) -> np.ndarray:
    """Flatten spatial dimensions (using C index order) of 4D PET and return 2D version"""

    num_voxels = np.prod(pet_data.shape[:-1])
    flattened_pet_data = pet_data.reshape(num_voxels, -1)

    return flattened_pet_data


def temporal_pca(flattened_pet_data: np.ndarray,
                 flattened_head_mask: np.ndarray) -> np.ndarray:
    """Run principal component analysis on spatially-flattened PET and return PC1, 2, 3 scores per index"""
    # head_data = flattened_pet_data[flattened_head_mask, :]
    # pca = PCA(n_components=2).fit(X=head_data)

    pass


def add_nonbrain_features_to_segmentation(segmentation_data: np.ndarray,
                                          spatially_flattened_pet_data: np.ndarray,
                                          head_mask_data: np.ndarray) -> np.ndarray:
    """Cluster non-brain and add labels to existing segmentation"""
    pass


def denoise_image_data(pet_data: np.ndarray,
                       t1_mri_data: np.ndarray,
                       segmentation_data: np.ndarray) -> np.ndarray:
    """Use Hamed Yousefi's method to denoise a PET image."""
    pass


def rearrange_voxels_to_wheel_space():
    """Use voxelwise distances from cluster feature averages to arrange voxel indices onto 2D 'wheel' space."""
    pass


def apply_smoothing_in_sinogram_space():
    """Smooth clustered feature image on its sinogram space using radon transform and predefined smoothing kernel."""
    pass
