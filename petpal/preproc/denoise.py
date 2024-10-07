""" Provides functions for denoising PET images. """
import logging
import os
import math

import numpy as np
from nibabel.filebasedimages import FileBasedImage
from sklearn.cluster import k_means

from ..utils.image_io import ImageIO

logger = logging.getLogger(__name__)


def denoise_image(pet_image: np.ndarray,
                  t1_image: np.ndarray,
                  freesurfer_segmentation: np.ndarray) -> np.ndarray:
    """Use Hamed Yousefi's method to denoise a PET image."""
    pass


def apply_3_tier_k_means_clustering(data: np.ndarray,
                                    num_clusters: list[int],
                                    **kwargs) -> [np.ndarray, np.ndarray]:
    """Separate data into num_clusters clusters using Lloyd's algorithm implemented in sklearn."""

    # Verify format of inputs
    if len(num_clusters) != 3:
        raise IndexError(
            'num_clusters must be a list of length 3, where num_clusters[0] is the number of clusters at the top-level,'
            ' num_clusters[1] is the number of clusters to separate each of the top-level clusters into, and so on.')

    if data.ndim != 2:
        raise IndexError('data input MUST be a 2-D numpy array, where the first dimension corresponds to the samples, '
                         'and the second dimension corresponds to the features')

    # Dimensions will be (# of clusters, # of features)
    centroids = np.zeros(shape=(np.prod(num_clusters), data.shape[1]))
    _, cluster_ids, _ = k_means(X=data,
                                n_clusters=num_clusters[0],
                                **kwargs)

    cluster_ids_2 = np.zeros(shape=cluster_ids.shape)
    for cluster in range(num_clusters[0]):
        print(f'ClusterID: {cluster}')
        cluster_data = data[cluster_ids == cluster, :]
        print(f'{cluster_data}\n{cluster_data.shape}')
        _, cluster_ids_temp, _ = k_means(X=cluster_data,
                                         n_clusters=num_clusters[1],
                                         **kwargs)
        print(f'cluster_ids_temp\n{cluster_ids_temp}\n{cluster_ids_temp.shape}')
        cluster_ids_2[cluster_ids == cluster] = cluster_ids[cluster_ids == cluster] * num_clusters[1] + cluster_ids_temp

    cluster_ids_3 = np.zeros(shape=cluster_ids.shape)
    for cluster in range(num_clusters[0] * num_clusters[1]):
        print(f'ClusterID: {cluster}')
        cluster_data = data[cluster_ids_2 == cluster, :]
        centroids_temp, cluster_ids_temp, _ = k_means(X=cluster_data,
                                                      n_clusters=num_clusters[2],
                                                      **kwargs)
        cluster_ids_3[cluster_ids_2 == cluster] = cluster_ids_temp + num_clusters[2] * cluster
        print(f'Centroids for cluster {cluster}\n{centroids_temp}\n{centroids_temp.shape}')

    cluster_ids = cluster_ids + cluster_ids_3

    return cluster_ids


def rearrange_voxels_to_wheel_space():
    """Use voxelwise distances from cluster feature averages to arrange voxel indices onto 2D 'wheel' space."""
    pass


def apply_smoothing_in_sinogram_space():
    """Smooth clustered feature image on its sinogram space using radon transform and predefined smoothing kernel."""
    pass


def temporal_pca():
    """Run principal component analysis on spatially-flattened PET dataset and return PC1, 2, and 3 scores for each index."""
    pass


def _prepare_inputs(path_to_pet: str,
                    path_to_mri: str,
                    path_to_freesurfer_segmentation: str) -> [np.ndarray]:
    """Read images from files into ndarrays, and ensure all images have the same dimensions as PET."""

    logger.debug("Logger Attached")
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
        raise OSError(f'{len(images_failed_to_load)} images could not be loaded. See errors below.\n{print(errors)}')

    # Extract ndarrays from each image.
    pet_data = image_loader.extract_image_from_nii_as_numpy(images_loaded[0])
    mri_data = image_loader.extract_image_from_nii_as_numpy(images_loaded[1])
    segmentation_data = image_loader.extract_image_from_nii_as_numpy(images_loaded[2])
    pet_data_3d_shape = pet_data.shape[:-1]

    if mri_data.shape != pet_data_3d_shape or segmentation_data.shape != pet_data_3d_shape:
        raise Exception(f'MRI and/or Segmentation has different dimensions from 3D PET image:\n'
                        f'PET Frame Shape: {pet_data_3d_shape}\n'
                        f'Segmentation Shape: {segmentation_data.shape}\n'
                        f'MRI Shape: {mri_data.shape}.\n'
                        f'Ensure that all non-PET data is registered to PET space')

    return [pet_data, mri_data, segmentation_data]
