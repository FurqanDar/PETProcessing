""" Provides functions for denoising PET images. """
import logging
import math

import numpy as np
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from scipy.ndimage import convolve, binary_fill_holes
from scipy.stats import zscore


from ..utils.useful_functions import weighted_series_sum
from ..utils.image_io import ImageIO
from ..preproc.image_operations_4d import binarize_image_with_threshold

logger = logging.getLogger(__name__)


class Denoiser:
    """Wrapper class for handling inputs, outputs, and logging for denoising."""

    pet_data = None
    mri_data = None
    segmentation_data = None
    updated_segmentation_data = None
    head_mask = None
    non_brain_mask = None


    def __init__(self,
                 path_to_pet: str,
                 path_to_mri: str,
                 path_to_segmentation: str,
                 verbosity: int = 0):

        if verbosity in [-2, -1, 0, 1, 2]:
            logger.setLevel(level=30-(10*verbosity))
        else:
            raise ValueError("Verbosity argument must be an int from -2 to 2. The default (0) corresponds to the "
                             "default logging level (warning). A higher value increases the verbosity and a lower "
                             f"value decreases it. Verbosity given was {verbosity}. See python's logging documentation "
                             "for more information.")

        try:
            self.pet_data, self.mri_data, self.segmentation_data = self._prepare_inputs(path_to_pet=path_to_pet,
                                                                                        path_to_mri=path_to_mri,
                                                                                        path_to_freesurfer_segmentation=path_to_segmentation)
        except OSError as e:
            raise OSError(e)
        except Exception as e:
            raise Exception(e)

    def __call__(self, *args, **kwargs):
        """Denoise Image"""
        pass

    def run_single_iteration(self):
        """"""
        self.head_mask = generate_head_mask(self.pet_data)
        flattened_pet_data = flatten_pet_spatially(self.pet_data)
        self.add_nonbrain_features_to_segmentation()

    def run(self):
        """"""

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


    def apply_3_tier_k_means_clustering(self,
                                        flattened_feature_data: np.ndarray,
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


    def extract_distances_to_cluster_centroids(self,
                                               cluster_data: np.ndarray,
                                               all_cluster_centroids: np.ndarray) -> np.ndarray:
        """Calculate distances from centroids in feature space for each voxel assigned to a cluster.

        Args:
            cluster_data (np.ndarray): 2D array of size (number of voxels in cluster, number of features).
            all_cluster_centroids (np.ndarray): 2D array of size (number of total clusters, number of features). Each
                cluster's feature centroids (mean scores) are stored.

        Returns:
            np.ndarray: 2D array of size
                (number of voxels in cluster, number of total clusters). For each voxel in the cluster, contains the
                SSD (sum of squared differences) from the feature centroids of all clusters.
        """

        calculate_ssd = lambda features: np.sum((all_cluster_centroids - features.T) ** 2, axis=1)
        cluster_feature_distances = np.apply_along_axis(calculate_ssd, axis=1, arr=cluster_data)
        return cluster_feature_distances


    def extract_distances_in_ring_space(self,
                                        cluster_locations: np.ndarray,
                                        ring_space_shape: (int, int)) -> np.ndarray:
        """Calculate distances from every cluster's assigned location (not centroid) for each pixel in the ring space"""


    def add_nonbrain_features_to_segmentation(self):
        """Cluster non-brain and add labels to existing segmentation"""
        self._generate_non_brain_mask()
        segmentation_data = self.segmentation_data
        non_brain_features = self._extract_non_brain_features()
        _, cluster_ids, _ = k_means(X=non_brain_features,
                n_clusters=5)

        start_label = np.max(segmentation_data) + 1

        flat_segmentation_data = segmentation_data.flatten()
        flat_non_brain_mask = self.non_brain_mask.flatten()
        flat_segmentation_data[flat_non_brain_mask] = start_label + cluster_ids

        segmentation_data_with_non_brain = flat_segmentation_data.reshape(segmentation_data.shape)

        self.updated_segmentation_data = segmentation_data_with_non_brain


    def _extract_non_brain_features(self) -> np.ndarray:
        """

        Returns:

        """
        non_brain_mask_data = self.non_brain_mask
        spatially_flat_non_brain_mask = non_brain_mask_data.flatten()
        flat_mri_data = self.mri_data.flatten()
        spatially_flat_pet = flatten_pet_spatially(self.pet_data)

        logger.debug(f'Original: \nmri_data shape {self.mri_data.shape}\nnon_brain_mask shape {self.non_brain_mask.shape}'
              f'\npet_data shape {self.pet_data.shape}\n')

        logger.debug(f'Flat: \nmri_data shape | type {flat_mri_data.shape} | {flat_mri_data.dtype}'
              f'\nnon_brain_mask shape | type {spatially_flat_non_brain_mask.shape} | {spatially_flat_non_brain_mask.dtype}'
              f'\npet_data shape | type {spatially_flat_pet.shape} | {spatially_flat_pet.dtype}\n')

        non_brain_pet_data = spatially_flat_pet[spatially_flat_non_brain_mask, :]

        logger.debug(f'PCA input shape: {non_brain_pet_data.shape}\n')

        pca_data = PCA(n_components=4).fit_transform(X=non_brain_pet_data)

        logger.debug(f'Flat MRI Data for non-brain region shape {flat_mri_data[spatially_flat_non_brain_mask].shape}\n')
        logger.debug(f'Flat MRI Data for non-brain region {flat_mri_data[spatially_flat_non_brain_mask]}\n')
        logger.debug(f'PCA results shape: {pca_data.shape}\n')

        mri_plus_pca_data = np.zeros(shape=(pca_data.shape[0], pca_data.shape[1]+1))
        mri_plus_pca_data[:, :-1] = pca_data
        mri_plus_pca_data[:, -1] = flat_mri_data[spatially_flat_non_brain_mask]
        mri_plus_pca_data = zscore(mri_plus_pca_data, axis=0) # TODO: Verify that this is the right axis with data

        logger.debug(f'Non-brain features: \n{mri_plus_pca_data}\n')
        return mri_plus_pca_data

    def _generate_non_brain_mask(self):
        """

        Returns:

        """
        segmentation_data = self.segmentation_data
        head_mask_data = self.head_mask
        brain_mask_data = np.where(segmentation_data > 0, 1, 0)
        non_brain_mask_data = head_mask_data - brain_mask_data

        self.non_brain_mask = non_brain_mask_data.astype(bool)

    def map_to_ring_space(self):
        """Use voxelwise distances from cluster feature centroids to arrange voxels onto 2D 'ring map'."""
        pass

    @staticmethod
    def _generate_empty_ring_space(num_voxels_in_cluster: int) -> np.ndarray:
        """Use the number of voxels in a cluster to create an empty 'ring space' that can contain the cluster data."""
        ring_space_dimensions = (math.floor(math.sqrt(2)*math.sqrt(num_voxels_in_cluster+1)) + 4
                                - math.floor(math.sqrt(num_voxels_in_cluster+1)) % 4)

        return np.zeros(shape=(ring_space_dimensions, ring_space_dimensions))


    def apply_smoothing_in_sinogram_space(self,
                                          image_data: np.ndarray,
                                          kernel: np.ndarray,
                                          **kwargs) -> np.ndarray:
        """Transform image to sinogram space, apply smoothing, and transform back to original domain"""
        pass

    def weighted_sum_smoothed_image_iterations(self):
        """Weight smoothed images (one from each iteration) by cluster 'belongingness' with respect to MRI. """
        pass


def flatten_pet_spatially(pet_data: np.ndarray) -> np.ndarray:
    """Flatten spatial dimensions (using C index order) of 4D PET and return 2D array (numVoxels x numFrames)"""

    num_voxels = np.prod(pet_data.shape[:-1])
    flattened_pet_data = pet_data.reshape(num_voxels, -1)

    return flattened_pet_data


def generate_head_mask(pet_data: np.ndarray,
                       thresh: float = 500.0) -> np.ndarray:
    """Function to extract 3D head mask PET data using basic morphological methods"""

    mean_slice = np.mean(pet_data, axis=3) # TODO: Use weighted series sum instead; more reliable
    thresholded_data = binarize_image_with_threshold(input_image_numpy=mean_slice, lower_bound=thresh)
    kernel = np.ones(shape=(3, 3, 3))
    neighbor_count = convolve(thresholded_data, kernel, mode='constant')
    thresholded_data[neighbor_count < 14] = 0
    mask_image = binary_fill_holes(thresholded_data)

    return mask_image