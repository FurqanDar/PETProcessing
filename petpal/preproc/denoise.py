""" Provides Denoiser Class to run cluster-based denoising on PET images.

TODO: Credit Hamed Yousefi and his publication formally once it's published.

"""

# Import Python Standard Libraries
import logging
import math
import os.path
import time
from typing import Union

# Import other libraries
import numpy as np
from docutils.nodes import image
from numba import njit
from skimage.transform import radon, iradon
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from scipy.ndimage import convolve, binary_fill_holes, binary_closing
from scipy.stats import zscore, norm
import nibabel as nib

# Import from petpal
from ..utils.image_io import ImageIO
from ..preproc.image_operations_4d import binarize_image_with_threshold
from ..preproc.image_operations_4d import SimpleAutoImageCropper

# Initialize logger
logger = logging.getLogger(__name__)

class Denoiser:
    """Wrapper class for handling inputs, outputs, and logging for denoising, as well as the main pipeline functions"""

    # Class attributes; The fewer the better with respect to memory.
    pet_image = None
    mri_image = None
    segmentation_image = None
    head_mask_data = None
    head_mask_lims = None
    updated_segmentation_data = None
    non_brain_mask_data = None

    def __init__(self,
                 path_to_pet: str,
                 path_to_mri: str,
                 path_to_segmentation: str,
                 path_to_wss: str,
                 verbosity: int = 0):

        if verbosity in [-2, -1, 0, 1, 2]:
            log_level = 30 - (10 * verbosity)
            logger.setLevel(level=log_level)
            file_handler = logging.FileHandler('/export/scratch1/oestreichk/denoise.log',
                                               mode='w')
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        else:
            raise ValueError("Verbosity argument must be an int from -2 to 2. The default (0) corresponds to the "
                             "default logging level (warning). A higher value increases the verbosity and a lower "
                             f"value decreases it. Verbosity given was {verbosity}. See python's logging documentation "
                             "for more information.")

        try:
            (self.pet_image,
             self.mri_image,
             self.segmentation_image,
             self.head_mask_data,
             self.head_mask_lims) = self._prepare_inputs(path_to_pet=path_to_pet,
                                                         path_to_mri=path_to_mri,
                                                         path_to_freesurfer_segmentation=path_to_segmentation,
                                                         path_to_wss=path_to_wss)
        except OSError as e:
            raise e
        except Exception as e:
            raise e



    # Should run the entire process; Probably just call run()
    def __call__(self):
        """Denoise Image"""
        pass

    # "Pipeline" Functions: Functions that string a number of other functions.
    def run_single_iteration(self,
                             num_clusters: list[int]):
        """Generate a denoised image using one iteration of the method, to be weighted with others downstream."""

        # TODO: Move these somewhere else (i.e. run()) so they're only called once.

        self.head_mask = generate_head_mask(self.pet_image)
        flattened_head_mask = self.head_mask.flatten()
        flattened_pet_data = flatten_pet_spatially(self.pet_image)
        self.non_brain_mask = self._generate_non_brain_mask()
        self.updated_segmentation_data = self._add_nonbrain_features_to_segmentation(non_brain_mask=self.non_brain_mask)
        flattened_head_pet_data = flattened_pet_data[flattened_head_mask, :]
        flattened_mri_data = self.mri_image.flatten()
        flattened_segmentation_data = self.updated_segmentation_data.flatten()

        feature_data = np.zeros(shape=(flattened_head_pet_data.shape[0], 6))
        feature_data[:, :-2] = self._temporal_pca(spatially_flattened_pet_data=flattened_head_pet_data,
                                                  num_components=4)
        feature_data[:, -2] = flattened_mri_data[flattened_head_mask]
        feature_data[:, -1] = flattened_segmentation_data[flattened_head_mask]

        feature_data = zscore(feature_data, axis=0)

        # TODO: Probably ought to set object attribute values only in these run*() methods, rather than in other methods

        centroids, cluster_ids = self.apply_3_tier_k_means_clustering(flattened_feature_data=feature_data,
                                                                      num_clusters=num_clusters)

        denoised_flattened_head_data = np.zeros(shape=flattened_head_pet_data.shape[0])
        smoothing_kernel = self._generate_2d_gaussian_filter()

        final_num_clusters = np.prod(num_clusters).astype(int)
        for cluster in range(final_num_clusters):
            start = time.time()
            logger.debug(f'Cluster {cluster}\n-------------------------------------------------------\n\n\n')
            cluster_data = feature_data[cluster_ids == cluster]
            centroids_temp = np.roll(centroids, shift=-cluster, axis=0)
            feature_distances = self._extract_distances_to_cluster_centroids(cluster_data=cluster_data,
                                                                             all_cluster_centroids=centroids_temp)
            num_voxels_in_cluster = len(cluster_ids[cluster_ids == cluster])
            cluster_voxel_indices = np.argwhere(cluster_ids == cluster).T[0]
            ring_space_side_length = self._calculate_ring_space_dimension(num_voxels_in_cluster=num_voxels_in_cluster)
            logger.debug(f'Ring Space Side Length: {ring_space_side_length}')
            cluster_locations = self._define_cluster_locations(num_clusters=final_num_clusters,
                                                               ring_space_side_length=ring_space_side_length)
            ring_space_distances = self._extract_distances_in_ring_space(num_clusters=final_num_clusters,
                                                                         cluster_locations=cluster_locations,
                                                                         ring_space_shape=(
                                                                             ring_space_side_length,
                                                                             ring_space_side_length))
            ring_space_map = self._generate_ring_space_map(cluster_voxel_indices=cluster_voxel_indices,
                                                           feature_distances=feature_distances,
                                                           ring_space_distances=ring_space_distances)

            ring_space_image = self._populate_ring_space_using_map(spatially_flattened_pet_data=flattened_head_pet_data,
                                                                   ring_space_map=ring_space_map,
                                                                   ring_space_shape=(ring_space_side_length,
                                                                                     ring_space_side_length))
            denoised_ring_space_image = self._apply_smoothing_in_radon_space(image_data=ring_space_image,
                                                                             kernel=smoothing_kernel)

            flattened_denoised_ring_space_data = denoised_ring_space_image.flatten()
            ring_space_map_only_cluster_data = ring_space_map[ring_space_map != -1]
            denoised_flattened_head_data[ring_space_map_only_cluster_data] = flattened_denoised_ring_space_data[ring_space_map != -1]
            end = time.time()
            logger.debug(f'Time to process cluster {cluster}:\n{end - start} seconds')

        denoised_flattened_pet_data = flattened_pet_data[:, 16] # TODO: Don't hardcode frame 16
        denoised_flattened_pet_data[flattened_head_mask] = denoised_flattened_head_data
        denoised_pet_data = denoised_flattened_pet_data.reshape(self.pet_image.shape[0], self.pet_image.shape[1], self.pet_image.shape[2])

        return denoised_pet_data

    def run(self):
        """"""

    # Static Methods
    @staticmethod
    def _temporal_pca(spatially_flattened_pet_data: np.ndarray,
                      num_components: int) -> np.ndarray:
        """


        Args:
            spatially_flattened_pet_data:
            num_components:

        Returns:

        """
        pca_data = PCA(n_components=num_components).fit_transform(X=spatially_flattened_pet_data)

        return pca_data

    @staticmethod
    def _calculate_ring_space_dimension(num_voxels_in_cluster: int) -> int:
        """
        Determine necessary ring space dimensions to contain all cluster data in the ring.

        Args:
            num_voxels_in_cluster (int): Total number of voxels assigned to the cluster.

        Returns:
            int: The side length of the ring space that can accommodate the cluster data.

        """
        ring_space_dimensions = math.ceil(math.sqrt(2)*math.sqrt(num_voxels_in_cluster))

        return ring_space_dimensions

    @staticmethod
    def _extract_distances_to_cluster_centroids(cluster_data: np.ndarray,
                                                all_cluster_centroids: np.ndarray) -> np.ndarray:
        """Calculate distances from centroids in feature space for each voxel assigned to a cluster.

        Args:
            cluster_data (np.ndarray): 2D array of size (number of voxels in cluster, number of features).
            all_cluster_centroids (np.ndarray): 2D array of size (number of total clusters, number of features). Each
                cluster's feature centroids (mean scores) are stored.

        Returns:
            np.ndarray: 2D array of size (number of voxels in cluster, number of total clusters). For each voxel in the
                cluster, contains the SSD (sum of squared differences) from the feature centroids of all clusters.
        """

        calculate_ssd = lambda features: np.sum((all_cluster_centroids - features) ** 2, axis=1)
        cluster_feature_distances = np.apply_along_axis(calculate_ssd, axis=1, arr=cluster_data)
        return cluster_feature_distances

    @staticmethod
    def _extract_distances_in_ring_space(num_clusters: int,
                                         cluster_locations: np.ndarray,
                                         ring_space_shape: (int, int)) -> np.ndarray:
        """Calculate distances from every cluster's assigned location (not centroid) for each pixel in the ring space"""
        width, height = ring_space_shape
        pixel_cluster_distances = np.zeros(shape=(width, height, num_clusters))
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        grid_coords = np.stack((x_coords, y_coords), axis=-1)
        for i, loc in enumerate(cluster_locations):
            pixel_cluster_distances[..., i] = np.linalg.norm(grid_coords - loc, axis=-1)

        return pixel_cluster_distances

    @staticmethod
    def _define_cluster_locations(num_clusters: int,
                                  ring_space_side_length: int) -> np.ndarray:
        """Given the dimensions of a 'ring space' and the number of clusters, return the location of each cluster"""

        cluster_locations = np.zeros(shape=(num_clusters, 2), dtype=int)
        center = (ring_space_side_length + 1) / 2
        cluster_locations[0] = [math.floor(center), math.floor(center)]
        cluster_angle_increment = 2 * math.pi / (num_clusters - 1)

        for i in range(1, num_clusters):
            x_location = math.floor(center + center * math.cos(i * cluster_angle_increment))
            y_location = math.floor(center + center * math.sin(i * cluster_angle_increment))
            cluster_locations[i] = [x_location, y_location]

        return cluster_locations

    @staticmethod
    @njit(fastmath=True)
    def _generate_ring_space_map(cluster_voxel_indices: np.ndarray,
                                 feature_distances: np.ndarray,
                                 ring_space_distances: np.ndarray) -> np.ndarray:
        """
        Use voxelwise distances from cluster feature centroids to create map to arrange voxels onto 2D 'ring map'.

        Args:
            cluster_voxel_indices (np.ndarray): Array of flattened voxel indices corresponding to PET data assigned to a
                cluster.
            feature_distances (np.ndarray): Array of shape (Number of Voxels in Cluster, Number of Clusters) containing
                distances from cluster feature centroids. Each distance must be the sum of squared differences for all
                features.
            ring_space_distances (np.ndarray): Array of shape
                (Ring Space Side Length, Ring Space Side Length, Number of Clusters) containing the euclidean distances
                from each cluster's assigned location in the ring space.

        Returns:
            np.ndarray: Array containing the voxel indices of the voxel assigned to each pixel in the ring map. Note
                that not all pixels are filled; these are set to np.nan.

        """
        x, y, _ = ring_space_distances.shape

        distance_to_origin_cluster_flat = ring_space_distances[:, :, 0].copy()
        distance_to_origin_cluster_flat = distance_to_origin_cluster_flat.reshape(x * y)

        pixels_emanating_from_center = np.argsort(distance_to_origin_cluster_flat)

        normalized_feature_distances = np.zeros_like(feature_distances)
        for col in range(normalized_feature_distances.shape[1]):
            normalized_feature_distances[:][col] = feature_distances[:][col] / np.linalg.norm(feature_distances[:][col])

        normalized_ring_space_distances = np.zeros_like(ring_space_distances)
        for col in range(ring_space_distances.shape[1]):
            for row in range(ring_space_distances.shape[0]):
                normalized_ring_space_distances[row][col][:] = ring_space_distances[row][col][:] / np.linalg.norm(ring_space_distances[row][col][:])

        image_to_ring_map = np.full_like(distance_to_origin_cluster_flat,
                                         fill_value=-1, dtype=np.uint32) # Maybe use a smaller datatype? Don't need 64-bit int

        for i in range(len(cluster_voxel_indices)):
            pixel_flat_index = pixels_emanating_from_center[i] # O(1)
            pixel_coordinates = (pixel_flat_index % x, pixel_flat_index // x)
            pixel_normalized_ring_space_distances = normalized_ring_space_distances[pixel_coordinates[0], pixel_coordinates[1], :] # TODO: Move this outside of loop
            best_candidate_voxel_index = np.argmax(
                np.dot(normalized_feature_distances, pixel_normalized_ring_space_distances)) # Try changing this to np.dot
            normalized_feature_distances[best_candidate_voxel_index][:] = -10 # O(1)
            image_to_ring_map[pixel_flat_index] = cluster_voxel_indices[best_candidate_voxel_index]

        return image_to_ring_map

    @staticmethod
    def _populate_ring_space_using_map(spatially_flattened_pet_data: np.ndarray,
                                       ring_space_map: np.ndarray,
                                       ring_space_shape: (int, int)) -> np.ndarray:
        """
        Fill pixels in ring space with original PET values using a map.

        Args:
            ring_space_map (np.ndarray): Map of voxel coordinates to pixel coordinates.
            ring_space_shape (tuple): Shape of ring space.

        Returns:
            np.ndarray: Image containing all PET data in a cluster rearranged into ring space.

        """
        populate_pixel_with_pet = lambda a: spatially_flattened_pet_data[a][
            16] if a != -1 else 0  # TODO: Make this do all timeframes
        # TODO: Use logical indexing instead of this
        populated_ring_map = np.array([populate_pixel_with_pet(i) for i in ring_space_map])
        ring_image = populated_ring_map.reshape(ring_space_shape)

        return ring_image

    @staticmethod
    def apply_3_tier_k_means_clustering(flattened_feature_data: np.ndarray,
                                        num_clusters: list[int],
                                        **kwargs) -> (np.ndarray, np.ndarray):
        """
        Separate data into num_clusters clusters using Lloyd's algorithm implemented in sklearn.

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
            cluster_data = flattened_feature_data[cluster_ids == cluster, :]
            _, cluster_ids_temp, _ = k_means(X=cluster_data,
                                             n_clusters=num_clusters[1],
                                             **kwargs)
            cluster_ids_2[cluster_ids == cluster] = cluster_ids[cluster_ids == cluster] * num_clusters[
                1] + cluster_ids_temp

        cluster_ids_3 = np.zeros(shape=cluster_ids.shape)
        for cluster in range(num_clusters[0] * num_clusters[1]):
            cluster_data = flattened_feature_data[cluster_ids_2 == cluster, :]
            centroids_temp, cluster_ids_temp, _ = k_means(X=cluster_data,
                                                          n_clusters=num_clusters[2],
                                                          **kwargs)
            cluster_ids_3[cluster_ids_2 == cluster] = cluster_ids_temp + num_clusters[2] * cluster
            for sub_cluster in range(num_clusters[2]):
                centroids[cluster * num_clusters[2] + sub_cluster, :] = centroids_temp[sub_cluster]

        cluster_ids = cluster_ids_3

        return centroids, cluster_ids

    @staticmethod
    def _generate_2d_gaussian_filter() -> np.ndarray:
        """

        Returns:

        """
        proj_angle = np.linspace(-150, 150, 301)
        proj_position = np.linspace(-3, 3, 7)
        norm_angle = norm.pdf(proj_angle, loc=0, scale=100)
        norm_angle = norm_angle / np.sum(norm_angle)
        angle_smoothing = np.tile(norm_angle[np.newaxis, :], (7, 1))
        norm_position = norm.pdf(proj_position, loc=0, scale=2)
        norm_position = norm_position / np.sum(norm_position)
        position_smoothing = np.tile(norm_position[:, np.newaxis], (1, 301))

        kernel = angle_smoothing * position_smoothing

        return kernel

    @staticmethod
    def _apply_smoothing_in_radon_space(image_data: np.ndarray,
                                        kernel: np.ndarray) -> np.ndarray:
        """
        Radon transform image, apply smoothing, and transform back to original domain

            ring_space_map: """
        theta = np.linspace(0.0, 180.0, 7240)
        radon_transformed_image = radon(image_data, theta=theta)
        smoothed_radon_image = convolve(radon_transformed_image, kernel, mode='constant')
        denoised_cluster_data = iradon(smoothed_radon_image, theta=theta, output_size=image_data.shape[0])

        return denoised_cluster_data

    # Non-Static Methods
    def _prepare_inputs(self,
                        path_to_pet: str,
                        path_to_mri: str,
                        path_to_freesurfer_segmentation: str,
                        path_to_wss: str) -> (Union[nib.nifti1.Nifti1Image, nib.nifti2.Nifti2Image],
                                              Union[nib.nifti1.Nifti1Image, nib.nifti2.Nifti2Image],
                                              Union[nib.nifti1.Nifti1Image, nib.nifti2.Nifti2Image],
                                              np.ndarray,
                                              tuple):
        """
        Read images from files into nibabel Image instances, and ensure all images have the same dimensions as PET.

        Args:
            path_to_pet (str):
            path_to_mri (str):
            path_to_freesurfer_segmentation (str):
            path_to_wss (str):
        """

        images_loaded = []
        images_failed_to_load = []
        errors = []
        image_loader = ImageIO()

        # Verify that all files can be loaded and saved as ndarrays.
        for path in [path_to_pet, path_to_mri, path_to_freesurfer_segmentation, path_to_wss]:
            try:
                images_loaded.append(image_loader.load_nii(path))
            except (FileNotFoundError, OSError) as e:
                images_failed_to_load.append(path)
                errors.append(e)

        # Log errors if any images couldn't be loaded
        if len(images_failed_to_load) > 0:
            raise OSError(
                f'{len(images_failed_to_load)} images could not be loaded. See errors below.\n{print(errors)}')

        # Unpack images
        pet_image, mri_image, segmentation_image, wss_image = images_loaded

        # Extract ndarrays from each image.
        pet_data = image_loader.extract_image_from_nii_as_numpy(pet_image)
        mri_data = image_loader.extract_image_from_nii_as_numpy(mri_image)
        segmentation_data = image_loader.extract_image_from_nii_as_numpy(segmentation_image)
        wss_data = image_loader.extract_image_from_nii_as_numpy(wss_image)
        pet_data_3d_shape = pet_data.shape[:-1]

        if pet_data.ndim != 4:
            raise Exception(
                f'PET data has {pet_data.ndim} dimensions, but 4 is expected. Ensure that you are loading a '
                f'4DPET dataset, not a single frame')

        if (mri_data.shape != pet_data_3d_shape or
            segmentation_data.shape != pet_data_3d_shape or
            wss_data.shape != pet_data_3d_shape):
            raise Exception(f'MRI and/or Segmentation has different dimensions from 3D PET image:\n'
                            f'PET Frame Shape: {pet_data_3d_shape}\n'
                            f'Segmentation Shape: {segmentation_data.shape}\n'
                            f'MRI Shape: {mri_data.shape}.\n'
                            f'Weighted Series Sum Shape: {wss_data.shape}.\n'
                            f'Ensure that all non-PET data is registered to PET space')

        # Extract Head Mask using Weighted Series Sum
        (head_mask_data, *lims) = self._generate_head_mask_from_wss(path_to_wss=path_to_wss)

        return pet_image, mri_image, segmentation_image, head_mask_data, lims

    def _write_cluster_segmentation_to_file(self,
                                            cluster_ids: np.ndarray,
                                            output_path) -> None:
        """

        Args:

        Returns:

        """
        image_io = ImageIO(verbose=True)
        head_mask = self.head_mask
        placeholder_image = np.zeros_like(self.mri_image)
        flat_placeholder_image = placeholder_image.flatten()
        flat_head_mask = head_mask.flatten()
        flat_placeholder_image[flat_head_mask] = cluster_ids
        cluster_image = flat_placeholder_image.reshape(self.mri_image.shape)
        segmentation_image = image_io.extract_np_to_nibabel(image_array=cluster_image,
                                                            header=self.mri_header,
                                                            affine=self.mri_affine)


        nib.save(segmentation_image, output_path)

        return

    def _add_nonbrain_features_to_segmentation(self,
                                               non_brain_mask: np.ndarray) -> np.ndarray:
        """Cluster non-brain and add labels to existing segmentation"""

        segmentation_data = self.segmentation_image
        non_brain_features = self._extract_non_brain_features(non_brain_mask_data=non_brain_mask)
        _, cluster_ids, _ = k_means(X=non_brain_features,
                                    n_clusters=5)

        start_label = np.max(segmentation_data) + 1

        flat_segmentation_data = segmentation_data.flatten()
        flat_non_brain_mask = non_brain_mask.flatten()
        flat_segmentation_data[flat_non_brain_mask] = start_label + cluster_ids

        segmentation_data_with_non_brain = flat_segmentation_data.reshape(segmentation_data.shape)

        return segmentation_data_with_non_brain

    def _extract_non_brain_features(self,
                                    non_brain_mask_data: np.ndarray) -> np.ndarray:
        """

        Returns:

        """

        spatially_flat_non_brain_mask = non_brain_mask_data.flatten()
        flat_mri_data = self.mri_image.flatten()
        spatially_flat_pet = flatten_pet_spatially(self.pet_image)

        non_brain_pet_data = spatially_flat_pet[spatially_flat_non_brain_mask, :]

        pca_data = self._temporal_pca(non_brain_pet_data, num_components=2)

        mri_plus_pca_data = np.zeros(shape=(pca_data.shape[0], pca_data.shape[1] + 1))
        mri_plus_pca_data[:, :-1] = pca_data
        mri_plus_pca_data[:, -1] = flat_mri_data[spatially_flat_non_brain_mask]
        mri_plus_pca_data = zscore(mri_plus_pca_data, axis=0)  # TODO: Verify that this is the right axis with data

        return mri_plus_pca_data

    def _generate_non_brain_mask(self) -> np.ndarray:
        """

        Returns:

        """
        segmentation_data = self.segmentation_image
        head_mask_data = self.head_mask
        brain_mask_data = np.where(segmentation_data > 0, 1, 0)
        non_brain_mask_data = head_mask_data - brain_mask_data

        return non_brain_mask_data.astype(bool)

    def _generate_head_mask_from_wss(self,
                                     path_to_wss: str) -> (np.ndarray, tuple, tuple, tuple):
        """
        Function to extract 3D head mask PET data using basic morphological methods.

        Args:
            path_to_wss (str):

        Returns:
            np.ndarray: 3D binary mask corresponding to the head voxels.
        """
        out_image_path = '/export/scratch1/oestreichk/Data/cropped_wss.nii.gz' # Update this when pipelines drop

        cropper = SimpleAutoImageCropper(input_image_path=path_to_wss,
                                         out_image_path=out_image_path)
        x_lims, y_lims, z_lims = cropper.get_index_pairs_for_all_dims(img_obj=cropper.input_img_obj)
        image_loader = ImageIO()
        cropped_image = image_loader.load_nii(out_image_path)
        cropped_data = image_loader.extract_image_from_nii_as_numpy(cropped_image)
        binary_cropped_data = binarize_image_with_threshold(input_image_numpy=cropped_data,
                                                            lower_bound=500)
        binary_cropped_data = binary_closing(binary_cropped_data)
        mask_data = binary_fill_holes(binary_cropped_data)

        return mask_data, x_lims, y_lims, z_lims

    def weighted_sum_smoothed_image_iterations(self):
        """
        Weight smoothed images (one from each iteration) by cluster 'belongingness' with respect to MRI."""
        pass

def flatten_pet_spatially(pet_data: np.ndarray) -> np.ndarray:
    """
    Flatten spatial dimensions (using C index order) of 4D PET and return 2D array (numVoxels x numFrames).

    Args:
        pet_data (np.ndarray): 4D PET data.

    Returns:
        np.ndarray: Array of size (M,N) where M is total number of voxels in a 3D frame of the PET and N is the number
            of frames.

    """

    num_voxels = np.prod(pet_data.shape[:-1])
    flattened_pet_data = pet_data.reshape(num_voxels, -1)

    return flattened_pet_data



