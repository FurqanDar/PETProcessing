"""
The 'image_operations_4d' module provides several functions used to do preprocessing
on 4D PET imaging series. These functions typically take one or more paths to imaging
data in NIfTI format, and save modified data to a NIfTI file, and may return the
modified imaging array as output.

TODOs:
    * (weighted_series_sum) Refactor the DecayFactor key extraction into its own function
    * (weighted_series_sum) Refactor verbose reporting into the class as it is unrelated to
      computation
    * (write_tacs) Shift to accepting color-key dictionaries rather than a file path.
    * (extract_tac_from_4dnifty_using_mask) Write the number of voxels in the mask, or the
      volume of the mask. This is necessary for certain analyses with the resulting tacs,
      such as finding the average uptake encompassing two regions.
    * Methods that create new images should copy over a previous metadata file, if one exists,
      and create a new one if it does not.

"""
import os
from scipy.ndimage import center_of_mass
import nibabel
import numpy as np
import ants
from ..utils import image_io, math_lib
from ..preproc import motion_corr


def crop_image(input_image_path: str,
               out_image_path: str,
               x_dim: int=256,
               y_dim: int=256):
    """
    Crops an image in the X and Y axes to exclude voxels outside of the head. This is done to
    reduce the size of the image for faster processing, while preserving scientifically
    valuable information. Preserves dimension along Z and time axes.

    The returned, cropped image is centered on the "center of mass" computed using 
    :py:func:`scipy.ndimage.center_of_mass`. If the image is 3D, the center of mass is computed
    directly on the image array. If the image is 4D, the image is first averaged over the time axis
    before computing the center of mass.

    Args:
        input_image_path (str): Path to input image to be cropped.
        out_image_path (str): Path to which cropped image is saved.
        x_dim (int): Size of the X axis of the returned image. Default value 256.
        y_dim (int): Size of the Y axis of the returned image. Default value 256.
    
    Returns:
        cropped_image (nibabel.nifti1.Nifti1Image): The cropped image.
    """
    image = nibabel.load(input_image_path)
    image_np = image.get_fdata()

    if len(image_np.shape)<4:
        center = center_of_mass(image_np)
    else:
        image_mean = np.mean(image_np,axis=-1)
        center = center_of_mass(image_mean)

    center = np.round(center).astype('int')
    x_half = x_dim // 2
    y_half = y_dim // 2

    cropped_image = image.slicer[center[0]-x_half:center[0]+x_half,
                                 center[1]-y_half:center[1]+y_half]
    nibabel.save(cropped_image,out_image_path)
    image_io.safe_copy_meta(input_image_path=input_image_path,
                            out_image_path=out_image_path)
    return cropped_image


def brain_mask(input_image_4d_path: str,
               out_image_path: str,
               atlas_image_path: str,
               atlas_mask_path: str,
               motion_target_option='mean_image',
               half_life: float=None):
    """
    Create a brain mask for a PET image. Create target PET image, which is then warped to a
    provided anatomical atlas. The transformation to atlas space is then applied to transform a
    provided mask in atlas space into PET space. This mask can then by used in various operations.

    Args:
        input_image_4d_path (str): Path to input 4D PET image.
        out_image_path (str): Path to which brain mask in PET space is written.
        atlas_image_path (str): Path to anatomical atlas image.
        atlas_mask_path (str): Path to brain mask in atlas space.
        motion_target_option: Used to determine 3D target in PET space. Default 'mean_image'.
    
    Note:
        Requires access to an anatomical atlas or scan with a corresponding brain mask on said
        anatomical data. FSL users can use the MNI152 atlas and mask available at 
        $FSLDIR/data/standard/.
    """
    atlas = ants.image_read(atlas_image_path)
    atlas_mask = ants.image_read(atlas_mask_path)
    pet_ref = ants.image_read(motion_corr.determine_motion_target(
        motion_target_option=motion_target_option,
        input_image_4d_path=input_image_4d_path,
        half_life=half_life
    ))
    xfm = ants.registration(
        fixed=atlas,
        moving=pet_ref,
        type_of_transform='SyN'
    )
    mask_on_pet = ants.apply_transforms(
        fixed=pet_ref,
        moving=atlas_mask,
        transformlist=xfm['invtransforms'],
        interpolator='nearestNeighbor'
    )
    mask = mask_on_pet.get_mask()
    ants.image_write(image=mask,filename=out_image_path)


def extract_tac_from_nifty_using_mask(input_image_4d_numpy: np.ndarray,
                                      segmentation_image_numpy: np.ndarray,
                                      region: int,
                                      verbose: bool) -> np.ndarray:
    """
    Creates a time-activity curve (TAC) by computing the average value within a region, for each 
    frame in a 4D PET image series. Takes as input a PET image, which has been registered to
    anatomical space, a segmentation image, with the same sampling as the PET, and a list of values
    corresponding to regions in the segmentation image that are used to compute the average
    regional values. Currently, only the mean over a single region value is implemented.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image, registered to anatomical space.
        segmentation_image_path (str): Path to a .nii or .nii.gz file containing a 3D segmentation
            image, where integer indices label specific regions. Must have same sampling as PET
            input.
        region (int): Value in the segmentation image corresponding to a region
            over which the TAC is computed.
        verbose (bool): Set to ``True`` to output processing information.

    Returns:
        tac_out (np.ndarray): Mean of PET image within regions for each frame in 4D PET series.

    Raises:
        ValueError: If the segmentation image and PET image have different
            sampling.
    """

    pet_image_4d = input_image_4d_numpy
    if len(pet_image_4d.shape)==4:
        num_frames = pet_image_4d.shape[3]
    else:
        num_frames = 1
    seg_image = segmentation_image_numpy

    if seg_image.shape[:3]!=pet_image_4d.shape[:3]:
        raise ValueError('Mis-match in image shape of segmentation image '
                         f'({seg_image.shape}) and PET image '
                         f'({pet_image_4d.shape[:3]}). Consider resampling '
                         'segmentation to PET or vice versa.')

    tac_out = np.zeros(num_frames, float)
    if verbose:
        print(f'Running TAC for region index {region}')
    masked_voxels = (seg_image > region - 0.1) & (seg_image < region + 0.1)
    masked_image = pet_image_4d[masked_voxels].reshape((-1, num_frames))
    tac_out = np.mean(masked_image, axis=0)
    return tac_out


def threshold(input_image_numpy: np.ndarray,
              lower_bound: float=-np.inf,
              upper_bound: float=np.inf):
    """
    Threshold an image above and/or below a pair of values.
    """
    bounded_image = np.zeros(input_image_numpy.shape)
    bounded_image_where = (input_image_numpy > lower_bound) & (input_image_numpy < upper_bound)
    bounded_image[bounded_image_where] = input_image_numpy[bounded_image_where]
    return bounded_image


def suvr(input_image_path: str,
         segmentation_image_path: str,
         ref_region: int,
         out_image_path: str,
         verbose: bool):
    """
    Computes an ``SUVR`` (Standard Uptake Value Ratio) by taking the average of
    an input image within a reference region, and dividing the input image by
    said average value.

    Args:
        input_image_path (str): Path to 3D weighted series sum or other
            parametric image on which we compute SUVR.
        segmentation_image_path (str): Path to segmentation image, which we use
            to compute average uptake value in the reference region.
        ref_region (int): Region number mapping to the reference region in the
            segmentation image.
        out_image_path (str): Path to output image file which is written to.
        verbose (bool): Set to ``True`` to output processing information.
    """
    pet_nibabel = nibabel.load(filename=input_image_path)
    pet_image = pet_nibabel.get_fdata()
    seg_nibabel = nibabel.load(filename=segmentation_image_path)
    seg_image = seg_nibabel.get_fdata()

    if len(pet_image.shape)!=3:
        raise ValueError("SUVR input image is not 3D. If your image is dynamic"
                         ", try running 'weighted_series_sum' first.")

    ref_region_avg = extract_tac_from_nifty_using_mask(input_image_4d_numpy=pet_image,
                                                       segmentation_image_numpy=seg_image,
                                                       region=ref_region,
                                                       verbose=verbose)

    suvr_image = pet_image / ref_region_avg[0]

    out_image = nibabel.nifti1.Nifti1Image(dataobj=suvr_image,
                                           affine=pet_nibabel.affine,
                                           header=pet_nibabel.header)
    nibabel.save(img=out_image,filename=out_image_path)

    image_io.safe_copy_meta(input_image_path=input_image_path,
                            out_image_path=out_image_path)

    return out_image


def gauss_blur(input_image_path: str,
               blur_size_mm: float,
               out_image_path: str,
               verbose: bool,
               use_fwhm: bool=True):
    """
    Blur an image with a 3D Gaussian kernal of a provided size in mm. Extracts
    Gaussian sigma from provided blur size, and voxel sizes in the image
    header. :py:func:`scipy.ndimage.gaussian_filter` is used to apply blurring.
    Uses wrapper around :meth:`gauss_blur_computation`.
    
    Args:
        input_image_path (str): Path to 3D or 4D input image to be blurred.
        blur_size_mm (float): Sigma of the Gaussian kernal in mm.
        out_image_path (str): Path to save the blurred output image.
        verbose (bool): Set to ``True`` to output processing information.
        use_FWHM (bool): If ``True``, ``blur_size_mm`` is interpreted as the
            FWHM of the Gaussian kernal, rather than the standard deviation.

    Returns:
        out_image (nibabel.nifti1.Nifti1Image): Blurred image in nibabel format.
    """
    input_nibabel = nibabel.load(filename=input_image_path)
    input_image = input_nibabel.get_fdata()
    input_zooms = input_nibabel.header.get_zooms()

    blur_image = math_lib.gauss_blur_computation(input_image=input_image,
                                                 blur_size_mm=blur_size_mm,
                                                 input_zooms=input_zooms,
                                                 use_fwhm=use_fwhm)

    out_image = nibabel.nifti1.Nifti1Image(dataobj=blur_image,
                                           affine=input_nibabel.affine,
                                           header=input_nibabel.header)
    nibabel.save(img=out_image,filename=out_image_path)

    image_io.safe_copy_meta(input_image_path=input_image_path,out_image_path=out_image_path)

    if verbose:
        print(f'Blurred image saved to {out_image_path}.')

    return out_image


def roi_tac(input_image_4d_path: str,
            roi_image_path: str,
            region: int,
            out_tac_path: str,
            verbose: bool,
            time_frame_keyword: str = 'FrameReferenceTime'):
    """
    Function to write Tissue Activity Curves for a single region, given a mask,
    4D PET image, and region mapping. Computes the average of the PET image 
    within each region. Writes a tsv table with region name, frame start time,
    and mean value within region.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    tac_extraction_func = extract_tac_from_nifty_using_mask
    pet_numpy = nibabel.load(input_image_4d_path).get_fdata()
    seg_numpy = nibabel.load(roi_image_path).get_fdata()


    extracted_tac = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                        segmentation_image_numpy=seg_numpy,
                                        region=region,
                                        verbose=verbose)
    region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
    header_text = 'mean_activity'
    np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')


def write_tacs(input_image_4d_path: str,
               label_map_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               verbose: bool,
               time_frame_keyword: str = 'FrameReferenceTime'):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and label map. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
    regions_abrev = label_map['abbreviation']
    regions_map = label_map['mapping']

    tac_extraction_func = extract_tac_from_nifty_using_mask
    pet_numpy = nibabel.load(input_image_4d_path).get_fdata()
    seg_numpy = nibabel.load(segmentation_image_path).get_fdata()

    for i, _maps in enumerate(label_map['mapping']):
        extracted_tac = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                            segmentation_image_numpy=seg_numpy,
                                            region=int(regions_map[i]),
                                            verbose=verbose)
        region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
        header_text = f'{time_frame_keyword}\t{regions_abrev[i]}_mean_activity'
        out_tac_path = os.path.join(out_tac_dir, f'tac-{regions_abrev[i]}.tsv')
        np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')


class SimpleAutoImageCropper(object):
    r"""
    Class for automatically cropping 3D or 4D medical images based on pixel intensity thresholds.

    This class provides functionality to load a medical image, determine the meaningful regions
    by thresholding, and crop the image to remove regions outside these boundaries.
    It also supports copying metadata from the original image.

    Attributes:
        input_image_path (str): The file path to the input image.
        out_image_path (str): The file path to save the cropped image.
        thresh (float): The threshold value used to determine the boundaries.
        verbose (bool): If True, prints information about image shapes.
        input_img_obj (nibabel.Nifti1Image): The loaded input image object.
        crop_img_obj (nibabel.Nifti1Image): The cropped image object.

    Example:
        
        .. code-block:: python
        
            from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
            cropper = SimpleAutoImageCropper(
                input_image_path='path/to/input_image.nii',
                out_image_path='path/to/output_image.nii',
                thresh_val=0.01,
                verbose=True,
                copy_metadata=True
            )

    See Also:
        - :meth:`get_cropped_image`
        - :meth:`get_index_pairs_for_all_dims`
        - :meth:`get_left_and_right_boundary_indices_for_threshold`
        - :meth:`gen_line_profile`
        
        
    """
    def __init__(self,
                 input_image_path: str,
                 out_image_path: str,
                 thresh_val: float = 1.0e-2,
                 verbose: bool = True,
                 copy_metadata: bool = True
                 ):
        r"""
        Initializes the SimpleAutoImageCropper with input image path, output image path, and other
        parameters.

        Loads the input image, generates the cropped image using the specified threshold, and saves
        it to the output path.

        Args:
            input_image_path (str): The file path to the input image.
            out_image_path (str): The file path to save the cropped image.
            thresh_val (float, optional): The threshold value used to determine the boundaries.
                Must be less than 0.5. Defaults to 1e-2.
            verbose (bool, optional): If True, prints information about image shapes. Defaults to
                True.
            copy_metadata (bool, optional): If True, copies metadata from the original image to the
                cropped image. Defaults to True.

        Raises:
            AssertionError: If the `thresh_val` is not less than 0.5.

        Example:
            
            .. code-block:: python

                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
                cropper = SimpleAutoImageCropper(
                    input_image_path='path/to/input_image.nii',
                    out_image_path='path/to/output_image.nii',
                    thresh_val=0.01,
                    verbose=True,
                    copy_metadata=True
                )

        """
        self.input_image_path = input_image_path
        self.out_image_path = out_image_path
        self.thresh = thresh_val
        self.verbose = verbose
        self.input_img_obj = nibabel.load(self.input_image_path)
        self.crop_img_obj = self.get_cropped_image(img_obj=self.input_img_obj, thresh=self.thresh)

        nibabel.save(filename=self.out_image_path, img=self.crop_img_obj)
        if copy_metadata:
            image_io.safe_copy_meta(input_image_path=self.input_image_path,
                                    out_image_path=self.out_image_path)

        if verbose:
            print(f"(info): Input image has shape:  {self.input_img_obj.shape}")
            print(f"(info): Output image has shape: {self.crop_img_obj.shape}")


    @staticmethod
    def gen_line_profile(img_arr: np.ndarray, dim: str = 'x'):
        r"""
        Generates a line profile by averaging the pixel intensities along specified dimensions.

        This function computes the mean pixel intensities along a specified dimension (x, y, or z)
        of a 3D or 4D image array.

        Args:
            img_arr (np.ndarray): The input image array.
            dim (str, optional): The dimension along which to compute the line profile.
                                 Must be one of 'x', 'y', or 'z'. Case-insensitive. Defaults to 'x'.

        Returns:
            np.ndarray: The computed line profile as a 1D array.

        Raises:
            AssertionError: If `dim` is not one of 'x', 'y', or 'z'.

        Example:
            
            .. code-block:: python
            
                import numpy as np
                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
                img_arr = np.random.rand(100, 100, 100)  # Example 3D array
                x_profile = SimpleAutoImageCropper.gen_line_profile(img_arr=img_arr, dim='x')
                print(x_profile)
        
        """
        tmp_dim = dim.lower()
        assert tmp_dim in ['x', 'y', 'z']
        if tmp_dim == 'x':
            return np.mean(img_arr, axis=(1, 2))
        if tmp_dim == 'y':
            return np.mean(img_arr, axis=(0, 2))
        if tmp_dim == 'z':
            return np.mean(img_arr, axis=(0, 1))

    @staticmethod
    def get_left_and_right_boundary_indices_for_threshold(line_prof: np.ndarray,
                                                          thresh: float = 1e-2):
        r"""
        Determines the left and right boundary indices above a threshold in a line profile.

        This function identifies the indices where the normalized line profile crosses the 
        specified threshold value, indicating the boundaries of the region of interest.

        Args:
            line_prof (np.ndarray): The input line profile as a 1D array.
            thresh (float, optional): The threshold value for determining boundaries. Must be less
                than 0.5. Defaults to 1e-2.

        Returns:
            tuple: A tuple containing the left and right boundary indices (left_index, right_index).

        Raises:
            AssertionError: If the `thresh` value is not less than 0.5.

        Example:
            
            .. code-block:: python
            
                import numpy as np
                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper as Crop

                line_prof = np.random.rand(100)  # Example normalized line profile
                boundaries = Crop.get_left_and_right_boundary_indices_for_threshold
                left_index, right_index = boundaries(line_prof=line_prof, thresh=0.01)
                print(left_index, right_index)
        
        """
        assert thresh < 0.5
        norm_prof = line_prof / np.max(line_prof)
        l_ind, r_ind = np.argwhere(norm_prof > thresh).T[0][[0, -1]]
        return l_ind, r_ind

    @staticmethod
    def get_index_pairs_for_all_dims(img_obj: nibabel.Nifti1Image, thresh: float = 1e-2):
        r"""
        Gets the boundary indices for each dimension of the input image based on a threshold value.

        This function computes the left and right boundary indices for all dimensions (x, y, z)
        by generating line profiles and applying a threshold to identify meaningful regions.

        Args:
            img_obj (nibabel.Nifti1Image): The input NIfTI image object.
            thresh (float, optional): The threshold value used to determine the boundaries.
                                      Must be less than 0.5. Defaults to 1e-2.

        Returns:
            tuple: A tuple of boundary index pairs for each dimension, formatted as
                   ((x_left, x_right), (y_left, y_right), (z_left, z_right)).

        Raises:
            AssertionError: If the `thresh` value is not less than 0.5.
            
        See Also:
            - :meth:`get_index_pairs_for_all_dims`

        Example:
            
            .. code-block:: python
            
                import nibabel as nib
                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
                input_image_path = 'path/to/input_image.nii'
                img_obj = nib.load(input_image_path)
    
                boundaries = SimpleAutoImageCropper.get_index_pairs_for_all_dims(img_obj=img_obj,
                                                                                 thresh=0.01)
                print(boundaries)
        
        """
        if len(img_obj.shape) > 3:
            tmp_data = np.mean(img_obj.get_fdata(), axis=-1)
        else:
            tmp_data = img_obj.get_fdata()

        prof_func = SimpleAutoImageCropper.gen_line_profile
        index_func = SimpleAutoImageCropper.get_left_and_right_boundary_indices_for_threshold

        x_line_prof = prof_func(img_arr=tmp_data, dim='x')
        x_left, x_right = index_func(line_prof=x_line_prof, thresh=thresh)

        y_line_prof = prof_func(img_arr=tmp_data, dim='y')
        y_left, y_right = index_func(line_prof=y_line_prof, thresh=thresh)

        z_line_prof = prof_func(img_arr=tmp_data, dim='z')
        z_left, z_right = index_func(line_prof=z_line_prof, thresh=thresh)

        return (x_left, x_right), (y_left, y_right), (z_left, z_right)

    @staticmethod
    def get_cropped_image(img_obj: nibabel.Nifti1Image, thresh: float = 1e-2):
        r"""
        Crops the input medical image based on a threshold value.

        This function determines the boundaries of the meaningful regions in the input image
        by thresholding and then crops the image to remove regions outside these boundaries.

        Args:
            img_obj (nibabel.Nifti1Image): The input NIfTI image object to be cropped.
            thresh (float, optional): The threshold value used to determine the boundaries.
                                      Must be less than 0.5. Defaults to 1e-2.

        Returns:
            nibabel.Nifti1Image: The cropped NIfTI image object.

        Raises:
            AssertionError: If the `thresh` value is not less than 0.5.
            
        See Also:
            - :meth:`get_index_pairs_for_all_dims`
            - :meth:`get_left_and_right_boundary_indices_for_threshold`
            - :meth:`gen_line_profile`

        Example:
            
            .. code-block:: python
            
                import nibabel as nib
                from petpal.preproc.image_operations_4d import SimpleAutoImageCropper
    
                input_image_path = 'path/to/input_image.nii'
                img_obj = nib.load(input_image_path)
    
                cropped_img = SimpleAutoImageCropper.get_cropped_image(img_obj=img_obj, thresh=0.01)
                nib.save(cropped_img, 'path/to/output_image.nii')
        
        """
        (x_l, x_r), (y_l, y_r), (z_l, z_r) = SimpleAutoImageCropper.get_index_pairs_for_all_dims(img_obj=img_obj,
                                                                                                 thresh=thresh)

        return img_obj.slicer[x_l:x_r, y_l:y_r, z_l:z_r, ...]
