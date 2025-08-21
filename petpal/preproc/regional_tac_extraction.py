"""
Regional TAC extraction
"""
import re
import os
from collections.abc import Callable
import pathlib
import numpy as np
import ants
import pandas as pd

from .segmentation_tools import combine_regions_as_mask, unique_segmentation_labels
from ..utils import image_io
from ..utils.scan_timing import ScanTimingInfo
from ..utils.useful_functions import check_physical_space_for_ants_image_pair
from ..utils.time_activity_curve import TimeActivityCurve


def extract_roi_voxel_tacs_from_image_using_mask(input_image: ants.core.ANTsImage,
                                                 mask_image: ants.core.ANTsImage,
                                                 verbose: bool = False) -> np.ndarray:
    """
    Function to extract ROI voxel tacs from an image using a mask image.

    This function returns all the voxel TACs, and unlike
    :func:`extract_mean_roi_tac_from_nifti_using_segmentation` does not calculate the mean over
    all the voxels.

    Args:
        input_image (ants.core.ANTsImage): Input 4D-image from which to extract ROI voxel tacs.
        mask_image (ants.core.ANTsImage): Mask image which determines which voxels to extract.
        verbose (bool, optional): If True, prints information about the shape of extracted voxel
            tacs.

    Returns:
        out_voxels (np.ndarray): Array of voxel TACs of shape (num_voxels, num_frames)

    Raises:
         AssertionError: If input image is not 4D-image.
         AssertionError: If mask image is not in the same physical space as the input image.

    Example:

        .. code-block:: python

            import ants
            import numpy as np

            from petpal.preproc import regional_tac_extraction
            tac_func = regional_tac_extraction.extract_roi_voxel_tacs_from_image_using_mask
            
            # Read images
            pet_img = ants.image_read("/path/to/pet.nii.gz")
            masked_region_img = ants.image_read("/path/to/mask_region.nii.gz")

            # Run ROI extraction and save
            time_series = tac_func(input_image=pet_img, mask_image=masked_region_img).T
            np.savetxt("time_series.tsv", time_series, delimiter='\t')
            
    """
    assert len(input_image.shape) == 4, "Input image must be 4D."
    assert check_physical_space_for_ants_image_pair(input_image, mask_image), (
        "Images must have the same physical dimensions.")

    out_voxels = apply_mask_4d(input_arr=input_image.numpy(),
                               mask_arr=mask_image.numpy(),
                               verbose=verbose)
    return out_voxels


def apply_mask_4d(input_arr: np.ndarray,
                  mask_arr: np.ndarray,
                  verbose: bool = False) -> np.ndarray:
    """
    Function to extract ROI voxel tacs from an array using a mask array.

    This function applies a 3D mask to a 4D image, returning the time series for each voxel in a
    single flattened numpy array.

    Args:
        input_arr (np.ndarray): Input 4D-image from which to extract ROI voxel tacs.
        mask_arr (np.ndarray): Mask image which determines which voxels to extract.
        verbose (bool, optional): If True, prints information about the shape of extracted voxel
            tacs.

    Returns:
        out_voxels (np.ndarray): Time series of each voxel in the mask, as a flattened numpy array.

    Raises:
         AssertionError: If input array is not 4D.
         AssertionError: If input and mask array shapes are mismatched.

    Example:

        .. code-block:: python

            import ants
            import numpy as np

            from petpal.preproc.regional_tac_extraction import apply_mask_4d
            
            # Read images
            pet_img = ants.image_read("/path/to/pet.nii.gz")
            masked_region_img = ants.image_read("/path/to/mask_region.nii.gz")

            # Get underlying arrays
            pet_arr = pet_img.numpy()
            masked_region_arr = masked_region_img.numpy()

            # Run ROI extraction and save
            time_series = apply_mask_4d(input_arr=pet_arr, mask_arr=masked_region_arr).T
            np.savetxt("time_series.tsv", time_series, delimiter='\t')

    """
    assert len(input_arr.shape) == 4, "Input array must be 4D."
    assert input_arr.shape[:3] == mask_arr.shape, (
            "Array must have the same physical dimensions.")

    x_inds, y_inds, z_inds = mask_arr.nonzero()
    out_voxels = input_arr[x_inds, y_inds, z_inds, :]
    if verbose:
        print(f"(ImageOps): Output TACs have shape {out_voxels.shape}")
    return out_voxels


def voxel_average_w_uncertainty(pet_voxels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spatially average flattened PET voxels and get the standard deviation as well.
    
    Args:
        pet_voxels (np.ndarray): 1D or 2D array of PET voxels.
    
    Returns:
        average_w_uncertainty (tuple[np.ndarray, np.ndarray]): Average and standard deviation of
            PET voxels."""
    pet_average = pet_voxels.mean((0))
    pet_uncertainty = pet_voxels.std((0))
    return (pet_average, pet_uncertainty)


def write_tacs(input_image_path: str,
               label_map_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               out_tac_prefix: str = '',
               verbose: bool = False):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and label map. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.

    Args:
        input_image_path (str): Path to the 4D PET image from which regional TACs will be
            extracted.
        label_map_path (str): Path to the dseg file linking regions to their mappings in the
            segmentation image.
        segmentation_image_path (str): Path to the segmentation image containing ROIs. Must be in
            the same space as input_image.
        out_tac_dir (str): Path to the directory where regional TACs will be written to.
        out_tac_prefix (str): Prefix for output TAC files. Typically the participant ID.
        verbose (bool): Set to True to print processing info. Default False.
    """
    label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
    regions_abrev = label_map['abbreviation']
    regions_map = label_map['mapping']

    pet_numpy = ants.image_read(input_image_path).numpy()
    seg_numpy = ants.image_read(segmentation_image_path).numpy()

    scan_timing_info = ScanTimingInfo.from_nifti(image_path=input_image_path)
    tac_times_in_mins = scan_timing_info.center_in_mins

    for i, _maps in enumerate(label_map['mapping']):
        region_mask = combine_regions_as_mask(segmentation_img=seg_numpy,
                                              label=int(regions_map[i]))
        pet_masked_region = apply_mask_4d(input_arr=pet_numpy,
                                          mask_arr=region_mask)
        extracted_tac, tac_uncertainty = voxel_average_w_uncertainty(pet_masked_region)
        region_tac = TimeActivityCurve(times=tac_times_in_mins,
                                       activity=extracted_tac,
                                       uncertainty=tac_uncertainty)
        if out_tac_prefix:
            out_tac_path = os.path.join(out_tac_dir,
                                        f'{out_tac_prefix}_seg-{regions_abrev[i]}_tac.tsv')
        else:
            out_tac_path = os.path.join(out_tac_dir, f'seg-{regions_abrev[i]}_tac.tsv')
        region_tac.to_tsv(filename=out_tac_path)
    if verbose:
        print('Finished writing TACs.')


def roi_tac(input_image_4d_path: str,
            roi_image_path: str,
            region: list[int] | int,
            out_tac_path: str,
            time_frame_keyword: str = 'FrameReferenceTime'):
    """
    Function to write Tissue Activity Curves for a single region, given a mask,
    4D PET image, and region mapping. Computes the average of the PET image 
    within each region. Writes a tsv table with region name, frame start time,
    and mean value within region.

    Args:
        input_image_path (str): Path to the 4D PET image from which the regional TAC will be
            extracted.
        roi_image_path (str): Path to the segmentation image containing ROIs. Must be in the same
            space as input_image.
        region (list[int] | int): The region or regions that will be extracted as a TAC.
        out_tac_dir (str): Path to the TSV where the regional TAC will be written to.
        time_frame_keyword (str): Keyword corresponding to either 'FrameReferenceTime' or
            'FrameTimesStart' to get the frame timing. Default 'FrameReferenceTime'.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.load_metadata_for_nifti_with_same_filename(input_image_4d_path)
    pet_numpy = ants.image_read(input_image_4d_path).numpy()
    seg_numpy = ants.image_read(roi_image_path).numpy()

    region_mask = combine_regions_as_mask(segmentation_img=seg_numpy,
                                          label=region)
    pet_masked_region = apply_mask_4d(input_arr=pet_numpy,
                                      mask_arr=region_mask)
    extracted_tac, tac_uncertainty = voxel_average_w_uncertainty(pet_masked_region)
    region_tac = TimeActivityCurve(times=pet_meta[time_frame_keyword],
                                   activity=extracted_tac,
                                   uncertainty=tac_uncertainty)
    region_tac.to_tsv(filename=out_tac_path)


class WriteRegionalTacs:
    """
    Write regional TACs

    Attrs:
        pet_arr (np.ndarray): Numpy array containing 4D PET data.
        seg_arr (np.ndarray): Numpy array containing 3D discrete segmentation data.
        tac_extraction_func (Callable): A function that takes a 2D M x N numpy array with M voxels
            and N time frames as well as any number of optional keyword arguments and returns a
            tuple of 1D N-length numpy arrays with the calculated TAC and uncertainty.
        scan_timing (ScanTimingInfo): Scan timing for the input PET image.
        region_names (list): Names of regions to use in the analysis.
        region_maps (list): Region mappings to use in the analysis, corresponding 1-1 with
            region_names.


    Example:

        .. code-block:: python

            from petpal.preproc.regional_tac_extraction import WriteRegionalTacs
            from petpal.utils.bids_utils import gen_bids_like_filepath, gen_bids_like_dir_path
            
            pet_image_path = gen_bids_like_filepath(sub_id='001',
                                                    ses_id='01',
                                                    suffix='pet',
                                                    ext='.nii.gz')
            seg_image_path = gen_bids_like_filepath(sub_id='001',
                                                    ses_id='01',
                                                    bids_dir='../derivatives/petpal/'
                                                    suffix='seg',
                                                    modality='seg',
                                                    space='pet',
                                                    ext='.nii.gz')
            tac_output_dir = gen_bids_like_dir_path(sub_id='001',
                                                    ses_id='01',
                                                    modality='tacs',
                                                    sup_dir='../derivatives/petpal/')
            tac_calculator = WriteRegionalTacs(input_image_path=pet_image_path,
                                               segmentation_path=seg_image_path,
                                               label_map_path='dseg.tsv')
            tac_calculator(out_tac_dir=tac_output_dir,
                           out_tac_prefix='sub-001_ses-01',
                           one_tsv_per_region=False)

    """
    def __init__(self,
                 input_image_path: str | pathlib.Path,
                 segmentation_path: str | pathlib.Path,
                 label_map_path: str | pathlib.Path):
        """Initialize WriteRegionalTacs.
        
        Args:
            input_image_path (str | pathlib.Path): Path to input 4D PET image.
            segmentation_path (str | pathlib.Path): Path to 3D discrete segmentation image. Must
                match input PET image space.
            label_map_path (str | pathlib.Path): Path to label map 'dseg.tsv' file containing names
                and mapping for regions of interest in the study."""
        self.pet_arr = ants.image_read(filename=input_image_path).numpy()
        self.seg_arr = ants.image_read(filename=segmentation_path).numpy()

        self.tac_extraction_func = voxel_average_w_uncertainty
        self.scan_timing = ScanTimingInfo.from_nifti(input_image_path)

        label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
        self.region_names = [self.str_to_camel_case(label) for label in label_map['abbreviation']]
        self.region_maps = label_map['mapping'].to_list()

    def set_tac_extraction_func(self, tac_extraction_func: Callable):
        """Sets the tac extraction function used to a different function.
        
        The selected function must take a 2D array of the masked voxels as input, and return the 
        calculated activity and uncertainty across the masked voxels outputs.

        Args:
            tac_extraction_func (Callable): Function that takes a 2D M x N numpy array with M
                voxels and N time frames as well as any number of optional keyword arguments and
                returns a tuple of 1D N-length numpy arrays with the calculated TAC and
                uncertainty.
        """
        self.tac_extraction_func = tac_extraction_func


    @staticmethod
    def capitalize_first_char_of_str(input_str: str) -> str:
        """
        Capitalize only the first character of a string, leaving the remainder unchanged.
        Args:
            input_str (str): The string to capitalize the first character of.
        Returns:
            output_str (str): The string with only the first character capitalized.
        """
        output_str = input_str[0].capitalize()+input_str[1:]
        return output_str


    @staticmethod
    def str_to_camel_case(input_str) -> str:
        """
        Take a string and return the string converted to camel case.

        Special characters (? * - _ / \\) are removed and treated as word separaters. Different
        words are then capitalized at the first character, leaving other alphanumeric characters
        unchanged.

        Args:
            input_str (str): The string to convert to camel case and remove special characters.
        Returns:
            camel_case_str (str): The string converted to camel case (e.g. CamelCase) with special
                characters removed.
        """
        split_str = re.split(r'[-_?*/\\]', input_str)
        capped_split_str = []
        capitalize_first = WriteRegionalTacs.capitalize_first_char_of_str
        for part in split_str:
            capped_str = capitalize_first(input_str=part)
            capped_split_str += [capped_str]
        camel_case_str = ''.join(capped_split_str)
        return camel_case_str


    def find_label_name(self, label: int) -> str:
        """Find the name for a label based on the provided label map. If a name is not found,
        return 'UNK' followed by the label index.
        
        Args:
            label (int): Label mapping for a region.
        
        Returns:
            region_name (str): Name of the region corresponding to the provided label."""
        try:
            label_map_loc = self.region_maps.index(label)
            region_name = self.region_names[label_map_loc]
        except ValueError:
            region_name = f'UNK{label}'
        return region_name


    def extract_tac(self,region_mapping: int | list[int], **tac_calc_kwargs) -> TimeActivityCurve:
        """
        Run self.tac_extraction_func on one region and save results to image.

        Args:
            region_mapping (int | list[int]): The integer ID or IDs corresponding to the ROI.
            **tac_calc_kwargs: Additional keyword arguments passed on to tac_extraction_func.
    
        Returns:
            region_tac (TimeActivityCurve): The calculated TAC for the region. 
        """
        region_mask = combine_regions_as_mask(segmentation_img=self.seg_arr,
                                              label=region_mapping)
        pet_masked_region = apply_mask_4d(input_arr=self.pet_arr,
                                          mask_arr=region_mask)
        extracted_tac, uncertainty = self.tac_extraction_func(pet_voxels=pet_masked_region,
                                                              **tac_calc_kwargs)
        region_tac = TimeActivityCurve(times=self.scan_timing.center_in_mins,
                                       activity=extracted_tac,
                                       uncertainty=uncertainty)
        return region_tac


    def write_tacs(self,
                   out_tac_prefix: str,
                   out_tac_dir: str | pathlib.Path,
                   one_tsv_per_region: bool=True,
                   **tac_calc_kwargs):
        """
        Function to write Tissue Activity Curves for each region, given a segmentation,
        4D PET image, and label map. Computes the average of the PET image within each
        region. Writes a JSON for each region with region name, frame start time, and mean 
        value within region.

        Args:
            out_tac_prefix (str): Prefix for the output files, usually the BIDS subject and
                session ID.
            out_tac_dir (str | pathlib.Path): Output path where files are saved.
            one_tsv_per_region (bool): If True, write one TSV TAC file for each region in the
                image. If False, write one TSV file with all TACs in the image.
            **tac_calc_kwargs: Additional keywords passed onto tac_extraction_func.
        """
        tacs_data = pd.DataFrame()

        tacs_data['frame_time'] = self.scan_timing.center_in_mins
        tacs_data['duration'] = self.scan_timing.duration_in_mins

        unique_labels = unique_segmentation_labels(self.seg_arr)

        for label in unique_labels:
            tac = self.extract_tac(label, **tac_calc_kwargs)
            region_name = self.find_label_name(label=label)
            if one_tsv_per_region:
                tac.to_tsv(filename=f'{out_tac_dir}/{out_tac_prefix}_seg-{region_name}_tacs.tsv')
            else:
                tacs_data[region_name] = tac.activity
                tacs_data[f'{region_name}_unc'] = tac.uncertainty

        if not one_tsv_per_region:
            tacs_data.to_csv(f'{out_tac_dir}/{out_tac_prefix}_tacs.tsv', sep='\t', index=False)

    def __call__(self,
                 out_tac_prefix: str,
                 out_tac_dir: str | pathlib.Path,
                 one_tsv_per_region: bool=True,
                 **tac_calc_kwargs):
        """Runs TAC computation and writing by running `self.write_tacs`.
        
        Args:
            out_tac_prefix (str): Prefix for the output files, usually the BIDS subject and
                session ID.
            out_tac_dir (str | pathlib.Path): Output path where files are saved.
            one_tsv_per_region (bool): If True, write one TSV TAC file for each region in the
                image. If False, write one TSV file with all TACs in the image.
            **tac_calc_kwargs: Additional keywords passed onto tac_extraction_func."""
        self.write_tacs(out_tac_prefix=out_tac_prefix,
                        out_tac_dir=out_tac_dir,
                        one_tsv_per_region=one_tsv_per_region,
                        **tac_calc_kwargs)
