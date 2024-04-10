"""
The 'image_operations_4d' module provides several functions used to do preprocessing
on 4D PET imaging series. These functions typically take one or more paths to imaging
data in NIfTI format, and save modified data to a NIfTI file, and may return the
modified imaging array as output.

Class :class:`ImageOps4D` is also included in this module, and provides specific
implementations of the functions presented herein.

TODOs:
    * (weighted_series_sum) Refactor the DecayFactor key extraction into its own function
    * (weighted_series_sum) Refactor verbose reporting into the class as it is unrelated to
      computation
    * (write_tacs) Shift save into a np.savetxt with the region name as the header comment
    * (write_tacs) Shift to accepting color-key dictionaries rather than a file path.

"""
import os
import re
import ants
import nibabel
from nibabel import processing
import numpy as np
from . import image_io
from . import math_lib


def weighted_series_sum(input_image_4d_path: str,
                        out_image_path: str,
                        half_life: float,
                        verbose: bool) -> np.ndarray:
    r"""
    Sum a 4D image series weighted based on time and re-corrected for decay correction.

    First, a scaled image is produced by multiplying each frame by its length in seconds,
    and dividing by the decay correction applied:

    .. math::
    
        f_i'=f_i\times \frac{t_i}{d_i}

    Where :math:`f_i,t_i,d_i` are the i-th frame, frame duration, and decay correction factor of
    the PET series. This scaled image is summed over the time axis. Then, to get the output, we
    multiply by a factor called `total decay` and divide by the full length of the image:

    .. math::

        d_{S} = \frac{\lambda*t_{S}}{(1-\exp(-\lambda*t_{S}))(\exp(\lambda*t_{0}))}

    .. math::
    
        S(f) = \sum(f_i') * d_{S} / t_{S}

    where :math:`\lambda=\log(2)/T_{1/2}` is the decay constant of the radio isotope,
    :math:`t_0` is the start time of the first frame in the PET series, the subscript :math:`S`
    indicates the total quantity computed over all frames, and :math:`S(f)` is the final weighted
    sum image.

    
    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image on which the weighted sum is calculated. Assume a metadata
            file exists with the same path and file name, but with extension .json,
            and follows BIDS standard.
        out_image_path (str): Path to a .nii or .nii.gz file to which the weighted
            sum is written.
        half_life (float): Half life of the PET radioisotope in seconds.
        verbose (bool): Set to `True` to output processing information.

    Returns:
        summed_image (np.ndarray): 3D image array, in the same space as the input,
            with the weighted sum calculation applied.

    Raises:
        ValueError: If `half_life` is zero or negative.
    """
    if half_life <= 0:
        raise ValueError('(ImageOps4d): Radioisotope half life is zero or negative.')
    pet_meta = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    pet_image = nibabel.load(input_image_4d_path)
    pet_series = pet_image.get_fdata()
    image_frame_start = pet_meta['FrameTimesStart']
    image_frame_duration = pet_meta['FrameDuration']

    if 'DecayCorrectionFactor' in pet_meta.keys():
        image_decay_correction = pet_meta['DecayCorrectionFactor']
    elif 'DecayFactor' in pet_meta.keys():
        image_decay_correction = pet_meta['DecayFactor']
    else:
        raise ValueError("Neither 'DecayCoorectionFactor' nor 'DecayFactor' exist in meta-data file")

    if 'TracerRadionuclide' in pet_meta.keys():
        tracer_isotope = pet_meta['TracerRadionuclide']
        if verbose:
            print(f"(ImageOps4d): Radio isotope is {tracer_isotope} "
                f"with half life {half_life} s")

    image_weighted_sum = math_lib.weighted_sum_computation(image_frame_duration=image_frame_duration,
                                                           half_life=half_life,
                                                           pet_series=pet_series,
                                                           image_frame_start=image_frame_start,
                                                           image_decay_correction=image_decay_correction)

    pet_sum_image = nibabel.nifti1.Nifti1Image(dataobj=image_weighted_sum,
                                               affine=pet_image.affine,
                                               header=pet_image.header)
    nibabel.save(pet_sum_image, out_image_path)
    if verbose:
        print(f"(ImageOps4d): weighted sum image saved to {out_image_path}")
    return pet_sum_image


def motion_correction(
        input_image_4d_path: str,
        reference_image_path: str,
        out_image_path: str,
        verbose: bool,
        **kwargs) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Correct PET image series for inter-frame motion. Runs rigid motion correction module
    from Advanced Normalisation Tools (ANTs) with default inputs. 

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be motion corrected.
        reference_image_path (str): Path to a .nii or .nii.gz file containing a 3D reference
            image in the same space as the input PET image. Can be a weighted series sum,
            first or last frame, an average over a subset of frames, or another option depending
            on the needs of the data.
        out_image_path (str): Path to a .nii or .nii.gz file to which the motion corrected PET
            series is written.
        verbose (bool): Set to `True` to output processing information.
        kwargs (keyword arguments): Additional arguments passed to `ants.motion_correction`.

    Returns:
        pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy array.
        pet_moco_params (list[str]): List of ANTS registration files applied to each frame.
        pet_moco_fd (list[float]): List of framewise displacement measure corresponding 
        to each frame transform.
    """
    pet_nibabel = nibabel.load(input_image_4d_path)
    pet_ref_image = nibabel.load(reference_image_path)
    pet_ants = ants.from_nibabel(pet_nibabel)
    pet_sum_image_ants = ants.from_nibabel(pet_ref_image)

    pet_moco_ants_dict = ants.motion_correction(pet_ants,
                                                pet_sum_image_ants,
                                                type_of_transform='Rigid',
                                                **kwargs)
    if verbose:
        print('(ImageOps4D): motion correction finished.')

    pet_moco_ants = pet_moco_ants_dict['motion_corrected']
    pet_moco_params = pet_moco_ants_dict['motion_parameters']
    pet_moco_fd = pet_moco_ants_dict['FD']
    pet_moco_np = pet_moco_ants.numpy()
    pet_moco_nibabel = ants.to_nibabel(pet_moco_ants)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(
        input_image_4d_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)

    nibabel.save(pet_moco_nibabel, out_image_path)
    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")
    return pet_moco_np, pet_moco_params, pet_moco_fd


def register_pet(input_calc_image_path: str,
                 input_reg_image_path: str,
                 reference_image_path: str,
                 out_image_path: str,
                 verbose: bool):
    """
    Computes and runs rigid registration of 4D PET image series to 3D anatomical image, typically
    a T1 MRI. Runs rigid registration module from Advanced Normalisation Tools (ANTs) with  default
    inputs. Will upsample PET image to the resolution of anatomical imaging.

    Args:
        input_calc_image_path (str): Path to a .nii or .nii.gz file containing a 3D reference
            image in the same space as the input PET image, to be used to compute the rigid 
            registration to anatomical space. Can be a weighted series sum, first or last frame,
            an average over a subset of frames, or another option depending on the needs of the 
            data.
        input_reg_image_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be registered to anatomical space.
        reference_image_path (str): Path to a .nii or .nii.gz file containing a 3D
            anatomical image to which PET image is registered.
        out_image_path (str): Path to a .nii or .nii.gz file to which the registered PET series
            is written.
        verbose (bool): Set to `True` to output processing information.
    """
    pet_sum_image = ants.image_read(input_calc_image_path)
    mri_image = ants.image_read(reference_image_path)
    pet_moco = ants.image_read(input_reg_image_path)
    xfm_output = ants.registration(moving=pet_sum_image,
                                   fixed=mri_image,
                                   type_of_transform='DenseRigid',
                                   write_composite_transform=True
        )
    if verbose:
        print(
            f'Registration computed transforming image {input_calc_image_path} to'
            f'{reference_image_path} space'
            )

    xfm_apply = ants.apply_transforms(moving=pet_moco,
                                      fixed=mri_image,
                                      transformlist=xfm_output['fwdtransforms'],
                                      imagetype=3)
    if verbose:
        print(f'Registration applied to {input_reg_image_path}')

    ants.image_write(xfm_apply, out_image_path)
    if verbose:
        print(f'Transformed image saved to {out_image_path}')


def resample_segmentation(input_image_4d_path: str,
                          segmentation_image_path: str,
                          out_seg_path: str,
                          verbose: bool):
    """
    Resamples a segmentation image to the resolution of a 4D PET series image. Takes the affine 
    information stored in the PET image, and the shape of the image frame data, as well as the 
    segmentation image, and applies NiBabel's `resample_from_to` to resample the segmentation to
    the resolution of the PET image. This is used for extracting TACs from PET imaging where the 
    PET and ROI data are registered to the same space, but have different resolutions.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image, registered to anatomical space, to which the segmentation file is resampled.
        segmentation_image_path (str): Path to a .nii or .nii.gz file containing a 3D segmentation
            image, where integer indices label specific regions.
        out_seg_path (str): Path to a .nii or .nii.gz file to which the resampled segmentation
            image is written.
        verbose (bool): Set to `True` to output processing information.
    """
    pet_image = nibabel.load(input_image_4d_path)
    seg_image = nibabel.load(segmentation_image_path)
    pet_series = pet_image.get_fdata()
    image_first_frame = pet_series[:, :, :, 0]
    seg_resampled = processing.resample_from_to(from_img=seg_image,
                                                to_vox_map=(image_first_frame.shape, pet_image.affine),
                                                order=0)
    nibabel.save(seg_resampled, out_seg_path)
    if verbose:
        print(f'Resampled segmentation saved to {out_seg_path}')


def extract_tac_from_4dnifty_using_mask(input_image_4d_path: str,
                                        segmentation_image_path: str,
                                        values: list[int],
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
        values (list[int]): List of values in the segmentation image, which correspond to regions
            for which the TAC is to be computed on. Only one region at a time is implemented.
        verbose (bool): Set to `True` to output processing information.

    Returns:
        tac_out (np.ndarray): Mean of PET image within regions for each frame in 4D PET series.

    Raises:
        NotImplementedError: If `values` has more than two regions, as this is future functionality
    """
    if len(values) > 1:
        raise NotImplementedError('extract_tac_from_4dnifty_using_mask can only average over one region at the moment.'
            ' Use a list with only one value.')

    pet_image_4d = nibabel.load(input_image_4d_path).get_fdata()
    num_frames = pet_image_4d.shape[3]
    seg_image = nibabel.load(segmentation_image_path).get_fdata()

    tac_out = np.zeros(num_frames, float)
    for region in values:
        if verbose:
            print(f'Running TAC for region index {region}')
        masked_voxels = seg_image == region
        masked_image = pet_image_4d[masked_voxels].reshape((-1, num_frames))
        tac_out = np.mean(masked_image, axis=0)
    return tac_out


def write_tacs(input_image_4d_path: str,
               color_table_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               verbose: bool,
               time_frame_keyword: str = 'FrameReferenceTime'):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and color table. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    color_table = image_io.ImageIO.read_color_table_json(ctab_file=color_table_path)
    regions_list = color_table['data']

    tac_extraction_func = extract_tac_from_4dnifty_using_mask

    for region_index, region_name in regions_list.items():
        region_json = {'region_name': region_name, 'tac': {'time': None, 'activity': None}}
        region_json['tac']['time'] = pet_meta[time_frame_keyword]
        extracted_tac = tac_extraction_func(input_image_4d_path=input_image_4d_path,
                                            segmentation_image_path=segmentation_image_path,
                                            values=[int(region_index)],
                                            verbose=verbose)
        region_json['tac']['activity'] = extracted_tac.tolist()
        out_tac_path = os.path.join(out_tac_dir, f'tac-{region_name}.json')
        image_io.write_dict_to_json(meta_data_dict=region_json, out_path=out_tac_path)


class ImageOps4D():
    """
    :class:`ImageOps4D` to provide basic implementations of the preprocessing functions in module
    `image_operations_4d`.

    Preprocessing can be run on individual subjects by specifying information such as the subject
    id, output path, paths to PET, anatomical, and segmentation images, etc. Then individual
    methods can be run in succession.

    Key methods include:
        - :meth:`run_weighted_series_sum`: Runs :meth:`weighted_series_sum` on input data.
        - :meth:`run_motion_correction`: Runs :meth:`motion_correction` on input data, with the
          output of :func:`weighted_series_sum` as reference.
        - :meth:`run_register_pet`: Runs :meth:`register_pet` on motion corrected PET with the
          output of :func:`weighted_series_sum` used to compute registration.
        - :meth:`run_mask_image_to_vals`: Runs :meth:`extract_tac_from_4dnifty_using_mask`, to be
          used with :meth:`run_write_tacs`.
        - :meth:`run_write_tacs`: Runs :meth:`write_tacs` on preprocessed PET data to produce
          regional TACs.
    
    Attributes:
        sub_id (str): The subject ID, used for naming output files.
        out_path (str): Path to an output directory, to which processed files are saved.
        image_paths (dict): A dictionary with designated keys for different types of images.
            Designated keys include 'pet' for input PET data, 'mri' for anatomical data, 'seg' for
            segmentation in anatomical space, 'pet_sum_image' for the output to
            :meth:`weighted_series_sum`, 'pet_moco' for motion corrected PET image, pet_moco_reg
            for motion corrected and registered PET image, and 'seg_resampled' for a segmentation
            resampled onto PET resolution.
        half_life (float): Half-life of the radioisotope used in PET study in seconds.
        color_table_path (str): Path to a color table .json file used to match region names to
            region indices.
        verbose (bool): Set to `True` to output processing information.
    
    See Also:
        :class:`ImageIO`
    
    """

    def __init__(
        self,
        sub_id: str,
        out_path: str,
        image_paths: dict = None,
        half_life: float = None,
        color_table_path: str = None,
        verbose: bool = True
        ):
        """
        Constructor for ImageOps4d, initializing class attributes.

        Args:
            sub_id (str): The subject ID, used for naming output files.
            out_path (str): Path to an output directory, to which processed files are saved.
            image_paths (dict): A dictionary with designated keys for different types of images.
                Designated keys include 'pet' for input PET data, 'mri' for anatomical data, 'seg'
                for segmentation in anatomical space, 'pet_sum_image' for the output to
                :meth:`weighted_series_sum`, 'pet_moco' for motion corrected PET image, pet_moco_reg
                for motion corrected and registered PET image, and 'seg_resampled' for a 
                segmentation resampled onto PET resolution.
            half_life (float): Half-life of the radioisotope used in PET study in seconds.
            color_table_path (str): Path to a color table .json file used to match region names to
                region indices.
            verbose (bool): Set to `True` to output processing information.
        """
        self.sub_id = sub_id
        if image_paths is None:
            image_paths = {}
        self.image_paths = image_paths
        self.half_life = half_life
        self.out_path = out_path
        self.color_table_path = color_table_path
        self.verbose = verbose

    def run_weighted_series_sum(self) -> np.ndarray:
        """
        Computes weighted sum image by running :meth:`weighted_series_sum` on input data. Write
        output as 'pet_sum_image'.
        """
        sum_image_path = os.path.join(self.out_path, 'sum_image')
        os.makedirs(sum_image_path, exist_ok=True)
        self.image_paths['pet_sum_image'] = os.path.join(sum_image_path,f'{self.sub_id}-sum.nii.gz')
        weighted_series_sum(
            input_image_4d_path=self.image_paths['pet'],
            out_image_path=self.image_paths['pet_sum_image'],
            half_life=self.half_life,
            verbose=self.verbose
            )

    def run_motion_correction(self) -> tuple[np.ndarray, list[str], list[float]]:
        """
        Motion correct PET image series by running :meth:`motion_correction` on input data, with the 
        output of weighted_series_sum as reference. Write output as 'pet_moco'.
        """
        moco_path = os.path.join(self.out_path, 'motion-correction')
        os.makedirs(moco_path, exist_ok=True)
        self.image_paths['pet_moco'] = os.path.join(moco_path, f'{self.sub_id}-moco.nii.gz')
        motion_correction(
            input_image_4d_path=self.image_paths['pet'],
            reference_image_path=self.image_paths['pet_sum_image'],
            out_image_path=self.image_paths['pet_moco'],
            verbose=self.verbose
            )

    def run_register_pet(self):
        """
        Registers PET to anatomical by running :meth:`register_pet` on motion corrected PET with the 
        output of weighted_series_sum used to compute registration. Write output as 'pet_moco_reg'.
        """
        reg_path = os.path.join(self.out_path, 'registration')
        os.makedirs(reg_path, exist_ok=True)
        self.image_paths['pet_moco_reg'] = os.path.join(reg_path, f'{self.sub_id}-reg.nii.gz')
        register_pet(
            input_calc_image_path=self.image_paths['pet_sum_image'],
            input_reg_image_path=self.image_paths['pet_moco'],
            reference_image_path=self.image_paths['mri'],
            out_image_path=self.image_paths['pet_moco_reg'],
            verbose=self.verbose
            )

    def run_mask_image_to_vals(self, values: list[int], resample_seg: bool = False) -> np.ndarray:
        """
        Creates a time-activity curve (TAC) by computing the average value within a region, for 
        each frame in a 4D PET image series. Takes as input a PET image, which has been registered
        to anatomical space, a segmentation image, with the same sampling as the PET, and a list of
        values corresponding to regions in the segmentation image that are used to compute the 
        average regional values. Currently, only the mean over a single region value is
        implemented.

        Args:
            values (list[int]): List of values corresponding to regions to be masked.
            resample_seg (bool): Determines whether or not to resample the segmentation.
                Set to True when the PET input (registered to MPR) and segmentation are 
                different resolutions.

        Returns:
            tac_out (np.ndarray): Mean of values within mask for each frame in 4D PET series.
        """
        if resample_seg:
            seg_res_path = os.path.join(self.out_path, 'segmentation')
            os.makedirs(seg_res_path, exist_ok=True)
            self.image_paths['seg_resampled'] = os.path.join(
                seg_res_path,
                f'{self.sub_id}-segmentation-resampled.nii.gz'
                )
            resample_segmentation(
                input_image_4d_path=self.image_paths['pet_moco_reg'],
                segmentation_image_path=self.image_paths['seg'],
                out_seg_path=self.image_paths['seg_resampled'],
                verbose=self.verbose
                )
        tac_out = extract_tac_from_4dnifty_using_mask(
            input_image_4d_path=self.image_paths['pet_moco_reg'],
            segmentation_image_path=self.image_paths['seg_resampled'],
            values=values,
            verbose=self.verbose
            )
        return tac_out

    def run_write_tacs(self):
        """
        Function to write Tissue Activity Curves for each region by running :meth:`write_tacs` on
        preprocessed PET data. Requires registration to anatomical and segmentation resampled to
        PET resolution.
        """
        tac_path = os.path.join(f'{self.out_path}', 'tacs')
        os.makedirs(tac_path, exist_ok=True)
        write_tacs(
            input_image_4d_path=self.image_paths['pet_moco_reg'],
            color_table_path=self.color_table_path,
            segmentation_image_path=self.image_paths['seg_resampled'],
            out_tac_dir=tac_path,
            verbose=self.verbose
            )
