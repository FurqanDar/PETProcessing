""""
4D pet tools.
"""
import os
import json
import ants
import nibabel
from nibabel import processing
import numpy as np
from . import image_io
from . import math_lib

class ImageOps4D():
    """
    A class, supplies tools to modify values of 4D images.
    
    Attributes:
        images (list[ImageIO]): Enforced order: PET, MRI, segmentation.
        image_meta (dict): Image metadata pulled from BIDS-compliant json file.
        half_life (float): Half life of radioisotope to be used for computations.
                           Default value 0.
    
    See Also:
        :class: `ImageIO`
    """
    def __init__(self,
        sub_id: str,
        out_path: str,
        image_paths: dict=None,
        half_life: float=0,
        color_table_path: str=None,
        verbose: bool=True
    ):
        """
        Constructor for ImageOps4d

        Args:
            sub_id (str):
            image_paths (dict): Dictionary containing paths to files used in relevant
                preprocessing steps. This variable has protected 
            out_path (str): Path to output directory to which preprocessing files are written.
            half_life (float): Half life of tracer radioisotope used for the study in seconds.
                Default value 0.
            color_table_path (str): Path to location of color table, matching region names to
                region values in a segmentation file. See:
                https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles.
            verbose (bool): Control output of debugging information. Default value True.
        """
        self.sub_id = sub_id
        if image_paths is None:
            image_paths = {}
        self.image_paths = image_paths
        # NB protected keywords: {'pet': pet_path,'mri': mri_path,'seg': seg_path}
        self.half_life = half_life
        self.out_path = out_path
        self.color_table_path = color_table_path
        self.verbose = verbose


    def weighted_series_sum(self) -> np.ndarray:
        """
        Sum a 4D image series weighted based on time and re-corrected for decay correction.

        Args:
            pet_series (np.ndarray): Input pet image to be summed.
            image_meta (dict): Metadata json file following BIDS standard, from which
                               we collect frame timing and decay correction info.
            half_life (float): Half life of the PET radioisotope in seconds.

        Returns:
            summed_image (np.ndarray): Summed image 
        """
        pet_meta = image_io.load_meta(self.image_paths['pet'])
        pet_image = nibabel.load(self.image_paths['pet'])
        pet_series = pet_image.get_fdata()
        image_frame_start = pet_meta['FrameTimesStart']
        image_frame_duration = pet_meta['FrameDuration']
        if 'DecayCorrectionFactor' in pet_meta.keys():
            image_decay_correction = pet_meta['DecayCorrectionFactor']
        elif 'DecayFactor' in pet_meta.keys():
            image_decay_correction = pet_meta['DecayFactor']
        if 'TracerRadionuclide' in pet_meta.keys():
            tracer_isotope = pet_meta['TracerRadionuclide']
            if self.verbose:
                print(f"(ImageOps4D): Radio isotope is {tracer_isotope}",
                    "with half life {self.half_life} s")
        image_weighted_sum = math_lib.weighted_sum_computation(
            image_frame_duration,
            self.half_life,
            pet_series,
            image_frame_start,
            image_decay_correction
        )
        pet_sum_image = nibabel.nifti1.Nifti1Image(
            dataobj=image_weighted_sum,
            affine=pet_image.affine,
            header=pet_image.header
        )
        sum_image_path = os.path.join(self.out_path,'sum_image')
        os.makedirs(sum_image_path,exist_ok=True)
        self.image_paths['pet_sum_image'] = os.path.join(
            sum_image_path,
            f'{self.sub_id}-sum-image.nii.gz')
        nibabel.save(pet_sum_image,self.image_paths['pet_sum_image'])

        return image_weighted_sum


    def motion_correction(self) -> tuple[np.ndarray, list[str], list[float]]:
        """
        Motion correct PET image series.

        Returns:
            pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy array.
            pet_moco_pars (list[str]): List of ANTS registration files applied to each frame.
            pet_moco_fd (list[float]): List of framewise displacement measure corresponding 
                to each frame transform.
        """
        pet_nibabel = nibabel.load(self.image_paths['pet'])
        pet_sum_image = nibabel.load(self.image_paths['pet_sum_image'])
        pet_sum_image_ants = ants.from_nibabel(pet_sum_image)
        pet_ants = ants.from_nibabel(pet_nibabel)
        pet_moco_ants_dict = ants.motion_correction(pet_ants,
            pet_sum_image_ants,
            type_of_transform='Rigid')
        pet_moco_ants = pet_moco_ants_dict['motion_corrected']
        pet_moco_pars = pet_moco_ants_dict['motion_parameters']
        pet_moco_fd = pet_moco_ants_dict['FD']
        pet_moco_np = pet_moco_ants.numpy()
        pet_moco_nibabel = ants.to_nibabel(pet_moco_ants)
        moco_path = os.path.join(self.out_path,'motion-correction')
        os.makedirs(moco_path,exist_ok=True)
        self.image_paths['pet_moco'] = os.path.join(moco_path,f'{self.sub_id}-moco.nii.gz')
        nibabel.save(pet_moco_nibabel,self.image_paths['pet_moco'])
        return pet_moco_np, pet_moco_pars, pet_moco_fd


    def register_pet(self):
        """
        Register PET image series to anatomical data. Computes transform based on weighted average,
        which is then applied to the 4D PET image series.
        """
        pet_sum_image = ants.image_read(self.image_paths['pet_sum_image'])
        mri_image = ants.image_read(self.image_paths['mri'])
        pet_moco = ants.image_read(self.image_paths['pet_moco'])
        print('loaded images')
        xfm_output = ants.registration(
            moving=pet_sum_image,
            fixed=mri_image,
            type_of_transform='DenseRigid',
            write_composite_transform=True)
        xfm_apply = ants.apply_transforms(
            moving=pet_moco,
            fixed=mri_image,
            transformlist=xfm_output['fwdtransforms'],
            imagetype=3,
            verbose=True)
        print('applied registration')
        reg_path = os.path.join(self.out_path,'registration')
        os.makedirs(reg_path,exist_ok=True)
        self.image_paths['pet_moco_reg'] = os.path.join(reg_path,f'{self.sub_id}-moco-reg.nii.gz')
        ants.image_write(xfm_apply,self.image_paths['pet_moco_reg'])

    def mask_image_to_vals(self,
                         values: list[int],
                         resample_seg: bool=False) -> np.ndarray:
        """
        Masks an input image based on a value or list of values, and returns an array
        with original image values in the regions based on values specified for the mask.

        Args:
            values (list[int]): List of values corresponding to regions to be masked.
            resample_seg (bool): Determines whether or not to resample the segmentation.
                Set to True when the PET input (registered to MPR) and segmentation are 
                different resolutions.

        Returns:
            tac_out (np.ndarray): Mean of values within mask for each frame in 4D PET series.
        """
        pet_image = nibabel.load(self.image_paths['pet_moco_reg'])
        seg_image = nibabel.load(self.image_paths['seg'])
        pet_series = pet_image.get_fdata()
        num_frames = pet_series.shape[3]
        if resample_seg:
            image_first_frame = pet_series[:,:,:,0]
            seg_resampled = processing.resample_from_to(from_img=seg_image,
                                to_vox_map=(image_first_frame.shape,
                                pet_image.affine),
                                order=0)
            seg_res_path = os.path.join(self.out_path,'segmentation')
            os.makedirs(seg_res_path,exist_ok=True)
            self.image_paths['seg_resampled'] = os.path.join(
                seg_res_path,
                f'{self.sub_id}-segmentation-resampled.nii.gz')
            nibabel.save(seg_resampled,self.image_paths['seg_resampled'])
        seg_for_masking = nibabel.load(self.image_paths['seg_resampled']).get_fdata()
        for region in values:
            masked_voxels = seg_for_masking==region
            masked_image = pet_series[masked_voxels].reshape((-1,num_frames))
            tac_out = np.mean(masked_image,axis=0)
        return tac_out


    def write_tacs(self):
        """
        Function to write Tissue Activity Curves for each region, given a segmentation,
        4D PET image, and color table. Computes the average of the PET image within each
        region. Writes a JSON for each region with region name, frame start time, and mean 
        value within region.
        """
        pet_meta = image_io.load_meta(self.image_paths['pet'])
        with open(self.color_table_path,'r',encoding='utf-8') as color_table_file:
            color_table = json.load(color_table_file)
        regions_list = color_table['data']
        res = True
        for region_pair in regions_list:
            region_index, region_name = region_pair
            region_json = {'region_name': region_name}
            series_means = self.mask_image_to_vals([region_index],resample_seg=res).tolist()
            res=False
            region_json['frame_start_time'] = pet_meta['FrameTimesStart']
            region_json['activity'] = series_means
            tac_path = os.path.join(f'{self.out_path}','tacs')
            os.makedirs(tac_path,exist_ok=True)
            with open(os.path.join(tac_path,f'{self.sub_id}-{region_name}-tac.json'),
                      'w',encoding='ascii') as out_file:
                json.dump(obj=region_json,fp=out_file,indent=4)
