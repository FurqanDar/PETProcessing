"""
Introduces class :class:`PreProc` which handles preprocessing of PET and other
neuroimaging data for a PET study. Acts as a wrapper for other tools supplied
in ``PETPAL``.

TODO:
    * Check if input files exist, throw error if no.
    * Verify images have the same shape and orientation.

"""
import os

import petpal.utils.useful_functions
from ..visualizations import qc_plots
from . import register, image_operations_4d, motion_corr, segmentation_tools

weighted_series_sum = petpal.utils.useful_functions.weighted_series_sum
write_tacs = image_operations_4d.write_tacs
roi_tac = image_operations_4d.roi_tac
resample_segmentation = segmentation_tools.resample_segmentation
suvr = image_operations_4d.suvr
gauss_blur = image_operations_4d.gauss_blur
register_pet = register.register_pet
warp_pet_atlas = register.warp_pet_atlas
apply_xfm_ants = register.apply_xfm_ants
apply_xfm_fsl = register.apply_xfm_fsl
thresh_crop = image_operations_4d.SimpleAutoImageCropper


_PREPROC_PROPS_ = {'FilePathCropInput': None,
                   'FilePathWSSInput': None,
                   'FilePathMocoInp': None,
                   'FilePathRegInp': None,
                   'FilePathAnat': None,
                   'FilePathTACInput': None,
                   'FilePathSeg': None,
                   'FilePathLabelMap': None,
                   'FilePathWarpInput': None,
                   'FilePathAtlas': None,
                   'FilePathSUVRInput': None,
                   'FilePathBlurInput': None,
                   'FilePathFSLPostmat': '',
                   'FilePathFSLPremat': '',
                   'FilePathWarpRef': None,
                   'FilePathWarp': None,
                   'FilePathAntsXfms': None,
                   'FilePathBSseg': None,
                   'HalfLife': None,
                   'StartTimeWSS': 0,
                   'EndTimeWSS': -1,
                   'CropXdim': None,
                   'CropYdim': None,
                   'MotionTarget': None,
                   'MocoTransformType': 'DenseRigid',
                   'MocoPars': None,
                   'RegPars': None,
                   'WarpPars': None,
                   'RefRegion': None,
                   'BlurSize': None,
                   'RegionExtract': None,
                   'TimeFrameKeyword': None,
                   'CropThreshold': None,
                   'Verbose': False}
_REQUIRED_KEYS_ = {
    'weighted_series_sum': ['FilePathWSSInput','HalfLife','Verbose'],
    'motion_corr': ['FilePathMocoInp','MotionTarget','Verbose'],
    'motion_corr_frames_above_mean': ['FilePathMocoInp','MotionTarget','Verbose'],
    'register_pet': ['MotionTarget','FilePathRegInp','FilePathAnat','Verbose'],
    'resample_segmentation': ['FilePathTACInput','FilePathSeg','Verbose'],
    'roi_tac': ['FilePathTACInput','FilePathSeg','RegionExtract','Verbose'],
    'write_tacs': ['FilePathTACInput','FilePathLabelMap','FilePathSeg','Verbose','TimeFrameKeyword'],
    'warp_pet_atlas': ['FilePathWarpInput','FilePathAnat','FilePathAtlas','Verbose'],
    'suvr': ['FilePathSUVRInput','FilePathSeg','RefRegion','Verbose'],
    'gauss_blur': ['FilePathBlurInput','BlurSize','Verbose'],
    'apply_xfm_ants': ['FilePathWarpInput','FilePathWarpRef','FilePathAntsXfms','Verbose'],
    'apply_xfm_fsl': ['FilePathWarpInput','FilePathWarpRef','FilePathWarp','FilePathFSLPremat','FilePathFSLPostmat','Verbose'],
    'thresh_crop': ['FilePathCropInput', 'CropThreshold', 'Verbose'],
    'vat_wm_ref_region': ['FilePathBSseg','FilePathSeg'],
    'crop_image': ['FilePathCropInput','CropXdim','CropYdim']
}


class PreProc:
    """
    :class:`ImageOps4D` to provide basic implementations of the preprocessing functions in module
    ``image_operations_4d``. Uses a properties dictionary ``preproc_props`` to
    determine the inputs and outputs of preprocessing methods.

    Key methods include:
        - :meth:`update_props`: Update properties dictionary ``preproc_props``
          with new properties.
        - :meth:`run_preproc`: Given a method in ``image_operations_4d``, run the
          provided method with inputs and outputs determined by properties
          dictionary ``preproc_props``.

    Attributes:
        -`output_directory`: Directory in which files are written to.
        -`output_filename_prefix`: Prefix appended to beginning of written
         files.
        -`preproc_props`: Properties dictionary used to set parameters for PET
         preprocessing. See :meth:`_init_preproc_props` for further details.

    Example:

    .. code-block:: python

        output_directory = '/path/to/processing'
        output_filename_prefix = 'sub-01'
        sub_01 = pet_cli.preproc.PreProc(output_directory,output_filename_prefix)
        params = {
            'FilePathWSSInput': '/path/to/pet.nii.gz',
            'FilePathAnat': '/path/to/mri.nii.gz',
            'HalfLife': 1220.04,  # C11 half-life in seconds
            'FilePathRegInp': '/path/to/image/to/be/registered.nii.gz',
            'FilePathMocoInp': '/path/to/image/to/be/motion/corrected.nii.gz',
            'MotionTarget': '/path/to/pet/reference/target.nii.gz',
            'FilePathTACInput': '/path/to/registered/pet.nii.gz',
            'FilePathLabelMap': '/path/to/label/map.tsv',
            'FilePathSeg': '/path/to/segmentation.nii.gz',
            'TimeFrameKeyword': 'FrameTimesStart',  # using start time or midpoint reference time
            'Verbose': True,
        }
        sub_01.update_props(params)
        sub_01.run_preproc('weighted_series_sum')
        sub_01.run_preproc('motion_corr')
        sub_01.run_preproc('register_pet')
        sub_01.run_preproc('write_tacs')


    See Also:
        :class:`ImageIO`

    """
    def __init__(self,
                 output_directory: str,
                 output_filename_prefix: str) -> None:
        """
        Initialize PreProc class.

        Args:
            output_directory (str): Directory to which output images are saved.
            output_filename_prefix (str): Prefix appended to beginning of saved
                files. Typically subject ID.
        """
        self.output_directory = os.path.abspath(output_directory)
        os.makedirs(self.output_directory,exist_ok=True)
        self.output_filename_prefix = output_filename_prefix
        self.preproc_props = self._init_preproc_props()


    @staticmethod
    def _init_preproc_props() -> dict:
        """
        Initializes preproc properties dictionary.

        The available fields in the preproc properties dictionary are described
        as follows:
        * FilePathWSSInput (str): Path to file on which to compute weighted series sum.
        * FilePathMocoInp (str): Path to PET file to be motion corrected.
        * FilePathRegInp (str): Path to PET file to be registered to anatomical data.
        * FilePathAnat (str): Path to anatomical image to which ``FilePathRegInp`` is registered.
        * FilePathTACInput (str): Path to PET file with which TACs are computed.
        * FilePathSeg (str): Path to a segmentation image in anatomical space.
        * FilePathLabelMap (str): Path to a label map file, indexing segmentation values to ROIs.
        * FilePathAtlas (str): Path to atlas image, e.g. MNI152 T1 atlas.
        * FilePathSUVRInput (str): Path to summed or parametric image on which to normalize to SUVR.
        * FilePathBlurInput (str): Path to image to blur with a gaussian kernal.
        * FilePathFSLPremat (str): Path to initial affine transform matrix in FSL format, used for FSL type warping.
        * FilePathFSLPostmat (str): Path to post-warp affine transform matrix in FSL format, used for FSL type warping.
        * FilePathWarpRef (str): Path to reference used to compute warp to atlas space. Typically anatomical scan.
        * FilePathAntsXfm (str): Path to list of Ants transforms used to apply ANTs type warping.
        * HalfLife (float): Half life of radioisotope. Used for a number of tools.
        * MotionTarget (str | tuple): Target for transformation methods. See :meth:`determine_motion_target`.
        * MocoPars (keyword arguments): Keyword arguments fed into the method call :meth:`ants.motion_correction`.
        * RegPars (keyword arguments): Keyword arguments fed into the method call :meth:`ants.registration` while performing PET to anat registration.
        * WarpPars (keyword arguments): Keyword arguments fed into the method call :meth:`ants.registration` while performing PET to atlas registration. 
        * RefRegion (int): Reference region used to normalize SUVR.
        * BlurSize (float): Size of gaussian kernal used to blur image.
        * RegionExtract (int): Region index in the segmentation image to extract TAC from, if running TAC on a single ROI.
        * TimeFrameKeyword (str): Keyword in metadata file corresponding to frame timing array to be used in analysis.
        * Verbose (bool): Set to ``True`` to output processing information.

        """
        return _PREPROC_PROPS_


    def update_props(self,new_preproc_props: dict) -> dict:
        """
        Update the processing properties with items from a new dictionary.

        Args:
            new_preproc_props (dict): Dictionary with updated properties 
                values. Can have any or all fields filled from the available
                list of fields.

        Returns:
            updated_props (dict): The updated ``preproc_props`` dictionary.
        """
        preproc_props = self.preproc_props
        valid_keys = [*preproc_props]
        updated_props = preproc_props.copy()
        keys_to_update = [*new_preproc_props]

        for key in keys_to_update:
            if key not in valid_keys:
                raise ValueError("Invalid preproc property! Expected one of:\n"
                                 f"{valid_keys}.\n Got {key}.")

            updated_props[key] = new_preproc_props[key]

        self.preproc_props = updated_props
        return updated_props

    def _check_input_paths_exist(self,
                                 method_name: str) -> None:
        """
        Verify that all filepaths in ``preproc_props`` exist for a given method.

        Args: method_name (str): Name of method to be checked for necessary input files.

        Raises: FileNotFoundError if any necessary files don't exist.

        """
        filepath_keys = [key for key in _REQUIRED_KEYS_[method_name] if key.startswith('FilePath')]
        errored_paths = []
        for key in filepath_keys:
            if self.preproc_props[key] is not None:
                if not os.path.exists(self.preproc_props[key]):
                    errored_paths.append(self.preproc_props[key])

        if len(errored_paths) > 0:
            raise FileNotFoundError(f"Input file(s) not found for the following path(s):\n{errored_paths}")

    def _check_method_props_exist(self,
                                  method_name: str) -> None:
        """
        Check if all necessary properties exist in the ``props`` dictionary to
        run the given method.

        Args:
            method_name (str): Name of method to be checked. Must be name of 
                a method in this module.
        """
        preproc_props = self.preproc_props
        existing_keys = [*preproc_props]
        accepted_keys = [*_REQUIRED_KEYS_]

        try:
            required_keys = _REQUIRED_KEYS_[method_name]
        except KeyError as e:
            raise KeyError(f"Invalid method_name! Must be one of: {accepted_keys} . Got '{method_name}'") from e

        for key in required_keys:
            if preproc_props[key] is None:
                raise ValueError(f"Preprocessing method {method_name} requires property"
                                 f" {key}, however {key} was not found in "
                                 "processing properties. Existing properties "
                                 f"are: {existing_keys}, while needed keys to "
                                 f"run {method_name} are: {required_keys}.")


    def generate_outfile_path(self,
                              method_short: str,
                              extension: str='nii.gz',
                              modality: str = None, ):
        r"""
        Generate the path to an output file, from the output directory,
        filename prefix, abbreviation of the method name, and filename
        extension.

        Args:
            method_short (str): Abbreviation of the method to generate outfile.
            extension (str): File type extension to return. Defaults to
                'nii.gz'.
            modality (str, optional): Modality of the image. Should be one of 'pet', 't1w', 'mpr', 'flair', 't2w'

        Returns:
            If modality is None, we return 'output_dir/fileprefix_{method_short}.{extension}'. Else, we return
            output_dir/fileprefix_desc-{method_short}_{modality}.{extension}
        """

        if modality is None:
            output_file_name = f'{self.output_filename_prefix}_{method_short}.{extension}'
        else:
            assert modality.lower() in ['pet', 't1w', 'mpr', 'flair', 't2w']
            output_file_name = f'{self.output_filename_prefix}_desc-{method_short}_{modality}.{extension}'
        return os.path.join(self.output_directory, output_file_name)


    def run_preproc(self,
                    method_name: str,
                    modality: str = None):
        """
        Run a specific preprocessing step.

        Args:
            method_name (str): Name of method to be run. Must be name of a method in this module.
            modality (str, optional): Modality of the image. Defaults to None. Should be one of 'pet', 't1w', 'mpr', 'flair', 't2w'
        """
        preproc_props = self.preproc_props
        self._check_method_props_exist(method_name=method_name)
        self._check_input_paths_exist(method_name=method_name)

        if method_name=='weighted_series_sum':
            outfile = self.generate_outfile_path(method_short='wss', modality=modality)
            weighted_series_sum(input_image_4d_path=preproc_props['FilePathWSSInput'],
                                out_image_path=outfile,
                                half_life=preproc_props['HalfLife'],
                                start_time=preproc_props['StartTimeWSS'],
                                end_time=preproc_props['EndTimeWSS'],
                                verbose=preproc_props['Verbose'])
        
        elif method_name == 'motion_corr_frames_above_mean':
            outfile = self.generate_outfile_path(method_short='moco', modality=modality)
            motion_corr.motion_corr_frames_above_mean_value(input_image_4d_path=preproc_props['FilePathMocoInp'],
                                                            out_image_path=outfile,
                                                            motion_target_option=preproc_props['MotionTarget'],
                                                            verbose=preproc_props['Verbose'],
                                                            type_of_transform=preproc_props['MocoTransformType'],
                                                            half_life=preproc_props['HalfLife'],
                                                            kwargs=preproc_props['MocoPars'])

        elif method_name=='motion_corr':
            outfile = self.generate_outfile_path(method_short='moco', modality=modality)
            moco_outputs = motion_corr.motion_corr(input_image_4d_path=preproc_props['FilePathMocoInp'],
                                                   motion_target_option=preproc_props['MotionTarget'],
                                                   type_of_transform=preproc_props['MocoTransformType'],
                                                   out_image_path=outfile,
                                                   verbose=preproc_props['Verbose'],
                                                   half_life=preproc_props['HalfLife'],
                                                   kwargs=preproc_props['MocoPars'])
            motion = moco_outputs[2]
            output_plot = os.path.join(self.output_directory,
                                       f'{self.output_filename_prefix}_motion.png')
            qc_plots.motion_plot(framewise_displacement=motion,
                                 output_plot=output_plot)
            return moco_outputs

        elif method_name=='register_pet':
            outfile = self.generate_outfile_path(method_short='reg', modality=modality)
            register_pet(motion_target_option=preproc_props['MotionTarget'],
                         input_reg_image_path=preproc_props['FilePathRegInp'],
                         reference_image_path=preproc_props['FilePathAnat'],
                         out_image_path=outfile,
                         verbose=preproc_props['Verbose'],
                         half_life=preproc_props['HalfLife'],
                         kwargs=preproc_props['RegPars'])

        elif method_name=='resample_segmentation':
            outfile = self.generate_outfile_path(method_short='seg-res', modality=modality)
            resample_segmentation(input_image_4d_path=preproc_props['FilePathTACInput'],
                                  segmentation_image_path=preproc_props['FilePathSeg'],
                                  out_seg_path=outfile,
                                  verbose=preproc_props['Verbose'])
            self.update_props({'FilePathSeg': outfile})

        elif method_name=='roi_tac':
            outfile = self.generate_outfile_path(method_short='tac', extension='.tsv', modality=modality)
            return roi_tac(input_image_4d_path=preproc_props['FilePathTACInput'],
                           roi_image_path=preproc_props['FilePathSeg'],
                           out_tac_path=outfile,
                           region=preproc_props['RegionExtract'],
                           verbose=preproc_props['Verbose'])

        elif method_name=='write_tacs':
            outdir = os.path.join(self.output_directory,'tacs')
            os.makedirs(outdir,exist_ok=True)
            write_tacs(input_image_path=preproc_props['FilePathTACInput'],
                       label_map_path=preproc_props['FilePathLabelMap'],
                       segmentation_image_path=preproc_props['FilePathSeg'],
                       out_tac_dir=outdir,
                       verbose=preproc_props['Verbose'],
                       time_frame_keyword=preproc_props['TimeFrameKeyword'])

        elif method_name=='warp_pet_atlas':
            outfile = self.generate_outfile_path(method_short='space-atlas', modality=modality)
            warp_pet_atlas(input_image_path=preproc_props['FilePathWarpInput'],
                           anat_image_path=preproc_props['FilePathAnat'],
                           atlas_image_path=preproc_props['FilePathAtlas'],
                           out_image_path=outfile,
                           verbose=preproc_props['Verbose'],
                           kwargs=preproc_props['WarpPars'])

        elif method_name=='apply_xfm_ants':
            outfile = self.generate_outfile_path(method_short='space-atlas', modality=modality)
            apply_xfm_ants(input_image_path=preproc_props['FilePathWarpInput'],
                           ref_image_path=preproc_props['FilePathWarpRef'],
                           out_image_path=outfile,
                           xfm_paths=preproc_props['FilePathAntsXfms'])

        elif method_name=='apply_xfm_fsl':
            outfile = self.generate_outfile_path(method_short='space-atlas', modality=modality)
            apply_xfm_fsl(input_image_path=preproc_props['FilePathWarpInput'],
                          ref_image_path=preproc_props['FilePathWarpRef'],
                          out_image_path=outfile,
                          warp_path=preproc_props['FilePathWarp'],
                          premat_path=preproc_props['FilePathFSLPremat'],
                          postmat_path=preproc_props['FilePathFSLPostmat'])

        elif method_name=='suvr':
            outfile = self.generate_outfile_path(method_short='suvr', modality=modality)
            suvr(input_image_path=preproc_props['FilePathSUVRInput'],
                 segmentation_image_path=preproc_props['FilePathSeg'],
                 ref_region=preproc_props['RefRegion'],
                 out_image_path=outfile,
                 verbose=preproc_props['Verbose'])

        elif method_name=='gauss_blur':
            outfile = self.generate_outfile_path(method_short=f"blur_{preproc_props['BlurSize']}mm", modality=modality)
            gauss_blur(input_image_path=preproc_props['FilePathBlurInput'],
                       blur_size_mm=preproc_props['BlurSize'],
                       out_image_path=outfile,
                       verbose=preproc_props['Verbose'])

        elif method_name=='vat_wm_ref_region':
            out_ref_region = self.generate_outfile_path(method_short='wm-ref', modality=modality)
            segmentation_tools.vat_wm_ref_region(
                input_segmentation_path=f"{preproc_props['FilePathSeg']}",
                out_segmentation_path=out_ref_region
            )
            outfile = self.generate_outfile_path(method_short='wm-merged', modality=modality)
            segmentation_tools.vat_wm_region_merge(
                wmparc_segmentation_path=f"{preproc_props['FilePathSeg']}",
                bs_segmentation_path=f"{preproc_props['FilePathBSseg']}",
                wm_ref_segmentation_path=out_ref_region,
                out_image_path=outfile
            )

        elif method_name=='thresh_crop':
            outfile = self.generate_outfile_path(method_short='threshcropped', modality=modality)
            thresh_crop(input_image_path=preproc_props['FilePathCropInput'],
                        out_image_path=outfile,
                        thresh_val=preproc_props['CropThreshold'],
                        verbose=preproc_props['Verbose'],
                        copy_metadata=True)

        elif method_name=='crop_image':
            outfile = self.generate_outfile_path(method_short='crop')
            image_operations_4d.crop_image(
                input_image_path=preproc_props['FilePathCropInput'],
                out_image_path=outfile,
                x_dim=preproc_props['CropXdim'],
                y_dim=preproc_props['CropYdim']
            )
        return None
