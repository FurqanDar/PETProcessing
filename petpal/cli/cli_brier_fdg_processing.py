import argparse
from ..preproc import preproc
import os
import numpy as np
from ..utils import image_io
from ..input_function.blood_input import BloodInputFunction
from ..kinetic_modeling.parametric_images import GraphicalAnalysisParametricImage
import nibabel


def fdg_protocol_with_arterial(sub_id: str,
                               ses_id: str,
                               bids_root_dir: str = None,
                               pet_dir_path: str = None,
                               anat_dir_path: str = None,
                               out_dir_path: str = None,
                               run_crop: bool = False,
                               run_wss: bool = False,
                               run_moco: bool = False,
                               run_reg: bool = False,
                               run_resample: bool = False,
                               run_patlak: bool = False,
                               run_cmrglc: bool = False,
                               verbose: bool = False):
    r"""
    Perform a complete FDG PET preprocessing and analysis pipeline using the specified options. Refer to the
    Notes for explanations of the default behavior of arguments. This function is intended to be used with
    BIDs-like datasets.
    
    This function conducts a series of preprocessing steps (each of which can be skipped)
        - threshold cropping
        - weighted series summation
        - motion correction,
        - registration
        - resampling of blood TAC data on scanner frame times
        - Patlak analysis (parametric images)
        - CMRglc (parametric image)
        
    Args:
        sub_id (str): Subject ID in the BIDS format (e.g., '01').
        ses_id (str): Session ID in the BIDS format (e.g., '01').
        bids_root_dir (str, optional): Root directory of the BIDS dataset. If not provided, assumes the
            current working directory is within a BIDS dataset and set to its parent directory.
        pet_dir_path (str, optional): Path to the PET directory. If not provided, constructs the path
            based on `bids_root_dir`, `sub_id`, and `ses_id`.
        anat_dir_path (str, optional): Path to the anatomical directory. If not provided, constructs the path
            based on `bids_root_dir`, `sub_id`, and `ses_id`.
        out_dir_path (str, optional): Output directory path. If not provided, constructs default output directory
            within the BIDS derivatives folder under `BIDS_ROOT_DIR`.
        run_crop (bool): Whether to perform threshold cropping. Default is False.
        run_wss (bool): Whether to perform weighted series summation. Default is False.
        run_moco (bool): Whether to perform motion correction. Default is False.
        run_reg (bool): Whether to perform registration. Default is False.
        run_resample (bool): Whether to resample blood TAC data on scanner times. Default is False.
        run_patlak (bool): Whether to perform Patlak analysis. Default is False.
        run_cmrglc (bool): Whether to generate CMRglc images. Default is False.
        verbose (bool): Whether to print verbose output during processing. Default is False.

    Returns:
        None
        
    Notes:
        The pipeline-function is intended to be used with BIDs-like datasets where we have the following assumptions
        about the naming conventions of different file types.
        - In the ``pet_dir_path`` directory we have the following files:
            - 4D-PET: ``sub-{sub_id}_ses-{ses_id}_pet.nii.gz``
            - Blood TAC: ``sub-{sub_id}_ses-{ses_id}_desc-decaycorrected_blood.tsv``
        - In the ``anat_dir_path`` directory we have the following file:
            - T1w image in MPRAGE: ``sub-{sub_id}_ses-{ses_id}_MPRAGE.nii.gz``
        - If ``bids_root_dir`` is not provided, it defaults to the parent directory, assuming the current working directory
          is within the ``code`` directory of a BIDS dataset.
        - Default output directory `out_dir_path` is constructed as:
          ``<BIDS_ROOT_DIR>/derivatives/petpal/pipeline_brier_fdg/sub-<sub_id>/ses-<ses_id>``.
        - Various preprocessing properties such as `CropThreshold`, `HalfLife`, and `TimeFrameKeyword` are set internally
          with standard default values tailored for FDG PET processing.
          
    See Also:
        * :class:`preproc`
        * :class:`BloodInputFunction`
        * :class:`GraphicalAnalysisParametricImage`
          
    """
    sub_ses_prefix = f'sub-{sub_id}_ses-{ses_id}'
    
    if bids_root_dir is not None:
        BIDS_ROOT_DIR = os.path.abspath(bids_root_dir)
    else:
        BIDS_ROOT_DIR = os.path.abspath("../")
    
    if out_dir_path is not None:
        out_dir = os.path.abspath(out_dir_path)
    else:
        out_dir = os.path.join(BIDS_ROOT_DIR, "derivatives", "petpal", "pipeline_brier_fdg", f"sub-{sub_id}", f"ses-{ses_id}")
        os.makedirs(out_dir, exist_ok=True)
        
    out_dir_preproc = os.path.join(out_dir, "preproc")
    out_dir_kinetic_modeling = os.path.join(out_dir, "kinetic_modeling")
    os.makedirs(out_dir_preproc, exist_ok=True)
    os.makedirs(out_dir_kinetic_modeling, exist_ok=True)
    
    sub_path = os.path.join(f"{BIDS_ROOT_DIR}", f"sub-{sub_id}", f"ses-{ses_id}")

    if pet_dir_path is not None:
        pet_dir = os.path.abspath(pet_dir_path)
    else:
        pet_dir = os.path.join(f"{sub_path}", "pet")
        
    if anat_dir_path is not None:
        anat_dir = os.path.abspath(anat_dir_path)
    else:
        anat_dir = os.path.join(f"{sub_path}", "anat")
    
    raw_pet_img_path = os.path.join(pet_dir, f"{sub_ses_prefix}_pet.nii.gz")
    t1w_reference_img_path = os.path.join(anat_dir, f"{sub_ses_prefix}_MPRAGE.nii.gz")
    raw_blood_tac_path = os.path.join(pet_dir, f"{sub_ses_prefix}_desc-decaycorrected_blood.tsv")
    resample_tac_path = os.path.join(out_dir_preproc, f"{sub_ses_prefix}_desc-onscannertimes_blood.tsv")
   
    out_mod = 'pet'
    lin_fit_thresh_in_mins = 30.0
    cmrlgc_lumped_const = 0.65
    cmrlgc_rescaling_const = 100.0
    
    preproc_props = {
        'FilePathAnat': t1w_reference_img_path,
        'FilePathCropInput': raw_pet_img_path,
        'CropThreshold': 8.5e-3,
        'HalfLife': 6586.2,
        'StartTimeWSS':0,
        'EndTimeWSS':1800,
        'RegPars': {'aff_metric': 'mattes', 'type_of_transform': 'Rigid'},
        'MocoTransformType' : 'Affine',
        'MocoPars' : {'verbose':True},
        'TimeFrameKeyword': 'FrameReferenceTime',
        'Verbose': verbose,
        }
    
    sub_preproc = preproc.PreProc(output_directory=out_dir_preproc, output_filename_prefix=sub_ses_prefix)
    preproc_props['FilePathWSSInput'] = sub_preproc._generate_outfile_path(method_short='threshcropped', modality=out_mod)
    preproc_props['FilePathMocoInp'] = preproc_props['FilePathWSSInput']
    preproc_props['MotionTarget'] = 'mean_image'
    preproc_props['FilePathRegInp'] = sub_preproc._generate_outfile_path(method_short='moco', modality=out_mod)
    preproc_props['FilePathTACInput'] = sub_preproc._generate_outfile_path(method_short='reg', modality=out_mod)
    sub_preproc.update_props(preproc_props)
    
    if run_crop:
        sub_preproc.run_preproc(method_name='thresh_crop', modality=out_mod)
    if run_wss:
        sub_preproc.run_preproc(method_name='weighted_series_sum', modality=out_mod)
    if run_moco:
        sub_preproc.run_preproc(method_name='motion_corr_frames_above_mean', modality=out_mod)
    if run_reg:
        sub_preproc.run_preproc(method_name='register_pet', modality=out_mod)
    if run_resample:
        resample_blood_data_on_scanner_times(pet4d_path=preproc_props['FilePathTACInput'],
                                             raw_blood_tac=raw_blood_tac_path,
                                             lin_fit_thresh_in_mins=lin_fit_thresh_in_mins,
                                             out_tac_path=resample_tac_path)
    if run_patlak:
        patlak_obj = GraphicalAnalysisParametricImage(input_tac_path=resample_tac_path,
                                                      pet4D_img_path=preproc_props['FilePathTACInput'],
                                                      output_directory=out_dir_kinetic_modeling,
                                                      output_filename_prefix=sub_ses_prefix)
        patlak_obj.run_analysis(method_name='patlak', t_thresh_in_mins=lin_fit_thresh_in_mins)
        patlak_obj.save_analysis()
        
    if run_cmrglc:
        patlak_slope_img = os.path.join(out_dir_kinetic_modeling, f"{sub_ses_prefix}_desc-patlak_slope.nii.gz")
        cmrglc_slope_path = os.path.join(out_dir_kinetic_modeling, f"{sub_ses_prefix}_desc-cmrglc_pet.nii.gz")
        plasma_glc_path   = os.path.join(pet_dir, f"{sub_ses_prefix}_desc-bloodconcentration_glucose.txt")
        
        save_cmrglc_image_from_patlak_ki(patlak_ki_image_path=patlak_slope_img,
                                         output_file_path=cmrglc_slope_path,
                                         plasma_glucose=read_plasma_glucose_concentration(plasma_glc_path),
                                         lumped_constant=cmrlgc_lumped_const,
                                         rescaling_const=cmrlgc_rescaling_const)

_PROG_NAME_ = r"petpal-brier-fdg-pipeline"
_FDG_CMR_EXAMPLE_ = (rf"""
Example:
    - Assuming if we are in the `code` directory of a BIDS directory:
        {_PROG_NAME_} --sub-id XXXX --ses-id XX
      
    - Running with all directory options:
        {_PROG_NAME_} -i 001 -s 01 -b /path/to/bids -p /path/to/pet -a /path/to/anat -o /path/to/output --verbose
        
        
    - Assuming we are in the `code` directory of a BIDS-like directory where we skip the kinetic-modeling
      (performing a threshold crop, motion correction and registration to T1):
        {_PROG_NAME_} -i 001 -s 01 --verbose --skip-patlak --skip-cmrglc --skip-blood-resample
      
    The pipeline-function is intended to be used with BIDs-like datasets where we have the following assumptions
        about the naming conventions of different file types:
            - In the ``pet_dir_path`` directory we have the following files:
                - 4D-PET: ``sub-{{sub_id}}_ses-{{ses_id}}_pet.nii.gz``
                - Blood TAC: ``sub-{{sub_id}}_ses-{{ses_id}}_desc-decaycorrected_blood.tsv``
            - In the ``anat_dir_path`` directory we have the following files:
                - T1w image in MPRAGE: ``sub-{{sub_id}}_ses-{{ses_id}}_MPRAGE.nii.gz``
        - If ``bids_root_dir`` is not provided, it defaults to the parent directory, assuming the current working directory
          is within the ``code`` directory of a BIDS dataset.
        - Default output directory `out_dir_path` is constructed as:
          ``<BIDS_ROOT_DIR>/derivatives/petpal/pipeline_brier_fdg/sub-<sub_id>/ses-<ses_id>``.
""")

def main():
    rf"""
    Command line interface for generating parametric CMRglc images.

    This CLI provides access to the comprehensive FDG PET preprocessing and analysis pipeline. Multiple
    processing steps can be executed or skipped based on the provided arguments. The default behavior assumes
    a BIDS-like directory structure if not explicitly overridden.

    Command Line Arguments:
        -i, --sub-id (str): Subject ID assuming sub-XXXX where XXXX is the subject ID. (Required)
        -s, --ses-id (str): Session ID assuming ses-XX where XX is the session ID. (Required)
        -b, --bids-dir (str, optional): Base directory for the BIDS-like data for the study. If not set, assumes
            the current working directory is the code/ directory within a BIDS-like directory. Default is None.
        -p, --pet4d-dir (str, optional): Directory where the raw 4D-PET, raw blood TAC, and blood glucose files
            are located. Default is None.
        -a, --anat-dir (str, optional): Directory where the T1-weighted MR image is located. Default is None.
        -o, --out-dir (str, optional): Directory where the outputs of the pipeline will be saved. Default is None.
        -v, --verbose (bool, optional): Verbose information for each step in the pipeline. Default is False.

        --skip-crop (bool, optional): Whether to skip the cropping step in the pipeline. Default is False.
        --skip-wss (bool, optional): Whether to skip the weighted-series-sum (wss) step in the pipeline. Default is False.
        --skip-moco (bool, optional): Whether to skip the motion correction (moco) step in the pipeline. Default is False.
        --skip-reg (bool, optional): Whether to skip the registration step in the pipeline. Default is False.
        --skip-blood-resample (bool, optional): Whether to skip the blood resample step in the pipeline. Default is False.
        --skip-patlak (bool, optional): Whether to skip the Patlak analysis step in the pipeline. Default is False.
        --skip-cmrglc (bool, optional): Whether to skip the CMRglc analysis step in the pipeline. Default is False.

    Returns:
        None

    Example:
        Example command to run the CLI assuming we are in the `code` directory of a BIDS-like directory:
        
        .. code-block:: bash
        
            {_PROG_NAME_} -i 001 -s 01  --verbose
            

        Example command to run the CLI with all directory options:
            
        .. code-block:: bash
            
            {_PROG_NAME_} -i 001 -s 01 -b /path/to/bids -p /path/to/pet -a /path/to/anat -o /path/to/output --verbose
            
        
        Example command to run the CLI (assuming we are in the `code` directory of a BIDS-like directory)
        where we skip the kinetic-modeling (performing a threshold crop, motion correction and registration to T1):
        
        .. code-block:: bash
            
            {_PROG_NAME_} -i 001 -s 01 --verbose --skip-patlak --skip-cmrglc --skip-blood-resample
            
            
            
    """
    
    parser = argparse.ArgumentParser(prog=f'{_PROG_NAME_}',
                                     description='Command line interface for generating parametric CMRglc images',
                                     epilog=_FDG_CMR_EXAMPLE_,
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-i', '--sub-id', required=True,
                        help='Subject ID assuming sub_XXXX where XXXX is the subject ID.')
    parser.add_argument('-s', '--ses-id', required=True,
                        help='Session ID assuming ses_XX where XX is the session ID.')
    
    parser.add_argument('-b', '--bids-dir', required=False, default=None,
                        help='Base directory for the BIDS-like data for the study. If not set, assumes current working '
                             'directory is the code/ directory of a BIDS-like directory.')
    parser.add_argument('-p', '--pet4d-dir', required=False, default=None,
                        help='Directory where the raw 4D-PET, raw blood tac, and blood glucose files are.')
    parser.add_argument('-a', '--anat-dir', required=False, default=None,
                        help='Directory where the T1-weighted MR image is.')
    parser.add_argument('-o', '--out-dir', required=False, default=None,
                        help='Directory where the outputs of the pipeline will be saved. If not set, will generate a '
                             'derivatives directory in the BIDs root directory.')
    
    parser.add_argument('-v', '--verbose', required=False, action='store_true', default=False,
                        help='Verbose information for each step in the pipeline.')
    
    
    parser.add_argument('--skip-crop', required=False, action='store_true', default=False,
                        help='Whether to skip the cropping step in the pipeline.')
    parser.add_argument('--skip-wss', required=False, action='store_true', default=False,
                        help='Whether to skip the weighted-series-sum (wss) step in the pipeline.')
    parser.add_argument('--skip-moco', required=False, action='store_true', default=False,
                        help='Whether to skip the moco step in the pipeline.')
    parser.add_argument('--skip-reg', required=False, action='store_true', default=False,
                        help='Whether to skip the registration step in the pipeline.')
    parser.add_argument('--skip-blood-resample', required=False, action='store_true', default=False,
                        help='Whether to skip the blood resample step in the pipeline.')
    parser.add_argument('--skip-patlak', required=False, action='store_true', default=False,
                        help='Whether to skip the patlak analysis step in the pipeline.')
    parser.add_argument('--skip-cmrglc', required=False, action='store_true', default=False,
                        help='Whether to skip the CMRglc analysis step in the pipeline.')

    args = parser.parse_args()
    
    fdg_protocol_with_arterial(sub_id=args.sub_id,
                               ses_id=args.ses_id,
                               bids_root_dir=args.bids_dir,
                               pet_dir_path=args.pet4d_dir,
                               anat_dir_path=args.anat_dir,
                               run_crop=not args.skip_crop,
                               run_wss=not args.skip_wss,
                               run_moco=not args.skip_moco,
                               run_reg=not args.skip_reg,
                               run_resample=not args.skip_blood_resample,
                               run_patlak=not args.skip_patlak,
                               run_cmrglc=not args.skip_cmrglc,
                               verbose=args.verbose)


def resample_blood_data_on_scanner_times(pet4d_path: str,
                                         raw_blood_tac: str,
                                         lin_fit_thresh_in_mins: float,
                                         out_tac_path: str):
    r"""
    Resample blood time-activity curve (TAC) based on PET scanner frame times. The function assumes
    that the PET meta-data have 'FrameReferenceTime' in seconds. The saved TAC is in minutes.

    This function takes the raw blood TAC sampled at arbitrary times, resamples it
    to the frame times of a 4D PET image, and saves the resampled TAC to a file.

    Args:
        pet4d_path (str): Path to the 4D PET image file.
        raw_blood_tac (str): Path to the file containing raw blood time-activity data.
        lin_fit_thresh_in_mins (float): Threshold in minutes for piecewise linear fit.
        out_tac_path (str): Path to save the resampled blood TAC.

    Returns:
        None. In the saved TAC file, the first column will be time in minutes,
            and the second column will be the activity.

    See Also:
        - :class:`BloodInputFunction`


    Example:

       .. code-block:: python

          resample_blood_data_on_scanner_times(
              pet4d_path='pet_image.nii.gz',
              raw_blood_tac='blood_tac.csv',
              lin_fit_thresh_in_mins=0.5,
              out_tac_path='resampled_blood_tac.csv'
          )


    """
    image_meta_data = image_io.load_metadata_for_nifty_with_same_filename(image_path=pet4d_path)
    frame_times = np.asarray(image_meta_data['FrameReferenceTime']) / 60.0
    blood_times, blood_activity = image_io.safe_load_tac(filename=raw_blood_tac)
    blood_intp = BloodInputFunction(time=blood_times, activity=blood_activity, thresh_in_mins=lin_fit_thresh_in_mins)
    resampled_blood = blood_intp.calc_blood_input_function(t=frame_times)
    resampled_tac = np.asarray([frame_times, resampled_blood], dtype=float)
    
    np.savetxt(X=resampled_tac.T, fname=out_tac_path)
    
    return None


def read_plasma_glucose_concentration(file_path: str, correction_scale: float = 1.0 / 18.0) -> float:
    r"""
    Temporary hacky function to read a single plasma glucose concentration value from a file.

    This function reads a single numerical value from a specified file and applies a correction scale to it.
    The primary use is to quickly extract plasma glucose concentration for further processing. The default
    scaling os 1.0/18.0 is the one used in the CMMS study to get the right units.

    Args:
        file_path (str): Path to the file containing the plasma glucose concentration value.
        correction_scale (float): Scale factor for correcting the read value. Default is `1.0/18.0`.

    Returns:
        float: Corrected plasma glucose concentration value.
    """
    return correction_scale * float(np.loadtxt(file_path))


def save_cmrglc_image_from_patlak_ki(patlak_ki_image_path: str,
                                     output_file_path: str,
                                     plasma_glucose: float,
                                     lumped_constant: float,
                                     rescaling_const: float):
    r"""
    Generate and save a CMRglc image by rescaling a Patlak-Ki image.

    This function reads a Patlak-Ki image, rescales it using provided parameters (plasma glucose,
    lumped constant, and a rescaling constant), and saves the resulting image as a CMRglc image.

    The final image will be `rescaling_constant * K_i * plasma_glucose / lumped_constant`.

    Args:
        patlak_ki_image_path (str): Path to the Patlak-Ki image file.
        output_file_path (str): Path to save the rescaled CMRglc image.
        plasma_glucose (float): Plasma glucose concentration value used for rescaling.
        lumped_constant (float): Lumped constant value used for rescaling.
        rescaling_const (float): Additional rescaling constant applied to the Patlak-Ki values.

    Returns:
        None
    """
    patlak_image = image_io.ImageIO(verbose=False).load_nii(image_path=patlak_ki_image_path)
    patlak_affine = patlak_image.affine
    cmr_vals = (plasma_glucose / lumped_constant) * patlak_image.get_fdata() * rescaling_const
    cmr_image = nibabel.Nifti1Image(dataobj=cmr_vals, affine=patlak_affine)
    nibabel.save(cmr_image, f"{output_file_path}")
    image_io.safe_copy_meta(input_image_path=patlak_ki_image_path, out_image_path=output_file_path)
