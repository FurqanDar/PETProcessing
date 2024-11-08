"""
Command-line interface (CLI) for processing PET images collected using the PiB tracer.

This module provides a CLI to generate TACs and SUVR parametric images using input PET and MRI data from a
BIDS-compliant directory. It assumes all input PET data was collected using Pittsburgh Compound B (PiB) as the
radiotracer.

The user MUST provide the following:
    * Subject ID (--sub): Subject index, following BIDS. Usually just a number like 011 or 1023.
    * Session ID (--ses): Session index, following BIDS. Same as above.
        * Note that if you're bids directory doesn't have session folder, you'll need to create them.

This script uses one instance of the class :class:`petpal.preproc.preproc.PreProc` to
run all processing from motion-correction to SUVR generation

Assumptions regarding input data:
    * Run this cli (petpal-pib-proc) with the help flag (-h) for a list of all assumptions for each optional flag.

Example:
    .. code-block:: bash

        petpal-pib-proc --sub 021 --ses 01 -v

See Also:
    :mod:`petpal.preproc`

"""

import argparse
import logging
import os
import petpal.preproc

logger = logging.getLogger(__name__)


def _parse_command_line_args() -> argparse.Namespace:
    """
    Retrieves arguments and flags from stdin using argparse.ArgumentParser().

    Returns:
        argsparse.Namespace: "args" object with key-value pairs for each argument passed from stdin.
    """
    parser = argparse.ArgumentParser(description='Script to process PiB PET data from 4D-PET to SUVR Parametric Image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sub', required=True, help='Subject ID assuming sub_XXXX where XXXX is the subject ID.')
    parser.add_argument('--ses', required=True, help='Session ID assuming ses_XX where XX is the session ID.')
    parser.add_argument('-b', '--bids-dir', required=False, default=None,
                        help='Path to bids directory. If not set, assumes current working directory is the code/ '
                             'directory of a BIDS-like directory. ')
    parser.add_argument('-p', '--pet', required=False, default=None,
                        help="Path to input 4D PET File. If not set, assumes file named sub-XXXX_ses-XX_T1w.nii.gz "
                             "can be found in subjects's anat/ directory.")
    parser.add_argument('-m', '--mri', required=False, default=None,
                        help="Path to input T1 MRI File. If not set, assumes file named sub-XXXX_ses-XX_pet.nii.gz"
                             " can be found in subject's pet/ directory.")
    parser.add_argument('-o', '--outdir', required=False, default=None,
                        help="Directory to store all outputs. If not set, will use existing derivatives/ directory "
                             "or create one.")
    parser.add_argument('-l', '--label-map', required=False, default=None,
                        help='Path to .tsv file containing segmentation LUT info. If not set, assumes file can be '
                             'found in derivatives/freesurfer/dseg.tsv.')
    parser.add_argument('-s', '--seg', required=False, default=None,
                        help='Path to segmentation file (.nii.gz). If not set, assumes file can be found in '
                             'derivatives/freesurfer/sub-XXXX-ses-XX/aparc+aseg.nii.gz.')
    parser.add_argument('-v', '--verbose', help='Display more information while running', action='store_true')
    parser.add_argument('--skip-moco', help='Do not perform motion correction', action='store_true')
    parser.add_argument('--skip-register', help='Do not perform registration', action='store_true')
    parser.add_argument('--skip-tacs', help='Do not generate TACs', action='store_true')
    parser.add_argument('--skip-suvr', help='Do not generate SUVR images', action='store_true')
    args = parser.parse_args()

    return args


def process_single_subject_session(preproc_instance: petpal.preproc.preproc.PreProc,
                                   skip_motion_correct: bool = False,
                                   skip_register: bool = False,
                                   skip_tacs: bool = False,
                                   skip_suvr: bool = False, ) -> None:
    """
    Processes one subject-session pair using an instance of the :class:`petpal.preproc.preproc.PreProc` class.

    Processing follows these steps:
        1. Motion Correct PET
        2. Register Motion-Corrected (AKA "MoCo'd") PET to MRI
        3. Extract TACs from each Freesurfer ROI
        4. Generate SUVR Parametric Image

    """

    # Step 1: Run Motion Correction on 4DPET image
    if not skip_motion_correct:
        logger.info(f'Performing Motion Correction on {preproc_instance.preproc_props["FilePathMocoInp"]}...')
        preproc_instance.run_preproc(method_name='thresh_crop', modality='pet')
        preproc_instance.run_preproc(method_name='motion_corr', modality='pet')

    # Step 2: Register MoCo'd PET to MRI space
    if not skip_register:
        logger.info(
            f'Registering {preproc_instance.preproc_props["FilePathRegInp"]} to '
            f'{preproc_instance.preproc_props["FilePathAnat"]} space...')
        preproc_instance.run_preproc(method_name='register_pet', modality='pet')

    # Step 3: Extract TACs from each Brain ROI using Freesurfer Segmentation Input
    if not skip_tacs:
        logger.info(
            f'Extracting TACs from registered PET using segmentation '
            f'{preproc_instance.preproc_props["FilePathSeg"]}...')
        try:
            preproc_instance.run_preproc(method_name='write_tacs', modality='pet')
        except ValueError:
            logger.info(
                f'Segmentation is not in anatomical space. Resampling segmentation '
                f'{preproc_instance.preproc_props["FilePathSeg"]} to anatomical space...')
            preproc_instance.run_preproc(method_name='resample_segmentation')
            preproc_instance.run_preproc(method_name='write_tacs')

    # Step 4: Compute SUVR with Cerebellar Gray Matter as Reference Region
    if not skip_suvr:
        logger.info(
            f'Starting SUVR Process for Registered and Motion-Corrected copy of '
            f'{preproc_instance.preproc_props["FilePathRegInp"]}...')
        preproc_instance.run_preproc(method_name='weighted_series_sum', modality='pet')
        preproc_instance.run_preproc(method_name='suvr')

    return


def main():
    """
    PiB Pipeline Command Line Interface.
    """

    # Unpack command line arguments
    args = _parse_command_line_args()

    subject_id = args.sub
    session_id = args.ses
    bids_dir = os.path.abspath('../') if args.bids_dir is None else os.path.abspath(args.bids_dir)
    label_map_path = os.path.join(bids_dir, 'derivatives', 'freesurfer',
                                  'dseg.tsv') if args.label_map is None else os.path.abspath(args.label_map)
    path_to_pet = os.path.join(bids_dir, f'sub-{subject_id}', f'ses-{session_id}', 'pet',
                               f'sub-{subject_id}_ses-{session_id}_pet.nii.gz') if args.pet is None else os.path.abspath(
        args.pet)
    path_to_mri = os.path.join(bids_dir, f'sub-{subject_id}', f'ses-{session_id}', 'anat',
                               f'sub-{subject_id}_ses-{session_id}_T1w.nii.gz') if args.mri is None else os.path.abspath(
        args.mri)
    path_to_seg = os.path.join(bids_dir, 'derivatives', 'freesurfer', f'sub-{subject_id}', f'ses-{session_id}',
                               'aparc+aseg.nii.gz') if args.seg is None else os.path.abspath(args.seg)
    outdir = os.path.join(bids_dir, 'derivatives', 'petpal', 'pib_processing', f'sub-{subject_id}',
                          f'ses-{session_id}') if args.outdir is None else os.path.abspath(args.outdir)
    verbose = args.verbose
    skip_motion_correct = args.skip_moco
    skip_register = args.skip_register
    skip_tacs = args.skip_tacs
    skip_suvr = args.skip_suvr

    logger.setLevel(level=logging.INFO if verbose else logging.WARNING)

    if not os.path.exists(bids_dir):
        raise FileNotFoundError(f'No such directory: {bids_dir}')

    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f'No such file {label_map_path}')

    if not os.path.exists(path_to_mri):
        raise FileNotFoundError(f'No such file {path_to_mri}')

    if not os.path.exists(path_to_seg):
        raise FileNotFoundError(f'No such file {path_to_seg}')

    if not os.path.exists(path_to_pet):
        raise FileNotFoundError(f'No such file {path_to_pet}')

    # TODO: Make a shared global dict with Half-Lives of common Radiotracers
    half_life = 1221.66  # C-11 Half-Life According to NIST

    subject = petpal.preproc.preproc.PreProc(output_directory=outdir,
                                             output_filename_prefix=subject_id)

    # Update props with everything necessary.
    properties = {
        'FilePathCropInput': path_to_pet,
        'FilePathAnat': path_to_mri,
        'FilePathSeg': path_to_seg,
        'FilePathWSSInput': subject.generate_outfile_path(method_short='reg', modality='pet'),
        'FilePathMocoInp': subject.generate_outfile_path(method_short='threshcropped', modality='pet'),
        'HalfLife': half_life,
        'CropThreshold': .01,
        'Verbose': verbose,
        'MotionTarget': (0, 600),  # Use the summed first 10 minutes as motion target
        'FilePathRegInp': subject.generate_outfile_path(method_short='moco', modality='pet'),
        'FilePathTACInput': subject.generate_outfile_path(method_short='reg', modality='pet'),
        'FilePathLabelMap': label_map_path,
        'TimeFrameKeyword': 'FrameTimesStart',
        'FilePathSUVRInput': subject.generate_outfile_path(method_short='wss', modality='pet'),
        'RefRegion': 8,  # TODO: Make a function for combining ROIs
        'StartTimeWSS': 1800,
        'EndTimeWSS': 3600
    }
    subject.update_props(properties)

    process_single_subject_session(subject, skip_motion_correct, skip_register, skip_tacs, skip_suvr)


if __name__ == "__main__":
    main()
