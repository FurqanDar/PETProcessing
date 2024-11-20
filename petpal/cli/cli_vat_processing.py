"""
Run dynamic VAT PET processing through standard protocol. Handles SUVR parametric imaging and
regional TACs for further kinetic analysis. File structure standardized in the NIL Movement
Disorders section is hard-coded. Requires list of subjects to process in a TSV, as well as paths to
directories containing registrations, BIDS PET data, and output folder to write results to.

Processing steps run:
    * Motion correction
    * Registration to MPRAGE
    * Generation of standard VAT white matter reference region
    * Write regional TACs to file
    * Calculation of SUVR
    * Blur output

This code does not handle:
    * PVC
    * Warp to atlas

"""
import os
import argparse
import pandas as pd
from petpal.preproc import preproc

_VAT_EXAMPLE_ = r"""
Example:
  - Running many subjects:
    petpal-vat-proc --subjects participants.tsv --out-dir /path/to/output --pet-dir /path/to/pet/folder/ --reg-dir /path/to/subject/Registrations/
"""


def vat_protocol(subjstring: str,
                 out_dir: str,
                 pet_dir: str,
                 reg_dir: str):
    """
    Set up parameters necessary to run VAT processing and run steps in order.

    Arguments:
        subjstring (str): Original subject ID used in study, not BIDS name.
        out_dir (str): Path to folder to which results are written.
        pet_dir (str): BIDS folder containing PET data.
        reg_dir (str): Folder containing standard registrations for all subjects.
    """
    sub, ses = rename_subs(subjstring)
    preproc_props = {
        'FilePathLabelMap': '/data/jsp/human2/goldmann/dseg.tsv',
        'FilePathAtlas': '/data/petsun43/data1/atlas/MNI152/MNI152_T1_2mm.nii',
        'FilePathWarpRef': '/data/petsun43/data1/atlas/MNI152/MNI152_T1_2mm.nii',
        'FilePathFSLPremat': '',
        'FilePathFSLPostmat': '',
        'HalfLife': 6586.2,
        'StartTimeWSS': 1800,
        'EndTimeWSS': 7200,
        'MotionTarget': (0,600),
        'RegPars': {'aff_metric': 'mattes','type_of_transform': 'DenseRigid'},
        'RefRegion': 1,
        'BlurSize': 6,
        'TimeFrameKeyword': 'FrameTimesStart',
        'Verbose': True
    }
    if ses=='':
        out_folder = f'{out_dir}/{sub}'
        out_prefix = f'{sub}_pet'
        preproc_props['FilePathMocoInp'] = f'{pet_dir}/{sub}/pet/{sub}_pet.nii.gz'
        preproc_props['FilePathSeg'] = f'{reg_dir}/{sub}/{sub}_aparc+aseg.nii'
        preproc_props['FilePathBSseg'] = f'{reg_dir}/{sub}/{sub}_brainstem.nii'
        preproc_props['FilePathAnat'] = f'{reg_dir}/{sub}/{sub}_mpr.nii'
    else:
        out_folder = f'{out_dir}/{sub}_{ses}'
        out_prefix = f'{sub}_{ses}_pet'
        preproc_props['FilePathMocoInp'] = f'{pet_dir}/{sub}/{ses}/pet/{sub}_{ses}_trc-18FVAT_pet.nii.gz'
        preproc_props['FilePathSeg'] = f'{reg_dir}/{sub}_{ses}/{sub}_{ses}_aparc+aseg.nii'
        preproc_props['FilePathBSseg'] = f'{reg_dir}/{sub}_{ses}/{sub}_{ses}_brainstem.nii'
        preproc_props['FilePathAnat'] = f'{reg_dir}/{sub}_{ses}/{sub}_{ses}_mpr.nii'
    sub_vat = preproc.PreProc(
        output_directory=out_folder,
        output_filename_prefix=out_prefix
    )
    real_files = [
        preproc_props['FilePathMocoInp'],
        preproc_props['FilePathSeg'],
        preproc_props['FilePathAnat'],
        preproc_props['FilePathBSseg']
    ]
    for check in real_files:
        if not os.path.exists(check):
            print(f'{check} not found')
            return None
    print(real_files)
    preproc_props['FilePathRegInp'] = sub_vat.generate_outfile_path(method_short='moco')
    sub_vat.update_props(preproc_props)
    sub_vat.run_preproc('motion_corr')
    sub_vat.run_preproc('register_pet')
    sub_vat.run_preproc('vat_wm_ref_region')
    preproc_props['FilePathSeg'] = sub_vat.generate_outfile_path(method_short='wm-merged')
    sub_vat.update_props(preproc_props)
    preproc_props['FilePathTACInput'] = sub_vat.generate_outfile_path(method_short='reg')
    preproc_props['FilePathWSSInput'] = sub_vat.generate_outfile_path(method_short='reg')
    preproc_props['FilePathSUVRInput'] = sub_vat.generate_outfile_path(method_short='wss')
    sub_vat.update_props(preproc_props)
    sub_vat.run_preproc('write_tacs')
    sub_vat.run_preproc('weighted_series_sum')
    sub_vat.run_preproc('suvr')
    preproc_props['FilePathWarpInput'] = sub_vat.generate_outfile_path(method_short='suvr')
    preproc_props['FilePathBlurInput'] = sub_vat.generate_outfile_path(method_short='suvr')
    sub_vat.update_props(preproc_props)
    sub_vat.run_preproc('gauss_blur')
    return None


def rename_subs(sub: str):
    """
    Handle converting original subject ID to BIDS name.

    Arguments:
        sub (str): Original subject ID

    Returns:
        subname (str): Subject portion of ID converted to BIDS format.
        sesname (str): Session portion of ID converted to BIDS format, if any.
    """
    if 'VAT' in sub:
        return [f'sub-{sub}', '']
    elif 'PIB' in sub:
        subname, sesname = sub.split('_')
        subname = subname.replace('-','')
        subname = f'sub-{subname}'
        sesname = f'ses-{sesname}'
        return [subname, sesname]


def main():
    """
    VAT command line interface. Handles options, loops through subjects and runs.
    """
    parser = argparse.ArgumentParser(prog='petpal-vat-proc',
                                     description='Command line interface for running VAT processing.',
                                     epilog=_VAT_EXAMPLE_, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s','--subjects',required=True,help='Path to participants.tsv')
    parser.add_argument('-o','--out-dir',required=True,help='Output directory analyses are saved to.')
    parser.add_argument('-p','--pet-dir',required=True,help='Path to parent directory of PET imaging data.')
    parser.add_argument('-r','--reg-dir',required=True,help='Path to parent directory of registrations computed from MPR to atlas space.')
    args = parser.parse_args()

    subs_sheet = pd.read_csv(args.subjects,sep='\t')
    subs = subs_sheet['participant_id']

    for sub in subs:
        vat_protocol(subjstring=sub,out_dir=args.out_dir,pet_dir=args.pet_dir,reg_dir=args.reg_dir)
