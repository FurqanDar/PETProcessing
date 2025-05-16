# pylint: skip-file
import os
import glob
import argparse
import pandas as pd
from petpal.kinetic_modeling import graphical_analysis,rtm_analysis
from petpal.preproc import image_operations_4d, motion_corr, register, segmentation_tools
from petpal.preproc import symmetric_geometric_transfer_matrix as sgtm
import petpal.preproc.regional_tac_extraction
from petpal.utils.bids_utils import gen_bids_like_dir_path, gen_bids_like_filename, gen_bids_like_filepath
from petpal.utils.image_io import km_regional_fits_to_tsv
from petpal.utils import useful_functions
from petpal.utils.constants import HALF_LIVES


_VAT_EXAMPLE_ = (r"""
Example:
  - Running many subjects:
    petpal-vat-proc --subjects participants.tsv --out-dir /path/to/output --pet-dir /path/to/pet/folder/ --reg-dir /path/to/subject/Registrations/
""")
def infer_group(sub_id: str):
    """
    Infer the group a subject belongs to based on the subject ID.

    PIB and VATNL belong to 'HC' group while VATDYS belongs to CD group.
    """
    group = 'UNK'
    if 'PIB' in sub_id:
        group = 'HC'
    if 'VATNL' in sub_id:
        group = 'HC'
    if 'VATDYS' in sub_id:
        group = 'CD'
    return group


def rename_subs(sub: str):
    """
    Handle converting subject ID to BIDS structure.

    VATDYS0XX -> sub-VATDYS0XX
    PIBXX-YYY_VYrZ -> sub-PIBXXYYY_ses-VYrZ

    returns:
        - subject part string
        - session part string
    """
    if 'VAT' in sub:
        return [f'sub-{sub}', 'ses-VYr0']
    elif 'PIB' in sub:
        subname, sesname = sub.split('_')
        subname = subname.replace('-','')
        subname = f'sub-{subname}'
        sesname = f'ses-{sesname}'
        return [subname, sesname]


def vat_protocol(subjstring: str,
                 out_dir: str,
                 pet_dir: str,
                 fs_dir: str,
                 skip: list):
    sub, ses = rename_subs(subjstring)
    sub_id = sub.replace('sub-','')
    ses_id = ses.replace('ses-','')
    segmentation_label_file = '/home/usr/goldmann/dseg.tsv'
    motion_target = (0,600)
    reg_pars = {'aff_metric': 'mattes','type_of_transform': 'DenseRigid'}
    half_life = HALF_LIVES['f18']
    suvr_start = 1800
    suvr_end = 7200
    pvc_fwhm_mm = 4.2
    if 'VAT' in sub:
        pet_file = f'{pet_dir}/{sub}/pet/{sub}_pet.nii.gz'
        freesurfer_file = glob.glob(f'{fs_dir}/{subjstring}*/mri/aparc+aseg.mgz')[0]
        brainstem_segmentation_file = glob.glob(f'{fs_dir}/{subjstring}*/mri/brainstemSsLabels.v13.FSvoxelSpace.mgz')[0]
        mprage_file = glob.glob(f'{fs_dir}/{subjstring}*/mri/nu.mgz')[0]
        atlas_warp_file = f'PLACEHOLDER'
        mpr_brain_mask_file = glob.glob(f'{fs_dir}/{subjstring}*/mri/brainmask.mgz')[0]
    else:
        pet_file = f'{pet_dir}/{sub}/{ses}/pet/{sub}_{ses}_trc-18FVAT_pet.nii.gz'
        freesurfer_file = glob.glob(f'{fs_dir}/{subjstring}*/mri/aparc+aseg.mgz')[0]
        brainstem_segmentation_file = glob.glob(f'{fs_dir}/{subjstring}*/mri/brainstemSsLabels.v13.FSvoxelSpace.mgz')[0]
        mprage_file = glob.glob(f'{fs_dir}/{subjstring}*/mri/nu.mgz')[0]
        atlas_warp_file = f'PLACEHOLDER'
        mpr_brain_mask_file = glob.glob(f'{fs_dir}/{subjstring}*/mri/brainmask.mgz')[0]
    real_files = [
        pet_file,
        freesurfer_file,
        mprage_file,
        brainstem_segmentation_file,
        atlas_warp_file,
        mpr_brain_mask_file
    ]
    for check in real_files:
        if not os.path.exists(check):
            print(f'{check} not found')
    print(real_files)


    def vat_bids_filepath(suffix,folder,**extra_desc):

        new_dir = gen_bids_like_dir_path(sub_id=sub_id,
                                         ses_id=ses_id,
                                         sup_dir=out_dir,
                                         modality=folder)
        os.makedirs(new_dir,exist_ok=True)
        new_file = gen_bids_like_filepath(sub_id=sub_id,
                                          ses_id=ses_id,
                                          bids_dir=out_dir,
                                          modality=folder,
                                          suffix=suffix,
                                          **extra_desc)

        return new_file


    # preprocessing
    pet_cropped_file = vat_bids_filepath(suffix='pet',folder='pet',crop='003')
    if 'crop' not in skip:
        image_operations_4d.SimpleAutoImageCropper(input_image_path=pet_file,
                                                   out_image_path=pet_cropped_file,
                                                   thresh_val=0.03)

    pet_moco_file = vat_bids_filepath(suffix='pet',folder='pet',moco='windowed')
    if 'moco' not in skip:
        motion_corr.windowed_motion_corr_to_target(input_image_path=pet_cropped_file,
                                                out_image_path=pet_moco_file,
                                                motion_target_option=motion_target,
                                                w_size=300)

    pet_reg_anat_file = vat_bids_filepath(suffix='pet',folder='pet',moco='windowed',space='mpr')
    if 'register' not in skip:
        register.register_pet(input_reg_image_path=pet_moco_file,
                            out_image_path=pet_reg_anat_file,
                            reference_image_path=mprage_file,
                            motion_target_option=motion_target,
                            half_life=half_life,
                            verbose=True,
                            **reg_pars)
    
    vat_wm_ref_region_roi_file = vat_bids_filepath(suffix='seg',folder='pet',desc='RefRegionROI')
    vat_wm_ref_segmentation_file = vat_bids_filepath(suffix='seg',folder='pet',desc='RefRegionSegmentation')
    if 'refregion' not in skip:
        segmentation_tools.vat_wm_ref_region(input_segmentation_path=freesurfer_file,
                                             out_segmentation_path=vat_wm_ref_region_roi_file)
        segmentation_tools.vat_wm_region_merge(wmparc_segmentation_path=freesurfer_file,
                                               bs_segmentation_path=brainstem_segmentation_file,
                                               wm_ref_segmentation_path=vat_wm_ref_region_roi_file,
                                               out_image_path=vat_wm_ref_segmentation_file)

    tac_save_dir = gen_bids_like_dir_path(sub_id=sub_id,ses_id=ses_id,sup_dir=out_dir,modality='tacs')
    os.makedirs(tac_save_dir,exist_ok=True)
    tac_prefix = f'{sub}_{ses}'
    if 'tacs' not in skip:
        petpal.preproc.regional_tac_extraction.write_tacs(input_image_path=pet_reg_anat_file,
                                                          label_map_path=segmentation_label_file,
                                                          segmentation_image_path=vat_wm_ref_segmentation_file,
                                                          out_tac_dir=tac_save_dir,
                                                          verbose=True,
                                                          out_tac_prefix=tac_prefix)

    # kinetic modeling
    wmref_tac_path = vat_bids_filepath(suffix='tac',folder='tacs',seg='WMRef',ext='.tsv')
    mrtm_save_dir = gen_bids_like_dir_path(sub_id=sub_id,ses_id=ses_id,sup_dir=out_dir,modality='mrtm_fits')
    km_save_dir = gen_bids_like_dir_path(sub_id=sub_id,ses_id=ses_id,sup_dir=out_dir,modality='km')
    os.makedirs(km_save_dir,exist_ok=True)
    mrtm_save_path = vat_bids_filepath(suffix='fits',model='mrtm1',folder='km',ext='.tsv')
    os.makedirs(mrtm_save_dir,exist_ok=True)
    mrtm1_path = gen_bids_like_filename(sub_id=sub_id,ses_id=ses_id,model='mrtm1',suffix='fits',ext='')
    if 'mrtm1' not in skip:
        mrtm1_analysis = rtm_analysis.MultiTACRTMAnalysis(ref_tac_path=wmref_tac_path,
                                                          roi_tacs_dir=tac_save_dir,
                                                          output_directory=mrtm_save_dir,
                                                          output_filename_prefix=mrtm1_path,
                                                          method='mrtm')
        mrtm1_analysis.run_analysis(t_thresh_in_mins=10)
        mrtm1_analysis.save_analysis()
        km_regional_fits_to_tsv(fit_results_dir=mrtm_save_dir,out_tsv_dir=mrtm_save_path)

    logan_save_dir = gen_bids_like_dir_path(sub_id=sub_id,ses_id=ses_id,sup_dir=out_dir,modality='logan_fits')
    logan_save_path = vat_bids_filepath(suffix='fits',model='altlogan',folder='km',ext='.tsv')
    os.makedirs(logan_save_dir,exist_ok=True)
    logan_path = gen_bids_like_filename(sub_id=sub_id,ses_id=ses_id,model='altlogan',suffix='fits',ext='')
    if 'logan' not in skip:
        graphical_model = graphical_analysis.MultiTACGraphicalAnalysis(
            input_tac_path=wmref_tac_path,
            roi_tacs_dir=tac_save_dir,
            output_directory=logan_save_dir,
            output_filename_prefix=logan_path,
            method='alt_logan',
            fit_thresh_in_mins=10
        )
        graphical_model.run_analysis()
        graphical_model.save_analysis()
        km_regional_fits_to_tsv(fit_results_dir=logan_save_dir,out_tsv_dir=logan_save_path)

    # suvr
    wss_file_path = vat_bids_filepath(suffix='pet',folder='pet',space='mpr',desc='WSS')
    suvr_file_path = vat_bids_filepath(suffix='pet',folder='pet',space='mpr',desc='SUVR')
    if 'suvr' not in skip:
        useful_functions.weighted_series_sum(input_image_4d_path=pet_reg_anat_file,
                                             half_life=half_life,
                                             verbose=True,
                                             start_time=suvr_start,
                                             end_time=suvr_end,
                                             out_image_path=wss_file_path)
        image_operations_4d.suvr(input_image_path=wss_file_path,
                                 out_image_path=suvr_file_path,
                                 segmentation_image_path=vat_wm_ref_segmentation_file,
                                 ref_region=1,
                                 verbose=True)

    if 'pvc' not in skip:
        suvr_pvc_path = vat_bids_filepath(suffix='pet',folder='pet',space='mpr',pvc='SGTM',desc='SUVR',ext='.tsv')
        sgtm.Sgtm(input_image_path=suvr_file_path,
                  segmentation_image_path=vat_wm_ref_segmentation_file,
                  fwhm=pvc_fwhm_mm,
                  out_tsv_path=suvr_pvc_path)


def main():
    """
    VAT command line interface
    """
    parser = argparse.ArgumentParser(prog='petpal-vat-proc',
                                     description='Command line interface for running VAT processing.',
                                     epilog=_VAT_EXAMPLE_, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s','--subjects',required=True,help='Path to participants.tsv')
    parser.add_argument('-o','--out-dir',required=True,help='Output directory analyses are saved to.')
    parser.add_argument('--vatdys-pet',required=True,help='Path to BIDS directory storing VATDYS PET')
    parser.add_argument('--vatdys-fs',required=True,help='Path to directory storing VATDYS FreeSurfer')
    parser.add_argument('--vatnl-pet',required=True,help='Path to BIDS directory storing VATNL PET')
    parser.add_argument('--vatnl-fs',required=True,help='Path to directory storing VATNL FreeSurfer')
    parser.add_argument('--pib-pet',required=True,help='Path to BIDS directory storing PIB PET')
    parser.add_argument('--pib-fs',required=True,help='Path to directory storing PIB FreeSurfer')

    parser.add_argument('--skip',required=False,help='List of steps to skip',nargs='+',default=[])
    args = parser.parse_args()


    subs_sheet = pd.read_csv(args.subjects,sep='\t')
    subs = subs_sheet['participant_id']

    for sub in subs:
        if sub[:6]=='VATDYS':
            pet_dir = args.vatdys_pet
            fs_dir = args.vatdys_fs
        elif sub[:5]=='VATNL':
            pet_dir = args.vatnl_pet
            fs_dir = args.vatnl_fs
        elif sub[:3]=='PIB':
            pet_dir = args.pib_pet
            fs_dir = args.pib_fs
        try:
            vat_protocol(sub,args.out_dir,pet_dir,fs_dir,skip=args.skip)
        except:
            print(f'Running subject {sub} failed; trying next one.')
main()
