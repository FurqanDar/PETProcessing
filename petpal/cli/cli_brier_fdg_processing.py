import argparse
import os
import petpal
from ..kinetic_modeling.parametric_images import generate_cmrglc_parametric_image_from_ki_image

_FDG_EXAMPLE_ = r"""
Example:
    - Run a FDG scan through CMR Glucose map pipeline:
      petpal-FDG-proc --sub sub-001 --ses ses-01 --glc ../sub-001/ses-01/pet/sub-001_ses-01_recording-manual_desc-PlasmaGlucose_blood.tsv
"""


def main():
    parser = argparse.ArgumentParser(prog='petpal-FDG-proc',
                                     description='Command line interface for running FDG processing.',
                                     epilog=_FDG_EXAMPLE_, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--sub',required=True,help='Subject or participant identifier')
    parser.add_argument('--ses',required=True,help='Session identifier')
    parser.add_argument('--glc',required=True,help='Path to plasma glucose concentration file')
    args = parser.parse_args()

    sub_id = args.sub
    ses_id = args.ses
    plasma_glc_path = args.glc
    seg_path = f'../derivatives/freesurfer/sub-{sub_id}/ses-{ses_id}/aparc+aseg.nii.gz'
    anat_path = f'../sub-{sub_id}/ses-{ses_id}/anat/sub-{sub_id}_ses-{ses_id}_T1w.nii.gz'
    ptac_path = f'../sub-{sub_id}/ses-{ses_id}/pet/sub-{sub_id}_ses-{ses_id}_recording-manual_blood.tsv'
    bids_dir = '..'
    dseg_file = 'dseg.tsv'


    cmrlgc_lumped_const = 0.65
    cmrlgc_rescaling_const = 100.0

    FDG_Pipeline = petpal.pipelines.pipelines.BIDS_Pipeline(sub_id=sub_id,
                                                            ses_id=ses_id,
                                                            pipeline_name='FDG_Pipeline',
                                                            raw_anat_img_path=anat_path,
                                                            segmentation_img_path=seg_path,
                                                            bids_root_dir=bids_dir,
                                                            segmentation_label_table_path=dseg_file,
                                                            raw_blood_tac_path=ptac_path)


    preproc_container = petpal.pipelines.steps_containers.StepsContainer(name='preproc')

    # Configure steps for preproc container
    thresh_crop_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_threshold_cropping(input_image_path=FDG_Pipeline.pet_path)
    registration_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_register_pet_to_t1(reference_image_path=FDG_Pipeline.anat_path,
                                                                                            half_life=petpal.utils.constants.HALF_LIVES['c11'])
    moco_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_windowed_moco()
    resample_blood_step = petpal.pipelines.preproc_steps.ResampleBloodTACStep.default_resample_blood_tac_on_scanner_times()

    # Add steps to preproc container
    preproc_container.add_step(step=thresh_crop_step)
    preproc_container.add_step(step=registration_step)
    preproc_container.add_step(step=moco_step)
    preproc_container.add_step(step=resample_blood_step)

    kinetic_modeling_container = petpal.pipelines.steps_containers.StepsContainer(name='km')

    # Configure steps for kinetic modeling container
    patlak_step = petpal.pipelines.kinetic_modeling_steps.ParametricGraphicalAnalysisStep.default_patlak()

    # Add steps to kinetic modeling container
    kinetic_modeling_container.add_step(step=patlak_step)

    km_out_path = os.path.join(bids_dir,
                               'derivatives',
                               'petpal',
                               'pipelines',
                               'FDG_Pipeline',
                               f'sub-{sub_id}',
                               f'ses-{ses_id}',
                               'km')
    patlak_slope_img = os.path.join(km_out_path, f"{sub_id}_{ses_id}_desc-patlak_slope.nii.gz")
    cmrglc_slope_path = os.path.join(km_out_path)

    FDG_Pipeline.add_container(step_container=preproc_container)
    FDG_Pipeline.add_container(step_container=kinetic_modeling_container)

    FDG_Pipeline.add_dependency(sending='thresh_crop', receiving='windowed_moco')
    FDG_Pipeline.add_dependency(sending='windowed_moco', receiving='register_pet_to_t1')
    FDG_Pipeline.add_dependency(sending='register_pet_to_t1', receiving='parametric_patlak_fit')
    FDG_Pipeline.add_dependency(sending='resample_PTAC_on_scanner', receiving='parametric_patlak_fit')

    FDG_Pipeline.update_dependencies(verbose=True)

    FDG_Pipeline()

    generate_cmrglc_parametric_image_from_ki_image(input_ki_image_path=patlak_slope_img,
                                                    output_image_path=cmrglc_slope_path,
                                                    plasma_glucose_file_path=plasma_glc_path,
                                                    glucose_rescaling_constant=1.0 / 18.0,
                                                    lumped_constant=cmrlgc_lumped_const,
                                                    rescaling_const=cmrlgc_rescaling_const)
