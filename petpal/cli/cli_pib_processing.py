import argparse
import petpal

_PIB_EXAMPLE_ = (r"""
Example:
    - Run a PIB scan through SUVR pipeline:
      petpal-pib-proc --sub sub-001 --ses ses-01
""")

def main():
    parser = argparse.ArgumentParser(prog='petpal-pib-proc',
                                     description='Command line interface for running PIB processing.',
                                     epilog=_PIB_EXAMPLE_, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--sub',required=True,help='Subject or participant identifier')
    parser.add_argument('--ses',required=True,help='Session identifier')
    args = parser.parse_args()

    sub_id = args.sub
    ses_id = args.ses
    seg_path = f'../derivatives/freesurfer/sub-{sub_id}/ses-{ses_id}/aparc+aseg.nii.gz'
    anat_path = f'../sub-{sub_id}/ses-{ses_id}/anat/sub-{sub_id}_ses-{ses_id}_T1w.nii.gz'
    bids_dir = '..'
    dseg_file = 'dseg.tsv'

    PiB_Pipeline = petpal.pipelines.pipelines.BIDS_Pipeline(sub_id=sub_id,
                                                            ses_id=ses_id,
                                                            pipeline_name='PiB_Pipeline',
                                                            raw_anat_img_path=anat_path,
                                                            segmentation_img_path=seg_path,
                                                            bids_root_dir=bids_dir,
                                                            segmentation_label_table_path=dseg_file)


    preproc_container = petpal.pipelines.steps_containers.StepsContainer(name='preproc')

    # Configure steps for preproc container
    thresh_crop_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_threshold_cropping(input_image_path=PiB_Pipeline.pet_path)
    registration_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_register_pet_to_t1(reference_image_path=PiB_Pipeline.anat_path,
                                                                                            half_life=petpal.utils.constants.HALF_LIVES['c11'])
    moco_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_windowed_moco()
    write_tacs_step = petpal.pipelines.preproc_steps.TACsFromSegmentationStep.default_write_tacs_from_segmentation_rois(segmentation_image_path=PiB_Pipeline.seg_img,
                                                                                                    segmentation_label_map_path=PiB_Pipeline.seg_table)
    wss_step = petpal.pipelines.preproc_steps.ImageToImageStep(name='weighted_series_sum',
                                            function=petpal.utils.useful_functions.weighted_series_sum,
                                            input_image_path='',
                                            output_image_path='',
                                            half_life=petpal.utils.constants.HALF_LIVES['c11'],
                                            start_time=1800,
                                            end_time=3600)

    # Add steps to preproc container
    preproc_container.add_step(step=thresh_crop_step)
    preproc_container.add_step(step=registration_step)
    preproc_container.add_step(step=moco_step)
    preproc_container.add_step(step=write_tacs_step)
    preproc_container.add_step(step=wss_step)

    kinetic_modeling_container = petpal.pipelines.steps_containers.StepsContainer(name='km')

    # Configure steps for kinetic modeling container
    suvr_step = petpal.pipelines.preproc_steps.ImageToImageStep(name='suvr',
                                                                function=petpal.preproc.image_operations_4d.suvr,
                                                                input_image_path='',
                                                                output_image_path='',
                                                                ref_region=8,
                                                                segmentation_image_path=seg_path,
                                                                verbose=False)

    # Add steps to kinetic modeling container
    kinetic_modeling_container.add_step(step=suvr_step)

    PiB_Pipeline.add_container(step_container=preproc_container)
    PiB_Pipeline.add_container(step_container=kinetic_modeling_container)

    PiB_Pipeline.add_dependency(sending='thresh_crop', receiving='windowed_moco')
    PiB_Pipeline.add_dependency(sending='windowed_moco', receiving='register_pet_to_t1')
    PiB_Pipeline.add_dependency(sending='register_pet_to_t1', receiving='write_roi_tacs')
    PiB_Pipeline.add_dependency(sending='register_pet_to_t1', receiving='weighted_series_sum')
    PiB_Pipeline.add_dependency(sending='weighted_series_sum', receiving='suvr')

    PiB_Pipeline.update_dependencies(verbose=True)

    PiB_Pipeline()
