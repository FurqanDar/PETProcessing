Creating an Example PiB Pipeline
================================

--------
Overview
--------

This tutorial details the required steps to construct a pipeline for a project using the PiB tracer. As it is unlikely
that all the details will fit each neuroimaging project perfectly, this tutorial will explain how to alter the steps to
accommodate other project designs.

A broad overview of the steps is given below, followed by a more detailed discussion of each step.

#. Create an instance of the :class:`~petpal.pipelines.pipelines.BIDS_Pipeline` Class.
#. Create :class:`~petpal.pipelines.steps_containers.StepsContainer` objects for preprocessing and kinetic modeling and add desired steps to those containers.
#. Add containers (created in the previous step) to the pipeline and define their steps' order using :meth:`~petpal.pipelines.steps_containers.StepsPipeline.add_dependency()` and :meth:`~petpal.pipelines.steps_containers.StepsPipeline.update_dependencies()`
#. (Recommended) Verify pipeline's construction using its :meth:`~petpal.pipelines.steps_containers.StepsPipeline.plot_dependency_graph()` method and :meth:`~petpal.pipelines.steps_containers.StepsPipeline.can_steps_potentially_run()`
#. Run the pipeline.

--------------------
In-depth Walkthrough
--------------------

.. note::
    For the rest of this guide, 'pipeline' refers to a BIDS_Pipeline object, and 'container' refers to
    a StepsContainer object.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 1. Initialize BIDS_Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is most compact and efficient to define all necessary paths for a pipeline when instantiating it. However, you can always set these path
attributes post hoc. An example of this process is shown below for a PiB pipeline which will use T1w
(T1-weighted) MRI as the anatomical image, `freesurfer <https://surfer.nmr.mgh.harvard.edu/>`_ outputs
(aparc+aseg.nii.gz in this case) as the segmentation image, and a segmentation table stored as 'dseg.tsv' (set after
initialization).

.. code-block:: python

    import petpal

    sub_id = '001'
    ses_id = '001'
    seg_path = f'/example/path/to/Data/PiB_BIDS/derivatives/freesurfer/sub-{sub_id}/ses-{ses_id}/aparc+aseg.nii.gz'
    anat_path = f'/example/path/to/Data/PiB_BIDS/sub-{sub_id}/ses-{ses_id}/anat/sub-{sub_id}_ses-{ses_id}_T1w.nii.gz'
    bids_dir = '/example/path/to/Data/PiB_BIDS'

    PiB_Pipeline = petpal.pipelines.pipelines.BIDS_Pipeline(sub_id=sub_id,
                                                            ses_id=ses_id,
                                                            pipeline_name='PiB_Pipeline',
                                                            raw_anat_img_path=anat_path,
                                                            segmentation_img_path=seg_path,
                                                            bids_root_dir=bids_dir)

    PiB_Pipeline.seg_table = f'/example/path/to/Data/PiB_BIDS/derivatives/freesurfer/dseg.tsv'


^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 2. Construct Containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    The name of a StepsContainer will determine the name of the folder where the pipeline will store outputs from that
    container's steps. For example, a container named 'preproc' will store its outputs in a folder titled 'preproc'.

See the code block below for examples of the two main ways to add steps to containers. The first way is to use
classmethods called 'default[name-of-function-here]', which provide a convenient starting point for most use cases. The
second way to add a step to a container is to manually create a step from the classes provided in
:mod:`~petpal.pipelines.kinetic_modeling_steps` and :mod:`~petpal.pipelines.preproc_steps`. Use this option when there isn't a 'default' function
available or when you want to use different defaults than those provided in the given 'default' function. An example of
this is shown below, where the weighted series sum should only be computed from 30-60 minutes (1800-3600 seconds), rather
than the default of the whole scan time.

.. important::
    The order of execution of steps is NOT defined by the order in which they are added to a container (as in the code
    block below), but rather as shown in Step 3.

.. code-block:: python

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

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 3. Add Containers to Pipeline and Order Their Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    If you're not sure of the name of a step (i.e. a step generated using a 'default' function), there are a number of
    functions to help (:meth:`~petpal.pipelines.steps_containers.StepsPipeline.print_steps_names()`, for one). Alternatively,
    you can pass a name keyword argument to the 'default' function to override the default name.

.. code-block:: python

    PiB_Pipeline.add_container(step_container=preproc_container)
    PiB_Pipeline.add_container(step_container=kinetic_modeling_container)

    PiB_Pipeline.add_dependency(sending='thresh_crop', receiving='windowed_moco')
    PiB_Pipeline.add_dependency(sending='windowed_moco', receiving='register_pet_to_t1')
    PiB_Pipeline.add_dependency(sending='register_pet_to_t1', receiving='write_roi_tacs')
    PiB_Pipeline.add_dependency(sending='register_pet_to_t1', receiving='weighted_series_sum')
    PiB_Pipeline.add_dependency(sending='weighted_series_sum', receiving='suvr')

    PiB_Pipeline.update_dependencies(verbose=True)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 4. (Recommended) Visualize and Verify Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point, the order of the steps is clear and the pipeline should be ready to run. To verify, use
:meth:`~petpal.pipelines.steps_containers.StepsPipeline.plot_dependency_graph()` to visualize the pipeline in a plot
like the one shown below and :meth:`~petpal.pipelines.steps_containers.StepsPipeline.can_steps_potentially_run()` to
determine if all arguments are prepared as the pipeline expects.

.. plot::
    :caption: The dependency graph of the PiB pipeline described.

    import petpal

    PiB_Pipeline = petpal.pipelines.steps_containers.StepsPipeline(name='PiB_Pipeline', step_containers=[])

    preproc_container = petpal.pipelines.steps_containers.StepsContainer(name='preproc')

    thresh_crop_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_threshold_cropping()
    registration_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_register_pet_to_t1()
    moco_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_windowed_moco()
    write_tacs_step = petpal.pipelines.preproc_steps.TACsFromSegmentationStep.default_write_tacs_from_segmentation_rois()
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
                                                                segmentation_image_path='',
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

    PiB_Pipeline.plot_dependency_graph()


^^^^^^^^^^^^^^^^^^^^^^^^
Step 5. Run the Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^

Running a pipeline is as simple as calling the object.

.. code-block:: python

    PiB_Pipeline()

Congrats! You've now created a functional pipeline to take a participants raw data and outputs TACs and an SUVr parametric image.