"""
This module provides a Command-line interface (CLI) for preprocessing imaging data to
produce regional PET Time-Activity Curves (TACs) and prepare data for parametric imaging analysis.

The user must provide:
    * The sub-command. Options: 'weighted-sum', 'motion-correct', 'register-pet', or 'write-tacs'.
    * Path to PET input data in NIfTI format. This can be source data, or with some preprocessing
      such as registration or motion correction, depending on the chosen operation.
    * Directory to which the output is written.
    * The name of the subject being processed, for the purpose of naming output files.
    * 3D imaging data, such as anatomical, segmentation, or PET sum, depending on the desired
      preprocessing operation.
    * Additional information needed for preprocessing, such as color table or half-life.
    

Examples:

   * Auto Crop:

   .. code-block:: bash

       petpal-preproc auto-crop -i /path/to/input_img.nii.gz -o petpal_crop.nii.gz -t 0.05


   * Windowed Motion Correction:

   .. code-block:: bash

       petpal-preproc window-motion-corr -i /path/to/input_img.nii.gz -o petpal_moco.nii.gz --window-size 120 --transform-type QuickRigid


   * Register to anatomical:

   .. code-block:: bash

       petpal-preproc register-pet -i /path/to/input_img.nii.gz -o petpal_reg.nii.gz --motion-target 0 600 --anatomical /path/to/anat.nii.gz --half-life 6584


   * Write regional tacs:

   .. code-block:: bash

       petpal-preproc write-tacs -i /path/to/input_img.nii.gz -o /tmp/petpal_tacs --segmentation /path/to/segmentation.nii.gz --label-map-path /path/to/dseg.tsv


   * Half life weighted sum of series:

   .. code-block:: bash

       petpal-preproc weighted-series-sum -i /path/to/input_img.nii.gz -o petpal_wss.nii.gz --half-life 6584 --start-time 1800 --end-time 7200


   * SUVR Image:

   .. code-block:: bash

       petpal-preproc suvr -i /path/to/input_img.nii.gz -o petpal_suvr.nii.gz --segmentation /path/to/segmentation.nii.gz --ref-region 1


   * Gaussian blur image:

   .. code-block:: bash

       petpal-preproc gauss-blur -i /path/to/input_img.nii.gz -o petpal_blur.nii.gz --blur-size-mm 8


   * Divide image by scale factor:

   .. code-block:: bash

       petpal-preproc rescale-image -i /path/to/input_img.nii.gz -o petpal_rescale.nii.gz --scale-factor 1000


   * Warp image to atlas:

   .. code-block:: bash

       petpal-preproc warp-pet-atlas -i /path/to/input_img.nii.gz -o petpal_reg-atlas.nii.gz --anatomical /path/to/anat.nii.gz --reference-atlas /path/to/atlas.nii.gz
 

See Also:
    * :mod:`~petpal.preproc.image_operations_4d` - module used for operations on 4D images.
    * :mod:`~petpal.preproc.motion_corr` - module for motion correction tools.
    * :mod:`~petpal.preproc.register` - module for MRI and atlas registration.

"""
import argparse
import ants

import petpal.preproc.regional_tac_extraction

from ..utils import useful_functions
from ..preproc import image_operations_4d, motion_corr, register


_PREPROC_EXAMPLES_ = r"""
Examples:
  - Auto crop:
    petpal-preproc auto-crop -i /path/to/input_img.nii.gz -o petpal_crop.nii.gz -t 0.05
  - Windowed moco:
    petpal-preproc window-motion-corr -i /path/to/input_img.nii.gz -o petpal_moco.nii.gz --window-size 120 --transform-type QuickRigid
  - Register to anatomical:
    petpal-preproc register-pet -i /path/to/input_img.nii.gz -o petpal_reg.nii.gz --motion-target 0 600 --anatomical /path/to/anat.nii.gz --half-life 6584
  - Write regional tacs:
    petpal-preproc write-tacs -i /path/to/input_img.nii.gz -o /tmp/petpal_tacs --segmentation /path/to/segmentation.nii.gz --label-map-path /path/to/dseg.tsv
  - Half life weighted sum of series:
    petpal-preproc weighted-series-sum -i /path/to/input_img.nii.gz -o petpal_wss.nii.gz --half-life 6584 --start-time 1800 --end-time 7200
  - SUVR:
    petpal-preproc suvr -i /path/to/input_img.nii.gz -o petpal_suvr.nii.gz --segmentation /path/to/segmentation.nii.gz --ref-region 1
  - Gauss blur:
    petpal-preproc gauss-blur -i /path/to/input_img.nii.gz -o petpal_blur.nii.gz --blur-size-mm 8
  - Divide image by scale factor:s
    petpal-preproc rescale-image -i /path/to/input_img.nii.gz -o petpal_rescale.nii.gz --scale-factor 1000
  - Warp to atlas:
    petpal-preproc warp-pet-atlas -i /path/to/input_img.nii.gz -o petpal_reg-atlas.nii.gz --anatomical /path/to/anat.nii.gz --reference-atlas /path/to/atlas.nii.gz
"""


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Adds common arguments ('--input-img', '--out-img') to a provided ArgumentParser
    object.

    This function modifies the passed ArgumentParser object by adding two arguments commonly
    used for the command line scripts. It uses the add_argument method of the ArgumentParser class.
    After running this function, the parser will be able to accept and parse these additional
    arguments from the command line when run.
    
    .. note::
        This function modifies the passed `parser` object in-place and does not return anything.

    Args:
        parser (argparse.ArgumentParser): The argument parser object to which the arguments are
            added.

    Raises:
        argparse.ArgumentError: If a duplicate argument tries to be added.

    Side Effects:
        Modifies the ArgumentParser object by adding new arguments.

    Example:
        .. code-block:: python

            parser = argparse.ArgumentParser()
            _add_common_args(parser)
            args = parser.parse_args(['--pet', 'pet_file', '--out-dir', './', '--prefix', 'prefix'])
            print(args.pet)
            print(args.out_dir)
            print(args.prefix)

    """
    parser.add_argument('-o',
                        '--out-img',
                        default='petpal_wss_output.nii.gz',
                        help='Output image filename')
    parser.add_argument('-i', '--input-img',required=True,help='Path to input image.',type=str)


def _generate_args() -> argparse.ArgumentParser:
    """
    Generates command line arguments for method :func:`main`.

    Returns:
        args (argparse.ArgumentParser): Arguments used in the command line and their corresponding
            values.
    """
    parser = argparse.ArgumentParser(prog='petpal-preproc',
                                     description="Command line interface for running PET "
                                                 "pre-processing steps.",
                                     epilog=_PREPROC_EXAMPLES_,
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help.")

    parser_wss = subparsers.add_parser('weighted-series-sum',
                                       help='Half-life weighted sum of 4D PET series.')
    _add_common_args(parser_wss)
    parser_wss.add_argument('--half-life',
                            required=True,
                            help='Half life of radioisotope in seconds.',
                            type=float)
    parser_wss.add_argument('--start-time',
                            required=False,
                            help='Start time of sum in seconds.',
                            type=float,
                            default=0)
    parser_wss.add_argument('--end-time',
                            required=False,
                            help='End time of sum in seconds.',
                            type=float,
                            default=-1)

    parser_crop = subparsers.add_parser('auto-crop',
                                        help='Automatically crop 4D PET image using threshold.')
    _add_common_args(parser_crop)
    parser_crop.add_argument('-t','--thresh-val', required=True,default=0.01,
                            help='Fractional threshold to crop image projections.',type=float)


    parser_moco = subparsers.add_parser('motion-correction',
                                        help='Motion correct 4D PET data.')
    _add_common_args(parser_moco)
    parser_moco.add_argument('--motion-target', default=None, nargs='+',
                            help="Motion target option. Can be an image path, "
                                 "'weighted_series_sum' or a tuple "
                                 "(i.e. '--motion-target 0 600' for first ten minutes).",
                            required=True)
    parser_moco.add_argument('--transform-type', required=False,default='Rigid',
                             help='Transformation type (Rigid or Affine).',type=str)
    parser_moco.add_argument('--half-life', required=False,
                             help='Half life of radioisotope in seconds.'
                                  'Required for some motion targets.',type=float)

    parser_tac = subparsers.add_parser('write-tacs',
                                       help='Write ROI TACs from 4D PET using segmentation masks.')
    parser_tac.add_argument('-i', '--input-img',required=True,help='Path to input image.',type=str)
    parser_tac.add_argument('-o',
                            '--out-tac-dir',
                            default='petpal_tacs',
                            help='Output TAC folder dir')
    parser_tac.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')
    parser_tac.add_argument('-l',
                            '--label-map-path',
                            required=True,
                            help='Path to label map dseg.tsv')

    parser_warp = subparsers.add_parser('warp-pet-atlas',
                                        help='Perform nonlinear warp on PET to atlas.')
    _add_common_args(parser_warp)
    parser_warp.add_argument('-a',
                             '--anatomical',
                             required=True,
                             help='Path to 3D anatomical image (T1w or T2w).',
                            type=str)
    parser_warp.add_argument('-r',
                             '--reference-atlas',
                             required=True,
                             help='Path to anatomical atlas.',
                             type=str)

    parser_suvr = subparsers.add_parser('suvr',help='Compute SUVR on a parametric PET image.')
    _add_common_args(parser_suvr)
    parser_suvr.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')
    parser_suvr.add_argument('-r',
                             '--ref-region',
                             help='Reference region to normalize SUVR to.',
                             required=True,
                             type=int)

    parser_blur = subparsers.add_parser('gauss-blur',help='Perform 3D gaussian blurring.')
    _add_common_args(parser_blur)
    parser_blur.add_argument('-b',
                             '--blur-size-mm',
                             help='Size of gaussian kernal with which to blur image.',
                             type=float)

    parser_rescale = subparsers.add_parser('rescale-image',help='Divide an image by a scalar.')
    _add_common_args(parser_rescale)
    parser_rescale.add_argument('-r',
                                '--scale-factor',
                                help='Divides image by this number',
                                type=float,
                                required=True)


    parser_window_moco = subparsers.add_parser('window-motion-corr',
                                               help='Windowed motion correction for 4D PET'
                                                    ' using ANTS')
    _add_common_args(parser_window_moco)
    parser_window_moco.add_argument('-t',
                                    '--motion-target',
                                    default='weighted_series_sum',
                                    type=str,
                                    help="Motion target option. Can be an image path , "
                                         "'weighted_series_sum' or 'mean_image'")
    parser_window_moco.add_argument('-w', '--window-size', default=60.0, type=float,
                                    help="Window size in seconds.",)
    xfm_types = ['QuickRigid', 'Rigid', 'DenseRigid', 'Affine', 'AffineFast']
    parser_window_moco.add_argument('-y', '--transform-type', default='QuickRigid', type=str,
                                    choices=xfm_types,
                                    help="Type of ANTs transformation to apply when registering.")

    parser_reg = subparsers.add_parser('register-pet',
                                       help='Register 4D PET to MRI anatomical space.')
    _add_common_args(parser_reg)
    parser_reg.add_argument('-a',
                            '--anatomical',
                            required=True,
                            help='Path to 3D anatomical image (T1w or T2w).',
                            type=str)
    parser_reg.add_argument('-t', '--motion-target', default=None, nargs='+',
                            help="Motion target option. Can be an image path, "
                                 "'weighted_series_sum' or a tuple (i.e. '-t 0 600' for first "
                                 "ten minutes).")
    parser_reg.add_argument('-l', '--half-life', help='Half life of radioisotope in seconds.',
                            type=float)

    return parser


def main():
    """
    Preproc command line interface
    """
    preproc_parser = _generate_args()
    args = preproc_parser.parse_args()

    if args.command is None:
        preproc_parser.print_help()
        raise SystemExit('Exiting without command')


    command = str(args.command).replace('-','_')

    if command=='weighted_series_sum':
        useful_functions.weighted_series_sum(input_image_4d_path=args.input_img,
                                             out_image_path=args.out_img,
                                             half_life=args.half_life,
                                             start_time=args.start_time,
                                             end_time=args.end_time,
                                             verbose=True)

    if command=='auto_crop':
        image_operations_4d.SimpleAutoImageCropper(input_image_path=args.input_img,
                                                   out_image_path=args.out_img,
                                                   thresh_val=args.thresh_val,
                                                   verbose=True)

    if command=='motion_correction':
        if len(args.motion_target)==1:
            motion_target = args.motion_target[0]
        else:
            motion_target = args.motion_target
        motion_corr.motion_corr(input_image_4d_path=args.input_img,
                                out_image_path=args.out_img,
                                motion_target_option=motion_target,
                                verbose=True,
                                type_of_transform=args.transform_type,
                                half_life=args.half_life)

    if command=='register_pet':
        register.register_pet(input_reg_image_path=args.input_img,
                              out_image_path=args.out_img,
                              reference_image_path=args.anatomical,
                              motion_target_option=args.motion_target,
                              verbose=True,
                              half_life=args.half_life)

    if command=='write_tacs':
        petpal.preproc.regional_tac_extraction.write_tacs(input_image_path=args.input_img,
                                                          out_tac_dir=args.out_tac_dir,
                                                          segmentation_image_path=args.segmentation,
                                                          label_map_path=args.label_map_path,
                                                          verbose=True)

    if command=='warp_pet_atlas':
        register.warp_pet_atlas(input_image_path=args.input_img,
                                anat_image_path=args.anatomical,
                                atlas_image_path=args.reference_atlas,
                                out_image_path=args.out_img,
                                verbose=True)

    if command=='gauss_blur':
        image_operations_4d.gauss_blur(input_image_path=args.input_img,
                                       blur_size_mm=args.blur_size_mm,
                                       out_image_path=args.out_img,
                                       verbose=True,
                                       use_fwhm=True)

    if command=='suvr':
        image_operations_4d.suvr(input_image_path=args.input_img,
                                 out_image_path=args.out_img,
                                 segmentation_image_path=args.segmentation,
                                 ref_region=args.ref_region)

    if command=='window_motion_corr':
        motion_corr.windowed_motion_corr_to_target(input_image_path=args.input_img,
                                                   out_image_path=args.out_img,
                                                   motion_target_option=args.motion_target,
                                                   w_size=args.window_size,
                                                   type_of_transform=args.transform_type)

    if command=='rescale_image':
        input_img = ants.image_read(filename=args.input_img)
        out_img = image_operations_4d.rescale_image(input_image=input_img,
                                                    rescale_constant=args.scale_factor)
        ants.image_write(image=out_img, filename=args.out_img)

if __name__ == "__main__":
    main()
