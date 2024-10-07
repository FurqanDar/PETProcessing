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
    * Half-life Weighted Sum:
    
        .. code-block:: bash
    
            petpal-preproc weighted-sum --out-dir /path/to/output --prefix sub_001 --pet /path/to/pet.nii --half-life 6586.26
    
    * Image Registration:
    
        .. code-block:: bash
    
            petpal-preproc register-pet --out-dir /path/to/output --prefix sub_001 --pet /path/to/pet.nii --anatomical /path/to/mri.nii --motion-target /path/to/pet/reference.nii
            
    * Motion Correction:
    
        .. code-block:: bash
            
            petpal-preproc motion-corr --out-dir /path/to/output --prefix sub_001 --pet /path/to/pet.nii --pet-reference /path/to/sum.nii
            
    * Extracting TACs Using A Mask And Color-Table:
    
        .. code-block:: bash
            
            petpal-preproc write-tacs --out-dir /path/to/output --pet /path/to/pet.nii --segmentation /path/to/seg_masks.nii --label-map-path /path/to/dseg.tsv

See Also:
    * :mod:`petpal.image_operations_4d` - module used for operations on 4D images.
    * :mod:`petpal.motion_corr` - module for motion correction tools.
    * :mod:`petpal.register` - module for MRI and atlas registration.
    * :mod:`petpal.preproc` - module to implement preprocessing tools.

"""
import os
import argparse
from ..preproc import preproc


_PREPROC_EXAMPLES_ = (r"""
Examples:
  - Weighted Sum:
    petpal-preproc weighted-sum --out-dir /path/to/output --prefix sub_001 --pet /path/to/pet.nii --half-life 6586.26
  - Registration:
    petpal-preproc register-pet --out-dir /path/to/output --prefix sub_001 --pet /path/to/pet.nii --anatomical /path/to/mri.nii --motion-target /path/to/pet/reference.nii
  - Motion Correction:
    petpal-preproc motion-corr --out-dir /path/to/output --prefix sub_001 --pet /path/to/pet.nii --pet-reference /path/to/sum.nii
  - Writing TACs From Segmentation Masks:
    petpal-preproc write-tacs --out-dir /path/to/output --pet /path/to/pet.nii --segmentation /path/to/seg_masks.nii --label-map-path /path/to/dseg.tsv
""")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds common arguments ('--pet', '--out-dir', and '--prefix') to a provided ArgumentParser object.

    This function modifies the passed ArgumentParser object by adding three arguments commonly used in the script.
    It uses the add_argument method of the ArgumentParser class. After running this function, the parser will
    be able to accept and parse these additional arguments from the command line when run.

    .. note::
        This function modifies the passed `parser` object in-place and does not return anything.

    Args:
        parser (argparse.ArgumentParser): The argument parser object to which the arguments are added.

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
    parser.add_argument('-o', '--out-dir', default='./', help='Output directory')
    parser.add_argument('-f', '--prefix', default="sub_XXXX", help='Output file prefix')
    parser.add_argument('-p', '--pet',required=True,help='Path to PET image.',type=str)
    parser.add_argument('-v', '--verbose', action='store_true',
                            help='Print processing information during computation.', required=False)


def _generate_args() -> argparse.Namespace:
    """
    Generates command line arguments for method :func:`main`.

    Returns:
        args (argparse.Namespace): Arguments used in the command line and their corresponding values.
    """
    parser = argparse.ArgumentParser(prog='petpal-preproc',
                                     description='Command line interface for running PET pre-processing steps.',
                                     epilog=_PREPROC_EXAMPLES_, formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help.")

    parser_wss = subparsers.add_parser('weighted-series-sum', help='Half-life weighted sum of 4D PET series.')
    _add_common_args(parser_wss)
    parser_wss.add_argument('-l', '--half-life', required=True, help='Half life of radioisotope in seconds.',
                            type=float)

    parser_reg = subparsers.add_parser('register-pet', help='Register 4D PET to MRI anatomical space.')
    _add_common_args(parser_reg)
    parser_reg.add_argument('-a', '--anatomical', required=True, help='Path to 3D anatomical image (T1w or T2w).',
                            type=str)
    parser_reg.add_argument('-t', '--motion-target', default=None, nargs='+',
                            help="Motion target option. Can be an image path, "
                                 "'weighted_series_sum' or a tuple (i.e. '-t 0 600' for first ten minutes).")
    parser_reg.add_argument('-l', '--half-life', help='Half life of radioisotope in seconds.',
                            type=float)

    parser_moco = subparsers.add_parser('motion-corr', help='Motion correction for 4D PET using ANTS')
    _add_common_args(parser_moco)
    parser_moco.add_argument('-t', '--motion-target', default=None, nargs='+',
                            help="Motion target option. Can be an image path, "
                                 "'weighted_series_sum' or a tuple (i.e. '-t 0 600' for first ten minutes).")
    parser_moco.add_argument('-l', '--half-life', help='Half life of radioisotope in seconds.',
                            type=float)

    parser_tac = subparsers.add_parser('write-tacs', help='Write ROI TACs from 4D PET using segmentation masks.')
    _add_common_args(parser_tac)
    parser_tac.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')
    parser_tac.add_argument('-l', '--label-map-path', required=True, help='Path to label map dseg.tsv')
    parser_tac.add_argument('-k', '--time-frame-keyword', required=False, help='Time keyword used for frame timing',default='FrameReferenceTime')

    parser_warp = subparsers.add_parser('warp-pet-atlas',help='Perform nonlinear warp on PET to atlas.')
    _add_common_args(parser_warp)
    parser_warp.add_argument('-a', '--anatomical', required=True, help='Path to 3D anatomical image (T1w or T2w).',
                            type=str)
    parser_warp.add_argument('-r','--reference-atlas',required=True,help='Path to anatomical atlas.',type=str)

    parser_res = subparsers.add_parser('resample-segmentation',help='Resample segmentation image to PET resolution.')
    _add_common_args(parser_res)
    parser_res.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')

    parser_suvr = subparsers.add_parser('suvr',help='Compute SUVR on a parametric PET image.')
    _add_common_args(parser_suvr)
    parser_suvr.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')
    parser_suvr.add_argument('-r','--ref-region',help='Reference region to normalize SUVR to.',required=True)

    parser_blur = subparsers.add_parser('gauss-blur',help='Perform 3D gaussian blurring.')
    _add_common_args(parser_blur)
    parser_blur.add_argument('-b','--blur-size-mm',help='Size of gaussian kernal with which to blur image.')

    return parser


def main():
    """
    Preprocessing command line interface
    """
    preproc_parser = _generate_args()
    args = preproc_parser.parse_args()

    if args.command is None:
        preproc_parser.print_help()
        raise Exception('Exiting without command')

    subject = preproc.PreProc(output_directory=os.path.abspath(args.out_dir),
                              output_filename_prefix=args.prefix)

    preproc_props = {
        'FilePathWSSInput': args.pet,
        'FilePathMocoInp': args.pet,
        'FilePathRegInp': args.pet,
        'FilePathTACInput': args.pet,
        'FilePathWarpInput': args.pet,
        'FilePathSUVRInput': args.pet,
        'FilePathBlurInput': args.pet,
        'Verbose': args.verbose
    }

    if 'anatomical' in args.__dict__.keys():
        preproc_props['FilePathAnat'] = args.anatomical
    if 'segmentation' in args.__dict__.keys():
        preproc_props['FilePathSeg'] = args.segmentation
    if 'label_map_path' in args.__dict__.keys():
        preproc_props['FilePathLabelMap'] = args.label_map_path
    if 'reference_atlas' in args.__dict__.keys():
        preproc_props['FilePathAtlas'] = args.reference_atlas
    if 'half_life' in args.__dict__.keys():
        preproc_props['HalfLife'] = args.half_life
    if 'motion_target' in args.__dict__.keys():
        if len(args.motion_target)==1:
            preproc_props['MotionTarget'] = args.motion_target[0]
        else:
            preproc_props['MotionTarget'] = args.motion_target
    if 'blur_size_mm' in args.__dict__.keys():
        preproc_props['BlurSize'] = args.blur_size_mm
    if 'time_frame_keyword' in args.__dict__.keys():
        preproc_props['TimeFrameKeyword'] = args.time_frame_keyword
    if 'ref-region' in args.__dict__.keys():
        preproc_props['RefRegion'] = args.ref_region

    command = str(args.command).replace('-','_')

    if args.verbose:
        print(f"Running {command} with parameters: {preproc_props}")

    subject.update_props(new_preproc_props=preproc_props)
    subject.run_preproc(method_name=command)


if __name__ == "__main__":
    main()
