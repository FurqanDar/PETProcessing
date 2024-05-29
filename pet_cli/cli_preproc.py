"""
This module provides a Command-line interface (CLI) for preprocessing imaging data to
produce regional PET Time-Activity Curves (TACs) and prepare data for parametric imaging analysis.

The user must provide:
    * The sub-command. Options: 'weighted-sum', 'motion-correct', 'register', or 'write-tacs'.
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
    
            pet-cli-preproc weighted-sum --pet /path/to/pet.nii --out-dir /path/to/output --half-life 6600.0
    
    * Image Registration:
    
        .. code-block:: bash
    
            pet-cli-preproc register --pet /path/to/pet.nii --anatomical /path/to/mri.nii --pet-reference /path/to/pet_sum.nii --out-dir /path/to/output
            
    * Motion Correction:
    
        .. code-block:: bash
            
            pet-cli-preproc motion-correct --pet /path/to/pet.nii --pet-reference /path/to/sum.nii --out-dir /path/to/output
            
    * Extracting TACs Using A Mask And Color-Table:
    
        .. code-block:: bash
            
            pet-cli-preproc write-tacs --pet /path/to/pet.nii --segmentation /path/to/seg_masks.nii --color-table-path /path/to/color_table.json --out-dir /path/to/output

See Also:
    * :mod:`pet_cli.image_operations_4d` - module used to preprocess PET imaging data.

"""
import os
import argparse
from . import image_operations_4d, preproc


_PREPROC_EXAMPLES_ = (r"""
Examples:
  - Weighted Sum:
    pet-cli-preproc weighted-sum --pet /path/to/pet.nii --out-dir /path/to/output --half-life 6600.0
  - Registration:
    pet-cli-preproc register --pet /path/to/pet.nii --anatomical /path/to/mri.nii --pet-reference /path/to/pet_sum.nii --out-dir /path/to/output
  - Motion Correction:
    pet-cli-preproc motion-correct --pet /path/to/pet.nii --pet-reference /path/to/sum.nii --out-dir /path/to/output
  - Writing TACs From Segmentation Masks:
    pet-cli-preproc write-tacs --pet /path/to/pet.nii --segmentation /path/to/seg_masks.nii --color-table-path /path/to/color_table.json --out-dir /path/to/output
  - Verbose:
    pet-cli-preproc -v [sub-command] [arguments]
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


def _generate_args() -> argparse.Namespace:
    """
    Generates command line arguments for method :func:`main`.

    Returns:
        args (argparse.Namespace): Arguments used in the command line and their corresponding values.
    """
    parser = argparse.ArgumentParser(prog='pet-cli-preproc',
                                     description='Command line interface for running PET pre-processing steps.',
                                     epilog=_PREPROC_EXAMPLES_, formatter_class=argparse.RawTextHelpFormatter)
    
    # create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help.")

    # create parser for "weighted-sum" command
    parser_wss = subparsers.add_parser('weighted-series-sum', help='Half-life weighted sum of 4D PET series.')
    _add_common_args(parser_wss)
    parser_wss.add_argument('-l', '--half-life', required=True, help='Half life of radioisotope in seconds.',
                            type=float)

    # create parser for "register" command
    parser_reg = subparsers.add_parser('register-pet', help='Register 4D PET to MRI anatomical space.')
    _add_common_args(parser_reg)
    parser_reg.add_argument('-a', '--anatomical', required=True, help='Path to 3D anatomical image (T1w or T2w).',
                            type=str)
    parser_reg.add_argument('-t', '--motion-target', default=None,
                            help='Motion target option. Can be an image path, or a tuple. See (ref).') # TODO: fix reference

    # create parser for the "motion-correct" command
    parser_moco = subparsers.add_parser('motion-corr', help='Motion correction for 4D PET using ANTS')
    _add_common_args(parser_moco)
    parser_moco.add_argument('-t', '--motion-target', default=None,
                            help='Motion target option. Can be an image path, or a tuple. See (ref).') # TODO: fix reference

    # create parser for the "write-tacs" command
    parser_tac = subparsers.add_parser('write-tacs', help='Write ROI TACs from 4D PET using segmentation masks.')
    _add_common_args(parser_tac)
    parser_tac.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')
    parser_tac.add_argument('-l', '--label-map-path', required=True, help='Path to label map dseg.tsv')
    parser_tac.add_argument('-k', '--time-frame-keyword', required=False, help='Time keyword used for frame timing',default='FrameTimeReference')

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


    verb_group = parser.add_argument_group('Additional information')
    verb_group.add_argument('-v', '--verbose', action='store_true',
                            help='Print processing information during computation.', required=False)

    args = parser.parse_args()
    arg_help = parser.print_help()
    return args, arg_help


def main():
    """
    Preprocessing command line interface
    """
    args, arg_help = _generate_args()

    if args.command is None:
        print(arg_help)
        return 0

    subject = preproc.PreProc(output_directory=os.path.abspath(args.out_dir),
                              output_filename_prefix=args.prefix)
    preproc_props = {
        'FilePathWSSInput': args.pet,
        'FilePathMocoInp': args.pet,
        'FilePathRegInp': args.pet,
        'FilePathAnat': args.anatomical,
        'FilePathTACInput': args.pet,
        'FilePathSeg': args.segmentation,
        'FilePathLabelMap': args.label_map_path,
        'FilePathWarpInput': args.pet,
        'FilePathAtlas': args.reference_atlas,
        'FilePathSUVRInput': args.pet,
        'FilePathBlurInput': args.pet,
        'HalfLife': args.half_life,
        'MotionTarget': args.motion_target,
        'BlurSize': args.blur_size_mm,
        'TimeFrameKeyword': args.time_frame_keyword,
        'Verbose': args.verbose
    }
    subject.update_props(new_preproc_props=preproc_props)
    subject.run_preproc(method_name=args.command)


if __name__ == "__main__":
    main()
