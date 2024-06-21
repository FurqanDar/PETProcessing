"""
Command-line interface (CLI) for generating PET parametric images using graphical analysis of Time-Activity Curves (TACs).

This module provides a CLI to work with the parametric_images module. It uses argparse to handle command-line arguments.

The user must provide:
    * Input TAC file path
    * Path to the 4D PET image file
    * Threshold in minutes (below which data points will be not be used for fitting)
    * The method name for generating the images. Supported methods are 'patlak', 'logan', or 'alt-logan'.
    * Output directory where the parametric images will be saved

An optional filename prefix for the output files can also be supplied.

This script uses the :class:'petpal.parametric_images.GraphicalAnalysisParametricImage' class to calculate and save
the images.

Example:
    .. code-block:: bash
    
         petpal-parametric-image --input-tac-path /path/to/input.tac --pet4D-img-path /path/to/pet4D.img --threshold-in-mins 30.0 --method-name patlak --output-directory ./images --output-filename-prefix image

See Also:
    :mod:`petpal.parametric_images` - module for initiating and saving the graphical analysis of PET parametric images.

"""

import argparse
from ..kinetic_modeling import parametric_images as pet_pim


def main():
    parser = argparse.ArgumentParser(prog="Parametric Images With Graphical Analyses",
                                     description="Generate parametric images using graphical analysis on PET TACs.",
                                     epilog="Example usage: petpal-parametric-image "
                                            "--input-tac-path /path/to/input.tac "
                                            "--pet4D-img-path /path/to/image4D.pet "
                                            "--output-directory /path/to/output --output-filename-prefix param_image "
                                            "--method-name patlak --threshold-in-mins 30.0")
    
    grp_io = parser.add_argument_group('I/O Paths')
    grp_io.add_argument("-i", "--input-tac-path", required=True,
                        help="Path to the input Time-Activity Curve (TAC) file.")
    grp_io.add_argument("-p", "--pet4D-img-path", required=True, help="Path to the 4D PET image file.")
    grp_io.add_argument("-o", "--output-directory", required=True,
                        help="Directory where the output parametric images will be saved.")
    grp_io.add_argument("-f", "--output-filename-prefix", default="", help="Optional prefix for the output filenames.")
    
    grp_params = parser.add_argument_group('Method Parameters')
    grp_params.add_argument("-t", "--threshold-in-mins", required=True, type=float,
                            help="Threshold in minutes below which data points will be discarded.")
    grp_params.add_argument("-m", "--method-name", required=True, choices=['patlak', 'logan', 'alt-logan'],
                            help="Name of the method for generating the plot.")
    
    args = parser.parse_args()
    
    param_img = pet_pim.GraphicalAnalysisParametricImage(input_tac_path=args.input_tac_path,
                                                         pet4D_img_path=args.pet4D_img_path,
                                                         output_directory=args.output_directory,
                                                         output_filename_prefix=args.output_filename_prefix)
    
    param_img.run_analysis(method_name=args.method_name, t_thresh_in_mins=args.threshold_in_mins)
    param_img.save_analysis()


if __name__ == "__main__":
    main()