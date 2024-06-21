"""
Command-line interface (CLI) for generating graphical analysis plots of PET Time-Activity Curves (TACs).

This module provides a CLI to work with the graphical_plots module. It uses argparse to handle command-line arguments.

The user must provide:
    * Input TAC file path
    * Region of Interest (ROI) TAC file path
    * Threshold in minutes (below which data points will be not be fit)
    * The method name for generating the plot. Supported methods are 'patlak', 'logan', or 'alt-logan'.
    * Output directory where the plots will be saved

An optional filename prefix for the output files can also be supplied.

This script uses the 'Plot' class from the 'graphical_plots' module to generate and save the plots.

Example:
    .. code-block:: bash
    
        petpal-graph-plot --input-tac-path /path/to/input.tac --roi-tac-path /path/to/roi.tac --threshold-in-mins 30.0 --method-name patlak --output-directory ./plots --output-filename-prefix plot

See Also:
    :mod:`petpal.visualization.graphical_plots` - module for creating and saving graphical analysis plots of PET TACs.
    
"""

import argparse
from ..visualizations import graphical_plots as pet_plt


def main():
    parser = argparse.ArgumentParser(prog="Graphical Analysis Plots",
                                     description="Generate graphical analysis plots of PET TACs.",
                                     epilog="Example: petpal-graph-plot "
                                            "--input-tac-path /path/to/input.tac --roi-tac-path "
                                            "/path/to/roi.tac --threshold-in-mins 30.0 --method-name patlak "
                                            "--output-directory /path/to/output --output-filename-prefix plot")
    
    grp_io = parser.add_argument_group("I/O Paths And Prefixes")
    grp_io.add_argument("-i", "--input-tac-path", required=True, help="Path to the input TAC file.")
    grp_io.add_argument("-r", "--roi-tac-path", required=True, help="Path to the Region of Interest (ROI) TAC file.")
    grp_io.add_argument("-o", "--output-directory", required=True,
                        help="Path to the directory where the plot output will be saved.")
    grp_io.add_argument("-p", "--output-filename-prefix", default="",
                        help="An optional prefix for the output filenames.")
    
    grp_params = parser.add_argument_group("Method Parameters")
    grp_params.add_argument("-t", "--threshold-in-mins", required=True, type=float,
                        help="Threshold in minutes below which data points will be discarded.")
    grp_params.add_argument("-m", "--method-name", required=True, choices=['patlak', 'logan', 'alt-logan'],
                        help="Name of the method for generating the plot.")
    
    args = parser.parse_args()
    
    grph_plot = pet_plt.Plot(input_tac_path=args.input_tac_path, roi_tac_path=args.roi_tac_path,
                             threshold_in_mins=args.threshold_in_mins, method_name=args.method_name,
                             output_directory=args.output_directory, output_filename_prefix=args.output_filename_prefix)
    
    grph_plot.save_figure()


if __name__ == "__main__":
    main()