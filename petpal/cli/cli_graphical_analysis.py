"""
Command-line interface (CLI) for conducting graphical analysis of PET Time-Activity Curves (TACs).

This module provides a CLI to interact with the graphical_analysis module. It leverages argparse to handle command-line
arguments.

The user must provide:
    * Input TAC file path
    * Region of Interest (ROI) TAC file path
    * Threshold in minutes (below which data points will not be considered for fitting)
    * The method name for conducting the analysis. Supported methods are 'patlak', 'logan', or 'alt-logan'.
    * Output directory where the analysis results will be saved

An optional filename prefix for the output files can also be supplied.

This script utilizes the :class:`petpal.kinetic_modeling.graphical_analysis.GraphicalAnalysis` class to perform the graphical analysis
and save the results accordingly.

Example:
    .. code-block:: bash
    
        petpal-graph-analysis --input-tac-path /path/to/input.tac --roi-tac-path /path/to/roi.tac --threshold-in-mins 30.0 --method-name patlak --output-directory ./analysis --output-filename-prefix analysis

See Also:
    :mod:`petpal.kinetic_modeling.graphical_analysis` - module responsible for conducting and saving graphical analysis of PET TACs.

"""

import argparse
from ..kinetic_modeling import graphical_analysis as pet_ga


def main():
    parser = argparse.ArgumentParser(prog="Graphical Analysis", description="Perform graphical analysis on TAC data.",
                                     epilog="Example: petpal-graph-analysis "
                                            "--input-tac-path /path/to/input.tac "
                                            "--pet4D-img-path /path/to/pet4D.img "
                                            "--output-directory /path/to/output "
                                            "--output-filename-prefix graph_ana"
                                            "--method-name patlak --threshold-in-mins 30.0 ")
    
    # IO group
    grp_io = parser.add_argument_group('IO Paths and Prefixes')
    grp_io.add_argument("-i", "--input-tac-path", required=True, help="Path to the input TAC file.")
    grp_io.add_argument("-r", "--roi-tac-path", required=True, help="Path to the ROI TAC file.")
    grp_io.add_argument("-o", "--output-directory", required=True, help="Path to the output directory.")
    grp_io.add_argument("-p", "--output-filename-prefix", required=True, help="Prefix for the output filenames.")
    
    # Analysis group
    grp_analysis = parser.add_argument_group('Analysis Parameters')
    grp_analysis.add_argument("-t", "--threshold-in-mins", required=True, type=float,
                           help="Threshold in minutes for the analysis.")
    grp_analysis.add_argument("-m", "--method-name", required=True, choices=['patlak', 'logan', 'alt-logan'],
                           help="Analysis method to be used.")
    
    # Additional group arguments
    grp_verbose = parser.add_argument_group('Additional Options')
    grp_verbose.add_argument("--print", action="store_true", help="Whether to print the analysis results.")
    
    args = parser.parse_args()
    
    graphical_analysis = pet_ga.GraphicalAnalysis(input_tac_path=args.input_tac_path,
                                                  roi_tac_path=args.roi_tac_path,
                                                  output_directory=args.output_directory,
                                                  output_filename_prefix=args.output_filename_prefix)
    
    graphical_analysis.run_analysis(method_name=args.method_name, t_thresh_in_mins=args.threshold_in_mins)
    graphical_analysis.save_analysis()
    
    if args.print:
        for key, val in graphical_analysis.analysis_props.items():
            print(f"{key:<20}:  {val}")
    

if __name__ == "__main__":
    main()