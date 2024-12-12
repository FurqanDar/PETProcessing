r"""
Command-line interface (CLI) for fitting Tissue Compartment Models (TCM) to PET Time-Activity Curves (TACs).

This module provides a CLI to interact with the tac_fitting module. It utilizes argparse to handle command-line
arguments.

The user must provide:
    * Input TAC file path
    * Region of Interest (ROI) TAC file path
    * Compartment model name for fitting. Supported models are '1tcm', '2tcm-k4zero', or 'serial-2tcm'.
    * Filename prefix for the output files
    * Output directory where the analysis results will be saved
    * Whether to ignore blood volume contributions while fitting
    * Threshold in minutes (input fitting threshold)

User can optionally provide:
    * Initial guesses for the fitting parameters
    * Lower and upper bounds for the fitting parameters
    * Maximum number of function iterations
    * Decay constant (in minutes) for per-frame weighting

This script utilizes the :class:`TCMAnalysis<petpal.tac_fitting.TCMAnalysis>` class to perform the TAC fitting and save
 the results accordingly.

Example:
    In the proceeding example, we assume that we have an input TAC named 'input_tac.txt', and an ROI TAC named
    'roi_tac.txt`. We want to try fitting a serial 2TCM to the ROI tac.
    
    .. code-block:: bash

        petpal-tcm-fit -i "input_tac.txt"\
        -r "roi_tac.txt"\
        -m "serial-2tcm"\
        -o "./" -p "cli_"\
        -t 35.0 -w 0.0063
        -g 0.1 0.1 0.1 0.1 0.1\
        -l 0.0 0.0 0.0 0.0 0.0\
        -u 5.0 5.0 5.0 5.0 5.0\
        -f 1000 -n 512 -b --print

See Also:
    :mod:`petpal.tac_fitting` - module for fitting TACs with TCMs.

"""

from typing import Union
import numpy as np
import argparse
from ..kinetic_modeling import tac_fitting as pet_fit

_EXAMPLE_ = ('Fitting a TAC to the serial 2TCM using the F18 decay constant (lambda=ln(2)/t_half_in_mins):\n\t'
             'petpal-tcm-fit -i "input_tac.txt"'
             ' -r "2tcm_tac.txt" '
             '-m "serial-2tcm" '
             '-o "./" -p "cli_" -t 35.0 '
             '-w 0.0063 '
             '-g 0.1 0.1 0.1 0.1 0.1 '
             '-l 0.0 0.0 0.0 0.0 0.0 '
             '-u 5.0 5.0 5.0 5.0 5.0 '
             '-f 1000 -n 512 -b '
             '--print')


def _generate_args() -> argparse.Namespace:
    r"""
    Generates and handles the arguments for the command-line interface.

    This function sets up the argument parser, adds required and optional arguments, and parses input arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Raises:
        argparse.ArgumentError: If necessary arguments are missing or invalid arguments are provided.
    """
    parser = argparse.ArgumentParser(prog='petpal-tcm-fit',
                                     description='Command line interface for fitting Tissue Compartment Models (TCM) '
                                                 'to PET Time Activity Curves (TACs).',
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=_EXAMPLE_)
    
    # IO group
    grp_io = parser.add_argument_group('IO Paths and Prefixes')
    grp_io.add_argument("-i", "--input-tac-path", required=True, help="Path to the input TAC file.")
    grp_io.add_argument("-r", "--roi-tac-path", required=True, help="Path to the ROI TAC file.")
    grp_io.add_argument("-o", "--output-directory", required=True, help="Path to the output directory.")
    grp_io.add_argument("-p", "--output-filename-prefix", required=True, help="Prefix for the output filenames.")
    
    # Analysis group
    grp_analysis = parser.add_argument_group('Analysis Parameters')
    grp_analysis.add_argument("-m", "--model", required=True, choices=['1tcm', '2tcm-k4zero', 'serial-2tcm'],
                              help="Analysis method to be used.")
    grp_analysis.add_argument("-t", "--input-fitting-threshold-in-mins", required=True, type=float,
                              help="Threshold in minutes for fitting the later half of the input function.")
    grp_analysis.add_argument("-b", "--ignore-blood-volume", required=False, default=False, action='store_true',
                              help="Whether to ignore any blood volume contributions while fitting.")
    grp_analysis.add_argument("-g", "--initial-guesses", required=False, nargs='+', type=float,
                              help="Initial guesses for each fitting parameter.")
    grp_analysis.add_argument("-l", "--lower-bounds", required=False, nargs='+', type=float,
                              help="Lower bounds for each fitting parameter.")
    grp_analysis.add_argument("-w", "--weighting-decay-constant", required=False, type=float, default=None,
                              help="Decay constant for computing per-frame weighting for fits.")
    grp_analysis.add_argument("-u", "--upper-bounds", required=False, nargs='+', type=float,
                              help="Upper bounds for each fitting parameter.")
    grp_analysis.add_argument("-f", "--max-fit-iterations", required=False, default=2500, type=int,
                              help="Maximum number of function iterations")
    grp_analysis.add_argument("-n", "--resample-num", required=False, default=512, type=int,
                              help="Number of samples for linear interpolation of provided TACs.")
    
    # Printing arguments
    grp_verbose = parser.add_argument_group('Additional Options')
    grp_verbose.add_argument("--print", action="store_true", help="Whether to print the analysis results.")
    
    return parser.parse_args()


def _generate_bounds(initial: Union[list, None],
                     lower: Union[list, None],
                     upper: Union[list, None]) -> Union[np.ndarray, None]:
    r"""
    Generates the bounds for the fitting parameters.

    This function takes lists of initial fitting parameters, lower bounds, and upper bounds.
    All lists must have the same length. If no initial parameters are provided, the function returns None.

    Args:
        initial (list, optional): List of initial guesses for fitting parameters. If None, no bounds are generated.
        lower (list, optional): List of lower bounds for fitting parameters.
        upper (list, optional): List of upper bounds for fitting parameters.

    Returns:
        (np.ndarray, optional): If initial is not None, then we return a numpy array of shape [n, 3],
        where n is the number of parameters, where column 0 has the initial guesses, column 1 has lower bounds, and
        column 2 has upper bounds. If initial is None, function will return None.

    Raises:
        ValueError: If initial is not None and the length of initial, lower, and upper are not the same.
    """
    if initial is not None:
        if (len(initial) != len(lower)) or (len(initial) != len(upper)) or (len(upper) != len(lower)):
            raise ValueError("The number of initial guesses, lower bounds and upper bounds must be the same.")
        return np.asarray(np.asarray([initial, lower, upper]).T)
    else:
        return None


def main():
    args = _generate_args()
    
    bounds = _generate_bounds(initial=args.initial_guesses, lower=args.lower_bounds, upper=args.upper_bounds)
    
    tac_fitting = pet_fit.TCMAnalysis(input_tac_path=args.input_tac_path,
                                      roi_tac_path=args.roi_tac_path,
                                      output_directory=args.output_directory,
                                      output_filename_prefix=args.output_filename_prefix,
                                      compartment_model=args.model,
                                      parameter_bounds=bounds, weights=None,
                                      resample_num=args.resample_num,
                                      aif_fit_thresh_in_mins=args.input_fitting_threshold_in_mins,
                                      max_func_iters=args.max_fit_iterations,
                                      ignore_blood_volume=args.ignore_blood_volume)
    tac_fitting.run_analysis()
    tac_fitting.save_analysis()
    
    if args.print:
        title_str = f"{'Param':<5} {'FitVal':<6}    {'StdErr':<6} ({'%Err':>6})|"
        print("-" * len(title_str))
        print(title_str)
        print("-" * len(title_str))
        vals = tac_fitting.analysis_props["FitProperties"]["FitValues"]
        errs = tac_fitting.analysis_props["FitProperties"]["FitStdErr"]
        for param_name in vals:
            val = vals[param_name]
            err = errs[param_name]
            print(f"{param_name:<5} {val:<6.4f} +- {err:<6.4f} ({err / val * 100:>5.2f}%)|")
        print("-" * len(title_str))


if __name__ == "__main__":
    main()
