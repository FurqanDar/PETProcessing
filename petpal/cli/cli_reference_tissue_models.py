r"""
This module provides a command-line interface for running various reference tissue model (RTM) analyses.

The module defines a set of command-line arguments for the input and output data, as well
as the analysis specifics. Several types of RTM analyses, such as SRTM, FRTM, MRTM and its variants,
are supported, each with its associated arguments.

The primary function in the module is `main()`, which is responsible for parsing command-line
arguments, running the selected analysis, and handling the results. The analysis is performed
using the :class:`RTMAnalysis<petpal.reference_tissue_models.RTMAnalysis>` class.

Example:
    Assuming that the package has been installed, we have:
    
    For running an FRTM analysis:
    
    .. code-block:: bash
    
        petpal-rtms frtm --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --initial-guesses 0.1 0.1 0.1 0.1 0.1 --lower-bounds 0.0 0.0 0.0 0.0 0.0 --upper-bounds 5.0 5.0 5.0 5.0 5.0
    
    For running an SRTM analysis:
    
    .. code-block:: bash
    
        petpal-rtms srtm --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --initial-guesses 0.1 0.1 0.1 0.1 0.1 --lower-bounds 0.0 0.0 0.0 0.0 0.0 --upper-bounds 5.0 5.0 5.0 5.0 5.0
        
    For running an FRTM2 analysis:
    
    .. code-block:: bash
    
        petpal-rtms frtm2 --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --k2-prime 0.5 --prefix sub_001 --print --initial-guesses 0.1 0.1 0.1 --lower-bounds 0.0 0.0 0.0 --upper-bounds 5.0 5.0 5.0
    
    For running an SRTM2 analysis:
    
    .. code-block:: bash
    
        petpal-rtms srtm2 --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --k2-prime 0.5 --prefix sub_001 --print --initial-guesses 0.1 0.1 --lower-bounds 0.0 0.0 0.0 --upper-bounds 5.0 5.0
    
    For running an MRTM analysis:
    
    .. code-block:: bash
    
        petpal-rtms mrtm --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --t-thresh-in-mins 30.0
    
    For running an MRTM2 analysis:
    
    .. code-block:: bash
    
        petpal-rtms mrtm2 --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --t-thresh-in-mins 30.0 --k2-prime 0.1
    
    
    For running the original MRTM analysis:
    
    .. code-block:: bash
    
        petpal-rtms mrtm-original --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --t-thresh-in-mins 30.0
    
    
    

See Also:
    * :mod:`petpal.reference_tissue_models`
    
"""

import argparse
import numpy as np
from typing import Union
from ..kinetic_modeling.rtm_analysis import RTMAnalysis


_RTM_EXAMPLES_ = (r"""
Examples:
  - FRTM (4 parameters: R1, k2, k3, k4):
    petpal-rtms frtm --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --initial-guesses 0.1 0.1 0.1 0.1 --lower-bounds 0.0 0.0 0.0 0.0 --upper-bounds 5.0 5.0 5.0 5.0
  - SRTM (3 parameters: R1, k2, BP):
    petpal-rtms srtm --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --initial-guesses 0.1 0.1 0.1 --lower-bounds 0.0 0.0 0.0 --upper-bounds 5.0 5.0 5.0
  - FRTM2 (3 parameters: R1, k3, k4):
    petpal-rtms frtm2 --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --k2-prime 0.5 --prefix sub_001 --print --initial-guesses 0.1 0.1 0.1 0.1 --lower-bounds 0.0 0.0 0.0 0.0 --upper-bounds 5.0 5.0 5.0 5.0
  - SRTM2 (2 parameters: R1, BP):
    petpal-rtms srtm2 --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --k2-prime 0.5 --prefix sub_001 --print --initial-guesses 0.1 0.1 0.1 0.1 --lower-bounds 0.0 0.0 0.0 --upper-bounds 5.0 5.0
  - MRTM (2 parameters: BP, k2_prime):
    petpal-rtms mrtm --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --t-thresh-in-mins 30.0
  - MRTM2 (2 parameters: BP):
    petpal-rtms mrtm2 --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --t-thresh-in-mins 30.0 --k2-prime 0.1
  - MRTM (Original) (2 parameters: BP, k2_prime):
    petpal-rtms mrtm-original --ref-tac-path /path/to/ref/tac --roi-tac-path /path/to/roi/tac --prefix sub_001 --print --t-thresh-in-mins 30.0
""")


def add_common_args(parser):
    r"""
    Adds common arguments to an ArgumentParser object.

    This function modifies the passed ArgumentParser object by adding five arguments commonly used for the command line
    scripts. It uses the add_argument method of the ArgumentParser class. After running this function, the parser will
    be able to accept and parse these additional arguments from the command line when run.

    Args:
        parser (argparse.ArgumentParser): The argument parser object to which the arguments will be added.

    Raises:
        argparse.ArgumentError: If a duplicate argument is added.

    Side Effects:
        The ArgumentParser object is modified by adding new arguments.
    """
    parser.add_argument('-e', '--ref-tac-path', required=True, type=str,
                        help='Absolute path for reference TAC')
    parser.add_argument('-i', '--roi-tac-path', required=True, type=str,
                        help='Absolute path for ROI TAC')
    parser.add_argument('-o', '--output-directory', required=True, type=str,
                        help='Absolute path for the output directory')
    parser.add_argument('-f', '--prefix', required=True, type=str,
                        help='Prefix for the output filename of the result')
    parser.add_argument('--print', required=False, action='store_true',
                        help="Whether to print the fitting results to screen.")


def parse_args():
    """
    Defines and parses command-line arguments.

    This function creates an ArgumentParser object, defines command-line arguments using that object,
    and then calls its parse_args method and returns the results.

    The function defines a set of sub-commands (srtm, frtm, mrtm, mrtm2, mrtm-original), each of which has a different
    set of requiring arguments. Each sub-command represents a different analysis method and requires different inputs.

    Returns:
        argparse.Namespace: An object representing the command-line arguments. The object's attributes are the
        various command-line arguments, where the attribute name matches the option string.

    Raises:
        argparse.ArgumentError: If the argument is already defined.
        argparse.ArgumentTypeError: If the argument type conversion function raises this exception.
        SystemExit: If invalid arguments are passed or `-h`/`--help` is chosen.

    """
    parser = argparse.ArgumentParser(prog='petpal-rtms',
                                     description='Command line interface for running RTM analyses on TACs',
                                     epilog=_RTM_EXAMPLES_,
                                     formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="method", help="Analysis method")

    parser_srtm = subparsers.add_parser('srtm', help='Perform SRTM analysis')
    parser_frtm = subparsers.add_parser('frtm', help='Perform FRTM analysis')
    parser_srtm_2 = subparsers.add_parser('srtm2', help='Perform SRTM2 analysis')
    parser_frtm_2 = subparsers.add_parser('frtm2', help='Perform FRTM2 analysis')
    
    parser_mrtm = subparsers.add_parser('mrtm',
                                        help='Perform Ichise\'s MRTM (2003) analysis')
    parser_mrtm_2 = subparsers.add_parser('mrtm2',
                                          help='Perform Ichise\'s MRTM2 (2003) analysis')
    parser_mrtm_original = subparsers.add_parser('mrtm-original',
                                                 help='Perform Ichise\'s original MRTM (1996) analysis')
    parser_list = [parser_srtm, parser_frtm, parser_srtm_2, parser_frtm_2,
                   parser_mrtm, parser_mrtm_2, parser_mrtm_original]
    
    for a_parser in parser_list[:]:
        add_common_args(a_parser)
        if a_parser.prog.endswith('2'):
            a_parser.add_argument('-k', '--k2-prime', required=True, type=float,
                                  help='k2_prime value for the reduced RTM analysis.')
            
        if 'mrtm' in a_parser.prog:
            a_parser.add_argument('-t', '--t-thresh-in-mins', required=True, type=float,
                                  help='Threshold time in minutes for the MRTM analyses.')
        else:
            a_parser.add_argument("-g", "--initial-guesses", required=False, nargs='+', type=float,
                                      help="Initial guesses for each fitting parameter.")
            a_parser.add_argument("-l", "--lower-bounds", required=False, nargs='+', type=float,
                                      help="Lower bounds for each fitting parameter.")
            a_parser.add_argument("-u", "--upper-bounds", required=False, nargs='+', type=float,
                                      help="Upper bounds for each fitting parameter.")

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
        if upper is None or lower is None:
            raise ValueError("The lower and upper bounds must also be set.")
        if (len(initial) != len(lower)) or (len(initial) != len(upper)) or (len(upper) != len(lower)):
            raise ValueError("The number of initial guesses, lower bounds and upper bounds must be the same.")
        return np.asarray(np.asarray([initial, lower, upper]).T)
    else:
        return None


def _pretty_print_results(vals: dict, errs: dict) -> None:
    r"""
    Formats and prints the analysis results to the console.

    This helper function creates a pretty, human-readable formatted string of the parameter names,
    their fitted values, standard errors, and percentage errors, and then prints this
    formatted string to the console.

    Args:
        vals (dict): A dictionary where the keys are the parameter names and the values are fitted values.
        errs (dict): A dictionary where the keys are the parameter names and the values are standard errors.

    Returns:
        None

    Side Effects:
        Prints the formatted result string to the console.

    Raises:
        KeyError: If the keys of 'vals' and 'errs' do not match.
    """
    title_str = f"{'Param':<5} {'FitVal':<6}    {'StdErr':<6} ({'%Err':>6})|"
    print("-" * len(title_str))
    print(title_str)
    print("-" * len(title_str))
    for param_name in vals:
        val = vals[param_name]
        err = errs[param_name]
        print(f"{param_name:<5} {val:<6.4f} +- {err:<6.4f} ({err / val * 100:>5.2f}%)|")
    print("-" * len(title_str))


def main():
    args = parse_args()
    
    if args.method is None:
        args.print_help()
        raise Exception('Exiting without command')
    
    analysis = RTMAnalysis(ref_tac_path=args.ref_tac_path,
                           roi_tac_path=args.roi_tac_path,
                           output_directory=args.output_directory,
                           output_filename_prefix=args.prefix,
                           method=args.method)
    
    if args.method.startswith('mrtm'):
        analysis.run_analysis(t_thresh_in_mins=args.t_thresh_in_mins,
                              k2_prime=getattr(args, 'k2_prime', None))
    else:
        bounds = _generate_bounds(initial=args.initial_guesses, lower=args.lower_bounds, upper=args.upper_bounds)
        if args.method.endswith('2'):
            analysis.run_analysis(bounds=bounds, k2_prime=getattr(args, 'k2_prime', None))
        else:
            analysis.run_analysis(bounds=bounds)
    
    analysis.save_analysis()

    if args.print:
        if args.method.startswith('mrtm'):
            print(f"BP      : {analysis.analysis_props['BP']:<.7f}")
            print(f"k2_prime: {analysis.analysis_props['k2Prime']:<.7f}")
        else:
            _pretty_print_results(vals=analysis.analysis_props["FitValues"],
                                  errs=analysis.analysis_props["FitStdErr"])
    

if __name__ == '__main__':
    main()