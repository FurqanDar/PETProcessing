"""
Command-line interface (CLI) for interpolating Time-Activity Curves (TACs).

This module provides a CLI to interpolate TACs, enabling the conversion of these curves to a desired time scale. It uses
argparse to handle command-line arguments.

The user must provide:
    * Input TAC file path
    * Interpolation interval in minutes
    * Output TAC file path

This script uses functionalities from :mod:'petpal.tac_interpolation' module to interpolate the TAC based on the
specified interval, or number of samples before the maximum value, and save the interpolated TAC to a specified output
file.

An option to print the interpolated TAC to screen is also provided for visual inspection of the results.

Example usage:
    Using :class:`petpal.cli_tac_interpolation.EvenlyInterpolate`:
        .. code-block:: bash
        
            petpal-tac-interpolation --input-tac-path /path/to/input.tac --output-tac-path /path/to/output.tac --delta-time 0.1
    
    Using :class:`petpal.cli_tac_interpolation.EvenlyInterpolateWithMax`:
        .. code-block:: bash
            
            petpal-tac-interpolation --input-tac-path /path/to/input.tac --output-tac-path /path/to/output.tac --samples-before-max 4
    
See Also:
    :mod:`tac_interpolation <petpal.tac_interpolation>` - module for performing interpolation on PET TACs.

TODO:
    * Refactor the reading and writing of TACs when IO module is mature.
    
"""


import argparse
import numpy as np
from ..kinetic_modeling import tac_interpolation as tac_intp
import os


def _safe_load_tac(filename: str) -> np.ndarray:
    """
    Loads time-activity curves (TAC) from a file.

    Tries to read a TAC from specified file and raises an exception if unable to do so. We assume that the file has two
    columns, the first corresponding to time and second corresponding to activity.

    Args:
        filename (str): The name of the file to be loaded.

    Returns:
        np.ndarray: A numpy array containing the loaded TAC. The first index corresponds to the times, and the second
        corresponds to the activity.

    Raises:
        Exception: An error occurred loading the TAC.
    """
    try:
        return np.array(np.loadtxt(filename).T, dtype=float, order='C')
    except Exception as e:
        print(f"Couldn't read file {filename}. Error: {e}")
        raise e


def _safe_write_tac(tac_times: np.ndarray, tac_values: np.ndarray, filename: str) -> None:
    """
    Writes time-activity curves (TAC) to a file.

    Tries to write a TAC to the specified file and raises an exception if unable to do so. The TAC is expected to be a
    numpy array where the first index corresponds to the times, and the second corresponds to the activity.

    Args:
        tac_times (np.ndarray): A numpy array containing the time points of the TAC.
        tac_values (np.ndarray): A numpy array containing the activity values of the TAC.
        filename (str): The name of the file to write to.

    Raises:
        Exception: An error occurred writing the TAC.
    """
    out_arr = np.array([tac_times, tac_values]).T
    try:
        np.savetxt(fname=f"{filename}.tsv", X=out_arr, delimiter="\t", header="Time\tActivity", fmt="%.6e")
    except Exception as e:
        print(f"Couldn't write file {filename}. Error: {e}")
        raise e


def _print_tac_to_screen(tac_times: np.ndarray, tac_values: np.ndarray):
    """
    Prints the Time-Activity Curve (TAC) times and values to the console.

    This function takes as input two numpy arrays, one with the TAC times and the other with the TAC values, and prints
    them to the console in a formatted manner. The format is '%.6e\t%.6e'.

    Args:
        tac_times (np.ndarray): A numpy array containing the TAC times.
        tac_values (np.ndarray): A numpy array containing the TAC values.

    """
    print(f"#{'Time':<9}\tActivity")
    for time, value in zip(tac_times, tac_values):
        print(f"{time:<.6e}\t{value:<.6e}")


def main():
    parser = argparse.ArgumentParser(prog="TAC Interpolation", description="Evenly resample TACs.",
                                     epilog="Example of usage: petpal-tac-interpolate -i /path/to/input/file.txt -o "
                                            "/path/to/output/file.txt --delta-time 0.1")
    
    io_grp = parser.add_argument_group("I/O")
    io_grp.add_argument("-i", "--tac-path", help="Path to TAC file.", required=True)
    io_grp.add_argument("-o", "--out-tac-path", help="Path of output file.", required=True)
    
    interp_grp = parser.add_argument_group("Interpolation")
    mutually_exclusive_group = interp_grp.add_mutually_exclusive_group(required=True)
    mutually_exclusive_group.add_argument("--delta-time", type=float,
                                          help="The time difference for the resampled times.")
    mutually_exclusive_group.add_argument("--samples-before-max", type=float,
                                          help="Number of samples before the max TAC value.")
    
    verb_group = parser.add_argument_group("Additional information")
    verb_group.add_argument("-p", "--print", action="store_true", help="Print the resampled TAC values.",
                            required=False)
    verb_group.add_argument("-v", "--verbose", action="store_true", help="Print the sizes of the input and output TACs",
                            required=False)
    
    args = parser.parse_args()
    args.tac_path = os.path.abspath(args.tac_path)
    args.out_tac_path = os.path.abspath(args.out_tac_path)
    
    in_tac_times, in_tac_values = _safe_load_tac(args.tac_path)
    
    if args.samples_before_max is not None:
        interpolator = tac_intp.EvenlyInterpolateWithMax(tac_times=in_tac_times, tac_values=in_tac_values,
                                                         samples_before_max=args.samples_before_max)
    else:
        interpolator = tac_intp.EvenlyInterpolate(tac_times=in_tac_times, tac_values=in_tac_values,
                                                  delta_time=args.delta_time)
    
    resampled_times, resampled_values = interpolator.get_resampled_tac()
    
    _safe_write_tac(tac_times=resampled_times, tac_values=resampled_values, filename=args.out_tac_path)
    
    if args.verbose:
        print(f"Input TAC size:  {len(in_tac_values)}.")
        print(f"Output TAC size: {len(resampled_values)}.")
    
    if args.print:
        _print_tac_to_screen(tac_times=resampled_times, tac_values=resampled_values)


if __name__ == "__main__":
    main()
