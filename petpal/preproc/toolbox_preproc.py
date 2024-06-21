import argparse
import json
from time import perf_counter
from . import preproc

_PREPROC_EXAMPLES_ = (r"""
Examples:
  - Create a blank params file:
    petpal-preproc -p path_to_new_params.json
  - Run one or more preprocessing methods in order:
    petpal-preproc -p path_to_existing_params.json -o /path/to/processing/dir -f sub-PETSTUDY001 -m [method1] [method2] ...
""")


def create_blank_params(params_file: str):
    preproc_props = preproc._PREPROC_PROPS_
    with open(params_file,'w') as f:
        json.dump(preproc_props,f,indent=4)


def main():
    """
    Runs preprocessing on the command line. Takes four inputs: a params file,
    an output directory, a prefix for files, and one or more methods to run in
    order. If the output directory is blank, default to current directory. If
    the filename prefix is blank, default to 'sub-001'.
    """
    parser = argparse.ArgumentParser(prog='petpal-preproc',
                                     description='Preprocessing command line interface',
                                     epilog=_PREPROC_EXAMPLES_,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p','--params',help='Path to a params file',required=True)
    parser.add_argument('-o','--output_directory',help='Directory to which output files are saved.',required=False)
    parser.add_argument('-f','--filename_prefix',help='Prefix appended to beginning of written file names.',required=False)
    parser.add_argument('-m','--method',help='Name of process or method to run',nargs='*',required=False)
    args = parser.parse_args()

    if args.method is None:
        create_blank_params(args.params)
        return 0

    with open(args.params,'r') as f:
        preproc_props = json.load(f)

    output_directory = args.output_directory
    if args.output_directory is None:
        output_directory = '.'

    filename_prefix = args.filename_prefix
    if args.filename_prefix is None:
        filename_prefix = 'sub-001'

    subject = preproc.PreProc(output_directory=output_directory,
                              output_filename_prefix=filename_prefix)
    subject.update_props(new_preproc_props=preproc_props)

    start = perf_counter()
    methods = args.method
    for method in methods:
        subject.run_preproc(method_name=method)
    finish = perf_counter()
    print(f'Finished processing in {finish-start} s')


if __name__ == "__main__":
    main()