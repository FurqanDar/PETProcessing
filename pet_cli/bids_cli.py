"""
This module provides a command-line interface (CLI) for interacting with the BIDS utilities package.

It allows users to perform tasks such as creating a BIDS scaffold, manipulating BIDS datasets,
and processing imaging data using command-line arguments.
"""
import argparse
from . import BIDS_utils


def main():
    """
    Parses command-line arguments and invokes the appropriate functions from the BIDS utilities package.

    The script supports various operations, including creating a BIDS scaffold,
    validating BIDS datasets (future), and converting user-provided files to BIDS format.
    """
    parser = argparse.ArgumentParser(prog="PET Processing",
                                     description="General purpose suite for processing PET images.",
                                     epilog="PET Processing complete.")

    io_grp = parser.add_argument_group("I/O")
    io_grp.add_argument('--project_path', required=True, help="Path to the BIDS project")
    io_grp.add_argument('--subject', required=True, help="Subject ID")
    io_grp.add_argument('--session', required=True, help="Session ID")
    io_grp.add_argument('--PET_image', required=True, help="Input 4D PET image file")
    io_grp.add_argument('--PET_sidecar', required=True, help="Input PET JSON sidecar file")
    io_grp.add_argument('--PET_label', required=True, help="Label of PET type (E.g. 'FDG', 'CS1P1')")
    io_grp.add_argument('--T1w_image', help="Input T1-weighted MR image file")
    io_grp.add_argument('--T1w_sidecar', help="Input T1-weighted MR JSON sidecar file")
    io_grp.add_argument('--T1w_label', help="Label of T1-weighted MRI type (E.g. 'T1w', 'MPRAGE')")
    io_grp.add_argument('--T2w_image', help="Input T2-weighted MR image file")
    io_grp.add_argument('--T2w_sidecar', help="Input T2-weighted MR JSON sidecar file")
    io_grp.add_argument('--T2w_label', help="Label of T2-weighted MRI type (E.g. 'T2w', 'FLAIR')")

    # spec_grp = parser.add_argument_group("Specific filepath naming")
    # spec_grp.add_argument('--project_path', required=True, help="Path to the BIDS project")
    # spec_grp.add_argument('--subject', required=True, help="Subject ID")
    # spec_grp.add_argument('--session', required=True, help="Session ID")
    # spec_grp.add_argument('--modality', required=True, help="Modality (E.g. 'anat', 'pet')")
    # spec_grp.add_argument('--image_type', required=True, help="Image type (E.g. 'T1w', 'FLAIR', 'FDG')")
    # spec_grp.add_argument('--acquisition', help="Acquisition details", default=None)
    # spec_grp.add_argument('--contrast_enhancing', help="Contrast enhancing agent ID", default=None)
    # spec_grp.add_argument('--reconstruction', help="Reconstruction method", default=None)
    # spec_grp.add_argument('--space', help="Reference image space", default=None)
    # spec_grp.add_argument('--description', help="Other descriptions", default=None)

    verb_group = parser.add_argument_group("Additional information")
    verb_group.add_argument("-p", "--print", action="store_true",
                            help="Print the calculated values to screen.", required=False)
    verb_group.add_argument("-v", "--verbose", action="store_true",
                            help="Print the shape of the mask and images files.", required=False)

    args = parser.parse_args()

    if args.PET_image and not args.PET_sidecar:
        parser.error("--PET_sidecar is required for scanner time-series when --PET_image is used.")
    if args.T1w_image and not args.T1w_label:
        parser.error("--T1w_label is required when --T1w_image is used.")
    if args.T2w_image and not args.T2w_label:
        parser.error("--T2w_label is required when --T2w_image is used.")

    bids_instance = BIDS_utils.BidsInstance(project_path=args.project_path, subject=args.subject)
    bids_instance.create_filepath(session=args.session,
                                  modality="pet",
                                  image_type=args.PET_label)
    bids_instance.write_symbolic_link(input_filepath=args.PET_image)
    bids_instance.cache_filepath(name="raw_PET_image")
    bids_instance.write_symbolic_link(input_filepath=args.PET_sidecar)
    bids_instance.cache_filepath(name="raw_PET_sidecar")
    if args.T1w_image:
        bids_instance.change_parts(modality="anat",
                                   image_type=args.T1w_label)
        bids_instance.write_symbolic_link(input_filepath=args.T1w_image)
        bids_instance.cache_filepath(name="raw_T1w_image")
        if args.T1w_sidecar:
            bids_instance.write_symbolic_link(input_filepath=args.T1w_sidecar)
            bids_instance.cache_filepath(name="raw_T1w_sidecar")
    if args.T2w_image:
        bids_instance.change_parts(modality="anat",
                                   image_type=args.T2w_label)
        bids_instance.write_symbolic_link(input_filepath=args.T2w_image)
        bids_instance.cache_filepath(name="raw_T2w_image")
        if args.T2w_sidecar:
            bids_instance.write_symbolic_link(input_filepath=args.T2w_sidecar)
            bids_instance.cache_filepath(name="raw_T2w_sidecar")


if __name__ == "__main__":
    main()
