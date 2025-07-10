"""
Command-line interface (CLI) for Partial Volume Correction (PVC) using SGTM and PETPVC methods.
This module provides a CLI to apply PVC to PET images using either the SGTM method or the PETPVC package.
It uses argparse to handle command-line arguments and chooses the appropriate method based on the provided input.
The user must provide:
    * PET image file path
    * Segmentation image file path
    * FWHM for Gaussian blurring
    * PVC method ('SGTM' or any other method for PETPVC)
    * Additional options for PETPVC
Example usage:
    Using SGTM method:
        .. code-block:: bash
            pvc_cli.py --method SGTM --pet-path /path/to/pet_image.nii --roi-path /path/to/roi_image.nii --fwhm (8.0, 7.0, 7.0)
    Using PETPVC method:
        .. code-block:: bash
            pvc_cli.py --method RBV --pet-path /path/to/pet_image.nii --roi-path /path/to/roi_image.nii --fwhm 6.0 --output-path /path/to/output_image.nii
See Also:
    SGTM and PETPVC methods implementation modules.
    :mod:`SGTM <pet_cli.symmetric_geometric_transfer_matrix>` - module for performing symmetric Geometric Transfer Matrix PVC.
    :mod:`PETPVC <pet_cli.partial_volume_corrections>` - wrapper for PETPVC package.
"""
import argparse

from ..preproc.symmetric_geometric_transfer_matrix import Sgtm


def sgtm_cli_run(input_image_path: str,
                 segmentation_image_path: str,
                 fwhm: float | tuple[float, float, float],
                 output_path: str):
    """
    Apply the SGTM method for Partial Volume Correction.
    """
    sgtm_obj = Sgtm(input_image_path=input_image_path,
                    segmentation_image_path=segmentation_image_path,
                    fwhm=fwhm)
    sgtm_obj(output_path=output_path)

def main():
    """
    Main function to handle command-line arguments and apply the appropriate PVC method.
    """
    parser = argparse.ArgumentParser(prog="PVC CLI",
                                     description="Apply Partial Volume Correction (PVC) to PET"
                                                 " images using sGTM. Works on 3D or 4D PET. 3D "
                                                 "result is the corrected uptake in each region, "
                                                 "4D result is the corrected TAC for each region.",
                                     epilog="Example of usage: pet-cli-pvc --pet-path"
                                            " /path/to/pet_image.nii --roi-path "
                                            "/path/to/roi_image.nii --fwhm 8.0")
    parser.add_argument("-i",
                        "--input-image",
                        required=True,
                        help="Path to the PET image file. Can be 3D or 4D.")
    parser.add_argument("-s",
                        "--segmentation_image",
                        required=True,
                        help="Path to the Segmentation image file.")
    parser.add_argument("-f",
                        "--fwhm",
                        required=True,
                        type=float,
                        help="Full Width at Half Maximum for Gaussian blurring (Tuple or single "
                             "float) in mm.")
    parser.add_argument("-o",
                        "--output",
                        required=True,
                        help="Path to PVC result. If input image is 3D, writes to a TSV file. If "
                             "input image is 4D, writes to a directory.")

    args = parser.parse_args()

    sgtm_cli_run(input_image_path=args.input_image,
                 segmentation_image_path=args.segmentation_image,
                 fwhm=args.fwhm,
                 output_path=args.output)

if __name__ == "__main__":
    main()
