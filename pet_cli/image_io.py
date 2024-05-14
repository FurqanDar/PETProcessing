"""
Image IO
"""
import json
import re
import os
import ants
import nibabel
from nibabel.filebasedimages import FileBasedHeader, FileBasedImage
import numpy as np
import pandas as pd
from pet_cli import useful_functions

def write_dict_to_json(meta_data_dict: dict, out_path: str):
    """
    Save a metadata file in python to a directory.

    Args:
        meta_data_dict (dict): A dictionary with imaging metadata, to be saved to file.
        out_path (str): Directory to which `meta_file` is to be saved.
    """
    with open(out_path, 'w', encoding='utf-8') as copy_file:
        json.dump(meta_data_dict, copy_file, indent=4)


def convert_ctab_to_dseg(ctab_path: str,
                         dseg_path: str,
                         column_names: list[str]=['mapping','name','r','g','b','a','ttype']):
    """
    Convert a FreeSurfer compatible color table into a BIDS compatible label
    map ``dseg.tsv``.

    Args:
        ctab_path (str): Path to FreeSurfer compatible color table.
        dseg_path (str): Path to ``dseg.tsv`` label mapfile to save.
        column_names (list[str]): List of columns present in color table. Must
            include 'mapping' and 'name'.
    """
    fs_ctab = pd.read_csv(ctab_path,
                          delim_whitespace=True,
                          header=None,
                          comment='#',
                          names=column_names)
    label_map = pd.DataFrame(columns=['name','abbreviation','mapping']).rename_axis('index')
    label_map['name'] = fs_ctab['name']
    label_map['mapping'] = fs_ctab['mapping']
    label_map['abbreviation'] = useful_functions.build_label_map(fs_ctab['name'])
    label_map = label_map.sort_values(by=['mapping'])
    label_map.to_csv(dseg_path,sep='\t')
    return label_map


class ImageIO():
    """
    :class:`ImageIO` to handle reading and writing imaging data and metadata.

    Provides several tools designed for reading and writing data within the Python environment.

    Key methods include:
        - :meth:`load_nii`: Loads a NIfTI file from a file path.
        - :meth:`save_nii`: Saves a loaded NIfTI file to a file path.
        - :meth:`extract_image_from_nii_as_numpy`: Extracts imaging data from a NIfTI file as a numpy array.
        - :meth:`extract_header_from_nii`: Extracts header information from a NIfTI file as a dictionary.
        - :meth:`extract_np_to_nibabel`: Wraps imaging information in numpy into an Nibabel image.

    Attributes:
        verbose (bool): Set to `True` to output processing information.
    """

    def __init__(self, verbose: bool = True, ):
        """
        Initializes :class:`ImageIO` and sets verbose.

        Args:
            verbose (bool): Set to True to print debugging info to shell. Defaults to True.
        """
        self.verbose = verbose

    def load_nii(self, image_path: str) -> FileBasedImage:
        """
        Wrapper to load nifti from image_path.

        Args:
            image_path (str): Path to a .nii or .nii.gz file.
        
        Returns:
            The nifti FileBasedImage.

        Raises:
            FileNotFoundError: If the provided image path cannot be found in the directory.
            OSError: If the provided image path does not have a NIfTI extension.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        
        if not re.search(r'\.nii\.gz$|\.nii$', image_path):
            raise OSError(f"{image_path} does not have the extension .nii or .nii.gz")
        
        image = nibabel.load(image_path)
        
        if self.verbose:
            print(f"(ImageIO): {image_path} loaded")

        return image

    def save_nii(self, image: nibabel.nifti1.Nifti1Image, out_file: str):
        """
        Wrapper to save nifti to file.

        Args:
            image (nibabel.nifti1.Nifti1Image): Nibabel-type image to write to file.
            out_file (str): File path to which image will be written.
        """
        nibabel.save(image, out_file)
        if self.verbose:
            print(f"(ImageIO): Image saved to {out_file}")

    def extract_image_from_nii_as_numpy(self, image: nibabel.nifti1.Nifti1Image) -> np.ndarray:
        """
        Convenient wrapper to extract data from a .nii or .nii.gz file as a numpy array.

        Args:
            image (nibabel.nifti1.Nifti1Image): Nibabel-type image to write to file.

        Returns:
            The data contained in the .nii or .nii.gz file as a numpy array.
        """
        image_data = image.get_fdata()

        if self.verbose:
            print(f"(ImageIO): Image has shape {image_data.shape}")

        return image_data

    def extract_header_from_nii(self, image: nibabel.nifti1.Nifti1Image) -> FileBasedHeader:
        """
        Convenient wrapper to extract header information from a .nii or .nii.gz 
        file as a nibabel file-based header.

        Args:
            image (nibabel.nifti1.Nifti1Image): Nibabel-type image to write to file.

        Returns:
            image_header (FileBasedHeader): The nifti header.
        """
        image_header = image.header

        if self.verbose:
            print(f"(ImageIO): Image header is: {image_header}")

        return image_header

    def extract_np_to_nibabel(self,
                              image_array: np.ndarray,
                              header: FileBasedHeader,
                              affine: np.ndarray) -> nibabel.nifti1.Nifti1Image:
        """
        Wrapper to convert an image array into nibabel object.

        Args:
            image_array (np.ndarray): Array containing image data.
            header (FileBasedHeader): Header information to include.
            affine (np.ndarray): Affine information we need to keep when rewriting image.

        Returns:
            image_nibabel (nibabel.nifti1.Nifti1Image): Image stored in nifti-like nibabel format. 
        """
        image_nibabel = nibabel.nifti1.Nifti1Image(image_array, affine, header)
        return image_nibabel

    @staticmethod
    def affine_parse(image_affine: np.ndarray) -> tuple:
        """
        Parse the components of an image affine to return origin, spacing, direction.

        Args:
            image_affine (np.ndarray): A 4x4 affine matrix defining spacing, origin,
                and direction of an image.
        """
        spacing = nibabel.affines.voxel_sizes(image_affine)
        origin = image_affine[:, 3]

        quat = nibabel.quaternions.mat2quat(image_affine[:3, :3])
        dir_3x3 = nibabel.quaternions.quat2mat(quat)
        direction = np.zeros((4, 4))
        direction[-1, -1] = 1
        direction[:3, :3] = dir_3x3

        return spacing, origin, direction

    def extract_np_to_ants(self, image_array: np.ndarray, affine: np.ndarray) -> ants.ANTsImage:
        """
        Wrapper to convert an image array into ants object.
        Note header info is lost as ANTs does not carry this metadata.

        Args:
            image_array (np.ndarray): Array containing image data.
            affine (np.ndarray): Affine information we need to keep when rewriting image.

        Returns:
            image_ants (ants.ANTsImage): Image stored in nifti-like nibabel format. 
        """
        origin, spacing, direction = self.affine_parse(affine)
        image_ants = ants.from_numpy(data=image_array, spacing=spacing, origin=origin, direction=direction)
        return image_ants

    @staticmethod
    def read_label_map_tsv(label_map_file: str) -> dict:
        """
        Static method to read a label map, translating region indices to region names, 
        as a dictionary. Assumes tsv format.

        Args:
            label_map_file (str): Path to a json-formatted label map file.

        Returns:
            label_map (pd.DataFrame): Dataframe matching region indices, names,
                abbreviations, and mappings.

        Raises:
            FileNotFoundError: If the provided ctab file cannot be found in the directory.
        """
        if not os.path.exists(label_map_file):
            raise FileNotFoundError(f"Image file {label_map_file} not found")

        label_map = pd.read_csv(label_map_file,sep='\t')

        return label_map

    @staticmethod
    def load_metadata_for_nifty_with_same_filename(image_path) -> dict:
        """
        Static method to load metadata. Assume same path as input image path.

        Args:
            image_path (str): Path to image for which a .json file of the
                same name as the file but with different extension exists.

        Returns:
            image_meta (dict): Dictionary where keys are fields in the image
                metadata file and values correspond to values in those fields.
        
        Raises:
            FileNotFoundError: If the provided image path cannot be found in the directory. 
            Additionally, occurs if the metadata .json file cannot be found.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        
        meta_path = re.sub(r'\.nii\.gz$|\.nii$', '.json', image_path)
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} not found. Does it have a different path?")
        
        with open(meta_path, 'r', encoding='utf-8') as meta_file:
            image_meta = json.load(meta_file)
        return image_meta
