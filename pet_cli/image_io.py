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

class ImageIO():
    """
    Class handling 3D and 4D image file utilities.
    """
    def __init__(self,
            verbose: bool=True,
            ):
        """
        Constructor for class ImageIO.

        Args:
            verbose (bool): Set to True to print debugging info to shell. Defaults to True.
        """
        self.verbose = verbose


    def load_nii(self,image_path: str) -> FileBasedImage:
        """
        Wrapper to load nifti from image_path.

        Args:
            image_path (str): Path to a .nii or .nii.gz file.
        
        Returns:
            The nifti FileBasedImage.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        if not re.search('.nii$ | .nii.gz$',image_path):
            raise OSError(f"{image_path} does not have the extension .nii or .nii.gz")
        image = nibabel.load(image_path)

        if self.verbose:
            print(f"(ImageIO): {image_path} loaded")

        return image


    def save_nii(self,
            image: nibabel.nifti1.Nifti1Image,out_file: str) -> int:
        """
        Wrapper to save nifti to file.

        Args:
            image (nibabel.nifti1.Nifti1Image): Nibabel-type image to write to file.
            out_file (str): File path to which image will be written.
        """
        nibabel.save(image,out_file)
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
        image_nibabel = nibabel.nifti1.Nifti1Image(image_array,affine,header)
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
        origin = image_affine[:,3]

        quat = nibabel.quaternions.mat2quat(image_affine[:3,:3])
        dir_3x3 = nibabel.quaternions.quat2mat(quat)
        direction = np.zeros((4,4))
        direction[-1,-1] = 1
        direction[:3,:3] = dir_3x3

        return spacing, origin, direction


    def extract_np_to_ants(self,
                           image_array: np.ndarray,
                           affine: np.ndarray) -> ants.ANTsImage:
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
        image_ants = ants.from_numpy(data=image_array,
                                     spacing=spacing,
                                     origin=origin,
                                     direction=direction)
        return image_ants

    @staticmethod
    def read_ctab(ctab_file: str) -> dict:
        """
        Static method to read a color table, translating region indices to region names, 
        as a dictionary. Assumes json format.

        Args:
            ctab_file (str): Path to a json-formatted color table file.

        Returns:
            ctab_json (dict): Dictionary where keys are region names and values are region indices.
        """
        if not os.path.exists(ctab_file):
            raise FileNotFoundError(f"Image file {ctab_file} not found")
        ctab_json = json.load(ctab_file)
        return ctab_json

    @staticmethod
    def load_meta(image_path) -> dict:
        """
        Static method to load metadata. Assume same path as input image path.

        Args:
            image_path (str): Path to image for which a .json file of the
                same name as the file but with different extension exists.

        Returns:
            image_meta (dict): Dictionary where keys are fields in the image
                metadata file and values correspond to values in those fields.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        meta_path = re.sub('.nii.gz|.nii','.json',image_path)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} not found")
        with open(meta_path,'r',encoding='utf-8') as meta_file:
            image_meta = json.load(meta_file)
        return image_meta
