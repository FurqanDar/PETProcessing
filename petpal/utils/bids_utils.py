"""
This module contains utilities for handling `Brain Imaging Data Structure (BIDS) <https://bids.neuroimaging.io/>`_ datasets. It simplifies the
creation and management of BIDS projects, offering functions for building project scaffolds, handling file paths,
and managing neuroimaging data files. Key features include scaffolding BIDS projects, caching file paths for efficient
retrieval, and supporting various neuroimaging file types through integration with `nibabel`.
"""
import os
import json
import numpy
import shutil
import warnings
from pathlib import Path
from bids_validator import BIDSValidator
from nibabel.nifti1 import Nifti1Image
from nibabel.filebasedimages import FileBasedImage


# from .registration_tools import ImageIO


class BidsInstance:
    """
    Manages BIDS dataset paths and facilitates the creation and organization of a BIDS project scaffold.

    Provides methods for constructing compliant file paths, caching for quick access, and operations on BIDS datasets,
    including file and directory management.

    Attributes:
        project_path (str): The root path of the BIDS project.
        path_cache (dict): A cache for storing and retrieving file paths.
        parts (dict): Components of the BIDS file path.
        filepath (str): The current file path being worked on.
        prefixes (dict): Prefixes for various components in BIDS file naming convention.
        directory_parts_names (tuple): Names of components used in directory paths.
        file_parts_names (tuple): Names of components used in file naming.

    Usage:
        Initialize with a project path and subject identifier to manage dataset paths and create BIDS scaffolds.
    """

    def __init__(self,
                 project_path: str,
                 subject: str):
        """
        Initializes a BidsInstance object with project path and subject.

        Calls :func:`_setup_dynamic_methods` and :func:`_create_bids_scaffold`.

        Args:
            project_path (str): The root directory of the BIDS project.
            subject (str): The subject identifier.
        """
        self.project_path = project_path
        self.path_cache = {}
        self.parts = {"derivative_directory": "main",
                      "subject": subject,
                      "session": None,
                      "modality": None,
                      "acquisition": None,
                      "contrast_enhancing": None,
                      "reconstruction": None,
                      "space": None,
                      "description": None,
                      "image_type": None,
                      "extension": None}
        self.filepath = ""
        self.prefixes = {"subject": "sub-",
                         "session": "ses-",
                         "acquisition": "acq-",
                         "contrast_enhancing": "ce-",
                         "reconstruction": "rec-",
                         "space": "space-",
                         "description": "desc-"}
        self.directory_parts_names = ("subject",
                                      "session",
                                      "modality")
        self.file_parts_names = ("subject",
                                 "session",
                                 "acquisition",
                                 "contrast_enhancing",
                                 "reconstruction",
                                 "space",
                                 "description",
                                 "image_type")
        self.required_metadata = ("FrameReferenceTime",
                                  "FrameTimesStart",
                                  "FrameDuration",
                                  ["DecayFactor", "DecayCorrectionFactor"],
                                  ["TracerRadionuclide", "Radiopharmaceutical"])
        self._setup_dynamic_methods()
        self._create_bids_scaffold()

    def _create_bids_scaffold(self):
        """
        Creates the necessary directories and files for a BIDS project scaffold.

        Example below

        .. code-block:: text

            project/
            \u251C\u2500\u2500 CHANGES
            \u251C\u2500\u2500 README
            \u251C\u2500\u2500 participants.json
            \u251C\u2500\u2500 participants.tsv
            \u251C\u2500\u2500 code/
            \u2502   \u2514\u2500\u2500 example.code
            \u251C\u2500\u2500 derivatives/
            \u2502   \u2514\u2500\u2500 example.file
            \u251C\u2500\u2500 sourcedata/
            \u2502   \u2514\u2500\u2500 example.file
            \u2514\u2500\u2500 sub-001/
                \u2514\u2500\u2500 ses-01/
                    \u2514\u2500\u2500 anat/
                        \u251C\u2500\u2500 example.nii
                        \u2514\u2500\u2500 example.json


        This structure helps visualize the organization of a simple project.
        """
        dirs_to_create = [
            "code",
            "derivatives",
            "sourcedata",
        ]
        files_to_create = {
            "CHANGES": "Initial commit.",
            #        "dataset_description.json": json.dumps({
            #            "Name": "Example Dataset",
            #            "BIDSVersion": "1.6.0",
            #            "DatasetType": "raw"
            #        }, indent=4),
            "participants.json": "{}",
            "participants.tsv": "subject_id\tsession_id",
            "README": "This BIDS dataset was created for the file outputs of the PetProcessing Pipeline."
        }

        # Create directories
        for dir_name in dirs_to_create:
            full_path = os.path.join(self.project_path, dir_name)
            os.makedirs(full_path, exist_ok=True)

        # Create files
        for file_name, content in files_to_create.items():
            full_path = os.path.join(self.project_path, file_name)  # change to only write if not existing
            if not os.path.exists(full_path):
                with open(full_path, 'w') as f:
                    f.write(content)

    def _update_participants(self, session: str) -> None:
        """
        Updates the participants.tsv file with a new session for the current subject.

        Args:
            session (str): The session identifier to add for the current subject.
        """
        if self.parts['session'] != session:
            self._add_subject_session_to_participants(subject=self.parts['subject'], session=session)

    def _add_subject_session_to_participants(self, subject: str, session: str) -> None:
        """
        Adds a subject-session pair to the participants.tsv file.

        Args:
            subject (str): The subject identifier.
            session (str): The session identifier.
        """
        participants_tsv_path = os.path.join(self.project_path, "participants.tsv")
        participants_tsv = self.load_file(filepath=participants_tsv_path)
        participants_tsv.append([subject, session])
        self.write_file(file_input=participants_tsv, filepath=participants_tsv_path)

    def _prefixed_dictionary(self) -> dict:
        """
        Prepares a dictionary with prefixed components for constructing BIDS file paths.

        Returns:
            dict: A dictionary with BIDS components prefixed according to BIDS naming conventions.
        """
        out_dict = self.parts.copy()
        for key in list(out_dict.keys()):
            value = out_dict[key]
            if isinstance(value, str):
                if key in self.prefixes:
                    if not value.startswith(self.prefixes[key]):
                        out_dict[key] = self.prefixes[key] + out_dict[key]
            else:
                out_dict.pop(key)
        return out_dict

    def create_filepath(self,
                        session: str,
                        modality: str,
                        image_type: str,
                        acquisition: str = None,
                        contrast_enhancing: str = None,
                        reconstruction: str = None,
                        space: str = None,
                        description: str = None,
                        derivative_directory: str = "main",
                        extension: str = "") -> None:
        """
        Constructs and updates the filepath attribute based on BIDS naming conventions.

        Args:
            session (str): The session identifier.
            modality (str): The modality or type of data.
            image_type (str): The type of image.
            acquisition (str, optional): The acquisition type.
            contrast_enhancing (str, optional): The contrast enhancing agent.
            reconstruction (str, optional): The reconstruction algorithm.
            space (str, optional): The space or coordinate system.
            description (str, optional): A description of the file.
            derivative_directory (str, optional): The directory for derivatives.
            extension (str, optional): The file extension.
        """

        parameters_dictionary = locals().copy()
        self._update_participants(session=session)
        self.parts = {**self.parts, **parameters_dictionary}
        self._compile_filepath()

    def _compile_filepath(self) -> None:
        """
        Compiles the full file path from the individual BIDS components stored in the object.
        """
        extension = self.parts['extension']
        parts_prefixed = self._prefixed_dictionary()
        filename_parts = [parts_prefixed[key] for key in self.file_parts_names if self.parts.get(key) is not None]
        filename = '_'.join(filename_parts) + extension

        derivative_directory = self.parts['derivative_directory']
        directory_parts = [parts_prefixed[key] for key in self.directory_parts_names if self.parts.get(key) is not None]
        if derivative_directory is not None and derivative_directory != "main":
            directory_parts = ["derivatives", derivative_directory] + directory_parts
        directory_path = os.path.join(self.project_path, '/'.join(directory_parts))
        self.filepath = os.path.join(directory_path, filename)

    def manual_filepath(self, filepath: str) -> None:
        """
        Manually sets the file path based on a provided path and updates class attributes.

        Args:
            filepath (str): The file path to set.
        """
        slash_parts = filepath.split("/")
        underscore_parts = slash_parts[-1].split("_")
        dot_parts = underscore_parts[-1].split(".")
        directory_parts = slash_parts[0:-1]
        filename_parts = underscore_parts[0:-1] + [dot_parts[0]]
        if len(dot_parts) > 1:
            extension = '.' + '.'.join(dot_parts[1:])
        else:
            extension = ""
        all_parts = directory_parts + filename_parts + [extension]

        for key, prefix in self.prefixes.items():
            for part in (part for part in all_parts if part.startswith(prefix)):
                if key == "session":
                    self._update_participants(session=part[len(prefix):])
                self.parts[key] = part[len(prefix):]
                break
        if "derivatives" in directory_parts:
            self.parts['derivative_directory'] = directory_parts[directory_parts.index("derivatives") + 1]
        self.parts['modality'] = directory_parts[-1]
        self.parts['image_type'] = filename_parts[-1]
        self.parts['extension'] = extension

        self._compile_filepath()

    def cache_filepath(self, name: str) -> None:
        """
        Caches the current file path with a given name for later retrieval.

        Args:
            name (str): The name under which to cache the current file path.

        Note:
            The filepath cache can be accessed using the names provided as keys in a dictionary: self.path_cache["name"].
        """
        self.path_cache[name] = self.filepath

    #    def load_from_cache(self, name: str) -> None: # get, if none -> warning
    #        return self.path_cache[name]

    def cache_sidecar_metadata(self, pet_sidecar_filepath: str) -> None:
        """
        Loads metadata from a JSON file specified by `pet_sidecar_filepath`, updates the instance
        with dynamic attributes based on the `required_metadata`, and issues warnings for any
        required metadata keys that are missing.

        This method dynamically sets attributes on the instance for each key in `required_metadata`
        that is found in the JSON file's keys. If `required_metadata` contains lists of keys,
        it checks for any of those keys in the JSON file and uses the first matching key to set
        an attribute named after the first key in the list. If a key from `required_metadata` (whether
        a single key or any key from a list) is not found in the JSON keys, a warning is issued indicating
        its absence.

        Args:
            pet_sidecar_filepath (str): The file path to the JSON file from which to load metadata. This
                file should contain a dictionary where each key-value pair corresponds to metadata that
                might be required.

        Returns:
            None

        Raises:
            JSONDecodeError: If the JSON file is malformed and cannot be decoded.
            FileNotFoundError: If the specified file does not exist.
            Warning: Issues a runtime warning through the `warnings` module if required keys are missing
                from the JSON data.
        """
        metadata_cache = load_json(filepath=pet_sidecar_filepath)
        json_keys = metadata_cache.keys()
        for keys in self.required_metadata:
            if isinstance(keys, list):
                if any(key in json_keys for key in keys):
                    for json_key in json_keys:
                        if json_key in keys:
                            setattr(self, keys[0], metadata_cache[json_key])
                            break
                else:
                    warnings.warn(f"{keys} is not found in {pet_sidecar_filepath}")
            else:
                if keys in json_keys:
                    setattr(self, keys, metadata_cache[keys])
                else:
                    warnings.warn(f"{keys} is not found in {pet_sidecar_filepath}")

    def change_session(self, value: str, compile_filepath: bool = True):
        """
        Updates the session part of the file path and optionally recompile the path.

        Args:
            value (str): The new session value to set.
            compile_filepath (bool, optional): Whether to recompile the file path after updating. Defaults to True.

        Note:
            This is the only "change_<element>" function that is not dynamically generated because session information needs to be stored in the participants.tsv file, thus needing special handling.
        """
        self._update_participants(session=value)
        self.parts['session'] = value
        if compile_filepath:
            self._compile_filepath()

    def _update_part(self, key, value, compile_filepath: bool = True):
        """
        Updates a specified part of the file path and optionally recompile the path.

        Args:
            key (str): The key of the part to update.
            value (str): The new value for the specified part.
            compile_filepath (bool, optional): Whether to recompile the file path after updating. Defaults to True.
        """
        self.parts[key] = value
        if compile_filepath:
            self._compile_filepath()

    def _setup_dynamic_methods(self):
        """
        Dynamically creates and assigns methods for changing individual parts of the BIDS file path.

        Note:
            This creates callable functions to change individual elements of a filepath dynamically such that for every option (E.g. "modality") there is a function "change_<option>" (E.g. "change_modality").
        """

        for key in self.parts.keys():
            if key != "session":  # so that session can call _update_participants()
                update_method = self._create_update_method(key)
                setattr(self, f'change_{key}', update_method)

    def _create_update_method(self, key):
        """
        Creates a method for updating a specific part of the BIDS file path.

        Args:
            key (str): The key of the part to create an update method for.

        Returns:
            function: A function that updates the specified part.
        """

        def _update_method(value, compile_filepath: bool = True):
            self._update_part(key, value, compile_filepath)

        return _update_method

    def change_parts(self,
                     session: str = None,
                     modality: str = None,
                     image_type: str = None,
                     acquisition: str = None,
                     contrast_enhancing: str = None,
                     reconstruction: str = None,
                     space: str = None,
                     description: str = None,
                     derivative_directory: str = "main",
                     extension: str = None, ) -> None:
        """
        Updates multiple parts of the BIDS file path at once based on provided arguments.

        This function calls all the individual "change_<element>" functions that comprise the inputs and compiles them into a path at the end.

        Args:
            session (str): Session identifier.
            modality (str, optional): Modality or type of data.
            image_type (str, optional): Type of image.
            acquisition (str, optional): Acquisition type.
            contrast_enhancing (str, optional): Contrast enhancing agent.
            reconstruction (str, optional): Reconstruction algorithm.
            space (str, optional): Space or coordinate system.
            description (str, optional): Description of the file.
            derivative_directory (str, optional): Directory for derivatives.
            extension (str, optional): File extension.
        """
        parameters_dictionary = locals().copy()
        for key, value in parameters_dictionary.items():
            if value is not None:
                # Construct the method name expected based on the key
                method_name = f'change_{key}'

                # Check if this instance has a method with that name
                if hasattr(self, method_name):
                    # Get the method
                    method = getattr(self, method_name)

                    # Call the method with the provided value
                    method(value, compile_filepath=False)
                elif key != "self":
                    print(f"No method found for key: {key}")

        self._compile_filepath()

    def write_symbolic_link(self, input_filepath: str, link_filepath: str = None) -> None:
        """
        Creates a symbolic link to a specified input file at a given location.

        Args:
            input_filepath (str): The path of the file to link to.
            link_filepath (str, optional): The path where the symbolic link should be created.
                Defaults to current filepath.
        """
        if link_filepath is None:
            link_filepath = self.filepath

        input_filename = os.path.basename(input_filepath)
        extension = '.' + '.'.join(input_filename.split('.')[1:])
        self.parts['extension'] = extension
        self._compile_filepath()
        link_filepath_base = link_filepath.split('.')[0]
        link_filepath = link_filepath_base + extension
        os.makedirs(os.path.dirname(link_filepath), exist_ok=True)

        if os.path.exists(link_filepath) or os.path.islink(link_filepath):
            os.remove(link_filepath)
        os.symlink(input_filepath, link_filepath)

    def write_file(self, file_input, filepath: str = None) -> None:
        """
        Writes input data to a file of an appropriate format based on the file's extension.

        Args:
            file_input: The data to be written to the file. Its type determines how it's written.
            filepath (str, optional): The path where the data should be written. Defaults to current filepath.
        """
        if filepath is None:
            filepath = self.filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if isinstance(file_input, FileBasedImage) or isinstance(file_input, Nifti1Image):
            print("Nifti")
            # ImageIO.save_nii(image=file_input, out_file=self.filepath)
        elif type(file_input) is dict:
            save_json(json_dict=file_input, filepath=filepath)
        elif type(file_input) is numpy.array:
            save_array_as_tsv(array=file_input, filepath=filepath)
        elif type(file_input) is list:
            save_tsv_simple(data=file_input, filepath=filepath)

    def load_file(self, filepath: str = None):
        """
        Loads a file based on its extension and returns its content.

        Args:
            filepath (str, optional): The path of the file to load. Defaults to current filepath.

        Returns:
            The content of the loaded file, in a format depending on the file type.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            RuntimeError: If unable to load the file or unsupported file format.
        """
        file = None
        if filepath is None:
            filepath = self.filepath
        if os.path.exists(filepath) or os.path.islink(filepath):
            if filepath.endswith(".nii") or filepath.endswith(".nii.gz"):
                print("Nifti")
                # file = ImageIO.load_nii(filepath=self.filepath)
            elif filepath.endswith(".json"):
                file = load_json(filepath=filepath)
            elif filepath.endswith(".tsv"):
                file = load_tsv_simple(filepath=filepath)
            else:
                raise ValueError(f"Unsupported file type for {filepath}.")
        else:
            raise FileNotFoundError(f"The file '{filepath}' does not exist.")
        if file is None:
            raise RuntimeError(f"Failed to load the file or unsupported file format: {filepath}")

        return file

    def delete_file(self, filepath: str = None) -> None:
        """
        Deletes a specified file within the project.

        Args:
            filepath (str, optional): The path of the file to delete. If None, deletes current filepath.
        """
        if filepath is None:
            filepath = self.filepath
        elif not filepath.startswith(self.project_path):
            filepath = os.path.join(self.project_path, filepath)
        try:
            os.remove(filepath)
            # print(f"File {file_path} has been removed successfully.")
        except FileNotFoundError:
            print(f"File {filepath} does not exist.")
        except PermissionError:
            print(f"No permission to delete the file {filepath}.")

    def delete_directory(self, directory_path: str) -> None:
        """
        Deletes a specified directory within the project.

        Args:
            directory_path (str): The path of the directory to delete.
        """
        directory_path = os.path.join(self.project_path, directory_path)
        try:
            shutil.rmtree(directory_path)
            # print(f"Directory {directory_path} and all its contents have been removed successfully.")
        except FileNotFoundError:
            print(f"Directory {directory_path} does not exist.")
        except PermissionError:
            print(f"No permission to delete the directory {directory_path}.")


def create_json(**kwargs) -> dict:
    """
    Creates a dictionary from provided keyword arguments.

    Args:
        **kwargs: Arbitrary keyword arguments.

    Returns:
        A dictionary constructed from the keyword arguments.
    """
    return kwargs


def update_json(json_dict: dict,
                **kwargs) -> dict:
    """
    Updates an existing JSON dictionary with additional key-value pairs.

    Args:
        json_dict (dict): The original dictionary to be updated.
        **kwargs: Arbitrary keyword arguments to update json_dict with.

    Returns:
        The updated dictionary.
    """
    json_dict.update(**kwargs)
    return json_dict


def save_json(json_dict: dict,
              filepath: str) -> None:
    """
    Saves a dictionary as a JSON file at the specified filepath.

    Args:
        json_dict (dict): The dictionary to save as JSON.
        filepath (str): The destination file path. If the path does not end with ".json", it will be appended.

    Note:
        The JSON file is formatted with an indentation of 4 spaces for readability.
    """
    if not filepath.endswith(".json"):
        filepath += ".json"
    with open(filepath, 'w') as file:
        json.dump(json_dict, file, indent=4)
        file.write('\n')


def load_json(filepath: str):
    """
    Loads a JSON file from the specified filepath and returns its content as a dictionary.

    Args:
        filepath (str): The path to the JSON file to be loaded.

    Returns:
        dict: The JSON file content as a dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filepath}")
        return None


def save_array_as_tsv(array: numpy.array,
                      filepath: str) -> None:
    """
    Saves a NumPy array as a TSV (Tab-Separated Values) file.

    Args:
        array (numpy.array): The NumPy array to save.
        filepath (str): The destination file path for the TSV file.

    Note:
        Each row of the array is saved as a separate line in the TSV file.
    """
    numpy.savetxt(filepath, array, delimiter='\t', fmt='%s')


def save_tsv_simple(filepath: str, data: list) -> None:
    """
    Saves a list of lists as a TSV (Tab-Separated Values) file.

    Args:
        filepath (str): The destination file path for the TSV file.
        data (list): A list of lists, where each sublist represents a row in the TSV file.
    """
    with open(filepath, 'w', encoding='utf-8') as file:
        for row in data:
            line = '\t'.join(row)
            file.write(line + '\n')


def load_tsv_simple(filepath: str) -> list:
    """
    Loads a TSV (Tab-Separated Values) file from the specified filepath and returns its content as a list of lists.

    Args:
        filepath (str): The path to the TSV file to be loaded.

    Returns:
        A list of lists, where each sublist represents a row in the TSV file, split by tabs.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = [line.strip().split('\t') for line in file]
    return data


def validate_filepath_as_bids(filepath: str) -> bool:
    """
    Validate whether a given filepath conforms to the Brain Imaging Data Structure (BIDS) standard.

    Args:
        filepath (str): The path to the file to be validated.

    Returns:
        bool: True if the file conforms to the BIDS standard, False otherwise.

    """
    validator = BIDSValidator()
    return validator.is_bids(filepath)


def validate_directory_as_bids(project_path: str) -> bool:
    """
    Validate whether all files in a given directory and its subdirectories (excluding specified ones)
    conform to the Brain Imaging Data Structure (BIDS) standard.

    Args:
        project_path (str): The root directory of the project to validate.

    Returns:
        bool: True if all files in the directory conform to the BIDS standard, False if any do not.

    Raises:
        FileNotFoundError: If the provided project_path does not exist or is inaccessible.

    Notes:
        Excludes directories typically not needed for BIDS validation, such as 'code', 'derivatives',
        'sourcedata', '.git', and 'stimuli'. Also skips directories starting with 'sub-' to focus on top-level structure.
    """
    excluded_dirs = {'code', 'derivatives', 'sourcedata', '.git', 'stimuli'}
    all_passed = True
    failed_file_paths = []

    for root, dirs, files in os.walk(project_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('sub-')]
        for file in files:
            filepath = os.path.join(root, file)
            if not validate_filepath_as_bids(filepath):
                failed_file_paths.append(filepath)
                all_passed = False

    if failed_file_paths:
        print("Failed file paths:")
        for path in failed_file_paths:
            print(path)
    else:
        print("All files passed validation.")

    return all_passed
