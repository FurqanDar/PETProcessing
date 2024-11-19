"""
This module contains a class to wrap the PETPVC package accessible via `PETPVC on GitHub <https://github.com/UCL/PETPVC>`_.
This package allows for numerous partial volume correction methods on PET data.

Note:
    This module is primarily a wrapper for the PETPVC package implemented via a Docker image
    hosted on `DockerHub <https://hub.docker.com/r/benthomas1984/petpvc>`_. The Docker image is Linux-based.

Requires:
    The module relies on the :mod:`docker` module.

"""

import os
import docker
from docker.errors import ImageNotFound, APIError, ContainerError
from typing import Union, Tuple


class PetPvc:
    """
    Handles operations for PET partial volume correction using a Docker container.

    This class manages the setup and execution of processes using the PETPVC package,
    accessible via `PETPVC on GitHub <https://github.com/UCL/PETPVC>`_ and a Docker image
    hosted on `DockerHub <https://hub.docker.com/r/benthomas1984/petpvc>`_. It manages
    image retrieval, input/output setup, and command execution.

    Attributes:
        client (docker.DockerClient): A Docker client connected to the local system.
        image_name (str): The Docker image name to use for PETPVC processes.

    Examples:
        Initialize the PETPVC handler and run a correction process:

        .. code-block:: python

            from petpvc_module import PetPvc
            petpvc_handler = PetPvc()
            petpvc_handler.run_petpvc(
                pet_4d_filepath="/path/to/input/pet_image.nii",
                output_filepath="/path/to/output/corrected_pet_image.nii",
                pvc_method="RBV",
                psf_dimensions=(6.0, 6.0, 6.0),
                mask_filepath="/path/to/input/brain_mask.nii",
                verbose=True)
                
    """

    def __init__(self):
        """
        Initializes the PetPvc instance and ensures the required Docker image is available.
        """
        self.client = docker.from_env()
        self.image_name = "benthomas1984/petpvc"
        self._pull_image_if_not_exists()

    def run_petpvc(self,
                   pet_4d_filepath: str,
                   output_filepath: str,
                   pvc_method: str,
                   psf_dimensions: Union[Tuple[float, float, float], float],
                   mask_filepath: str = None,
                   verbose: bool = False,
                   debug: bool = False) -> None:
        """
        Executes the PETPVC correction process within a Docker container.

        Args:
            pet_4d_filepath (str): The file path to the input 4D PET image.
            output_filepath (str): The file path where the output image should be saved.
            pvc_method (str): The partial volume correction method to apply.
            psf_dimensions (Union[Tuple[float, float, float], float]): The full-width half-max (FWHM) in mm along x, y, z axes.
            mask_filepath (str, optional): The file path to the mask image. Defaults to None.
            verbose (bool, optional): If True, prints the output from the Docker container. Defaults to False.
            debug (bool, optional): If True, adds the --debug flag to the command for detailed logs. Defaults to False.

        Prints:
            Outputs from the Docker container if verbose is True.

        Raises:
            docker.errors.ImageNotFound: If the Docker image is not available.
            docker.errors.APIError: If the Docker client encounters an API error.
        """
        pet_4d_filepath = os.path.abspath(pet_4d_filepath)
        output_filepath = os.path.abspath(output_filepath)
        docker_mask_input = None
        if mask_filepath:
            mask_filepath = os.path.abspath(mask_filepath)

        # Determine the common path
        common_path = os.path.commonpath(
            [pet_4d_filepath, output_filepath, mask_filepath]) if mask_filepath else os.path.commonpath(
            [pet_4d_filepath, output_filepath])
        if not common_path:
            common_path = os.path.dirname(pet_4d_filepath)
        if common_path == "/":
            docker_pet_input = f"/data/{pet_4d_filepath}"
            docker_output = f"/data/{output_filepath}"
            if mask_filepath:
                docker_mask_input = f"/data/{mask_filepath}"
        else:
            common_path = common_path.rstrip(os.sep)
            docker_pet_input = f"/data/{pet_4d_filepath.replace(common_path, '').lstrip(os.sep).replace(os.sep, '/')}"
            docker_output = f"/data/{output_filepath.replace(common_path, '').lstrip(os.sep).replace(os.sep, '/')}"
            if mask_filepath:
                docker_mask_input = f"/data/{mask_filepath.replace(common_path, '').lstrip(os.sep).replace(os.sep, '/')}"

        docker_volumes = {common_path: {'bind': '/data', 'mode': 'rw'}}

        command = f"petpvc --input {docker_pet_input} --output {docker_output} --pvc {pvc_method}"
        if mask_filepath:
            command += f" --mask {docker_mask_input}"
        if isinstance(psf_dimensions, tuple):
            command += f" -x {psf_dimensions[0]} -y {psf_dimensions[1]} -z {psf_dimensions[2]}"
        else:
            command += f" -x {psf_dimensions} -y {psf_dimensions} -z {psf_dimensions}"
        if debug:
            command += " --debug"

        print(f"Running Docker command: {command}")
        print(f"Volume bindings: {docker_volumes}")

        try:
            container = self.client.containers.run(self.image_name, command, volumes=docker_volumes, detach=False,
                                                   stream=True, auto_remove=False)
            if verbose:
                for line in container:
                    print(line.decode('utf-8').strip())
        except ContainerError as e:
            print(f"ContainerError: {e}")
            container_id = e.container.id
            logs = self.client.containers.get(container_id).logs(stream=True)
            for line in logs:
                print(line.decode('utf-8').strip())
            self.client.containers.get(container_id).remove()
            raise
        except APIError as e:
            print(f"APIError: {e}")
            raise

    def _pull_image_if_not_exists(self) -> None:
        """
        Checks if the Docker image is locally available and pulls it if not.
        """
        try:
            self.client.images.get(self.image_name)
            print(f"Image {self.image_name} is already available locally.")
        except ImageNotFound:
            print(f"Image {self.image_name} not found locally. Pulling from Docker Hub...")
            self.client.images.pull(self.image_name)
            print(f"Successfully pulled {self.image_name}.")
        except APIError as error:
            print(f"Failed to pull image due to API error: {error}")
