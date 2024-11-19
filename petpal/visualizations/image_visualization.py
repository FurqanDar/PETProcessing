"""
Provides a class for creating a GIF from a NIfTI image which iterates through its slices based
on a specified view ('coronal', 'sagittal', or 'axial').

Classes defined in the module:
    * :class:`NiftiGifCreator`: This class creates a GIF from a NIfTI image by iterating through its slices based on
      a specified view ('coronal', 'sagittal', or 'axial'). The GIF is then written to the
      specified output directory with a filename prefix specified by the user.

TODO:
    * Add functionality to quickly generate slices of images.
    * Add flexibility in :class:`NiftiGifCreator` to accept a NumPy array or a file-path. This could also be added
      to the image slice class.

"""

import os
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from ..utils import image_io as pet_pim

nifty_loader = pet_pim.safe_load_4dpet_nifti


class NiftiGifCreator:
    """
    This class is designed to create a GIF from a NIfTI image by iterating through its slices based on a specified view
    ('coronal', 'sagittal', or 'axial' (or 'transverse'). The GIF is then written to the specified output directory
    with a filename prefix specified by the user.
    
    .. important::
        
        Makes a GIF of the image in the space defined by the nifty file.

    Attributes:
        path_to_image (str): Path to the NIfTI image file.
        view (str): Specifies on which plane the NIfTI image is sliced.
        output_directory (str): Directory where the GIF should be written.
        prefix (str): Filename prefix for the GIF.
        fig (matplotlib.figure.Figure): Matplotlib figure object for the GIF.
        ax (matplotlib.axes.Axes): Matplotlib axes object for the GIF.
        im_kw (dict): Dictionary of arguments passed to Axes.imshow for the GIF.
        image (numpy.ndarray): 3D numpy array representing the NIfTI image.
        vmax (float): Maximum value for the plot's color limit.
        ani_image (matplotlib.image.AxesImage): An animated image rendered in the figure.
        cbar (matplotlib.colorbar): Colorbar associated with ani_image.

    Example:
        An example of how to initialize and use the NiftiGifCreator class:

        .. code-block:: python

            import petpal.image_visualization as pet_vis

            gif_creator = pet_vis.NiftiGifCreator(path_to_image="./input.nii.gz",
                                                  view="axial",
                                                  output_directory="./output",
                                                  output_filename_prefix="test")
            gif_creator.make_gif()
            gif_creator.write_gif()
    """
    def __init__(self,
                 path_to_image: str,
                 view: str,
                 output_directory: str,
                 output_filename_prefix: str = "",
                 fig_title: str = "Patlak-$K_i$ Parametric Image",
                 cbar_label: str = "$K_i$ (Infusion Rate)"):
        """
        Initialize the :class:`NiftiGifCreator`. Assigns instance variables and sets up figure for animation.

        Args:
            path_to_image (str): absolute path to the image file.
            view (str): desired view for the gif. Can be 'coronal', 'sagittal', 'axial'.
            output_directory (str): absolute path to the directory where gif is to be stored.
            output_filename_prefix (str, optional): a string to be prepended to the filename of the gif.
            fig_title (str, optional): Title of the figure where the images will be plotted.
            cbar_label (str, optional): Label for the colorbar of the figure.

        Raises:
            ValueError: If `view` is not 'coronal', 'sagittal' or 'axial'
        """
        
        self.view = view.lower()
        self._validate_view()
        
        self.path_to_image = os.path.abspath(path_to_image)
        self.output_directory = os.path.abspath(output_directory)
        self.prefix = output_filename_prefix
        
        self.image = nifty_loader(self.path_to_image).get_fdata()
        
        self.vmax = max(np.max(self.image), np.abs(np.min(self.image)))
        
        self.fig, self.ax = plt.subplots(1, 1, constrained_layout=True)
        self.ani = None
        self.cbar = None
        
        self.im_kw = {'origin'      : 'lower',
                     'cmap'         : 'bwr',
                     'vmin'         : -self.vmax / 3.,
                     'vmax'         : self.vmax / 3.,
                     'interpolation': 'none'}
        
        self.ani_image = self.make_first_frame()
        self.set_figure_title_and_labels(title=fig_title, cbar_label=cbar_label)
    
    def _validate_view(self):
        """
        Validate the value of ``view``. It must be one of ['coronal', 'sagittal', 'axial', 'x', 'y', 'z'].

        Raises:
            ValueError: If ``view`` is not 'coronal', 'sagittal', 'axial', 'x', 'y', 'z'
        """
        if self.view not in ['coronal', 'sagittal', 'axial', 'x', 'y', 'z', 'transverse']:
            raise ValueError("Invalid view. Please choose from 'coronal', 'sagittal', 'axial' (or 'transverse'), "
                             "'x', 'y', or 'z'.")
    
    def make_first_frame(self):
        """
        Makes the first frame of the animation by plotting an image on ``ax`` according to ``view``.

        Returns:
            im (matplotlib.image.AxesImage): Image plotted in the first frame.
            
        Side Effects:
            Modifies ``ani_image`` by creating the first frame of the GIF
        """
        if self.view in ['x', 'sagittal']:
            img = self.image[0, :, :].T
        elif self.view in ['y', 'coronal']:
            img = self.image[:, 0, :].T
        else:
            img = self.image[:, :, 0].T
        out_im = self.ax.imshow(img, **self.im_kw)
        
        return out_im
    
    def set_figure_title_and_labels(self, title: str, cbar_label: str):
        """
        Sets the title of the figure and labels of the color bar.

        Args:
            title (str): Title to be set for the figure.
            cbar_label (str): Label to be set for the color bar.
            
        Side Effects:
            - Modifies ``cbar`` with a new colorbar
            - Changes the title of ``fig``
            - Modifies the x and y axes of ``ani_image``
            
        """
        self.cbar = self.fig.colorbar(self.ani_image, ax=self.ax, shrink=1.0)
        self.cbar.set_label(cbar_label, rotation=270)
        self.fig.suptitle(title)
        self.ani_image.axes.get_xaxis().set_visible(False)
        self.ani_image.axes.get_yaxis().set_visible(False)
        
    def update_frame(self, i):
        """
        Update function for matplotlib.animation.FuncAnimation. Updates the plotted image on the current axes.

        Args:
            i (int): Index of the current frame

        Returns:
            Tuple containing ``im`` (matplotlib.image.AxesImage), which is updated to the i-th frame according to
            ``view``.
            
        Side Effects:
            Modifies ``ani_image`` by updating its frame data
            
        """
        if self.view in ['x', 'sagittal']:
            img = self.image[i, :, :].T
        elif self.view in ['y', 'coronal']:
            img = self.image[:, i, :].T
        else:
            img = self.image[:, :, i].T
        
        self.ani_image.set_data(img)
        
        return self.ani_image,
    
    def make_gif(self, frames: Iterable = None):
        """
        Makes the GIF using FuncAnimation from matplotlib.animation module.

        Args:
            frames (Iterable, optional): Iterable of frame indices to be included in the GIF. If not provided, all
                frames will be used.
            
        Side Effects:
            Modifies ``ani`` by creating a new :class:`matplotlib.animation.FuncAnimation` object
        """
        if frames is None:
            tot_dims = self.image.shape
            if self.view in ['x', 'sagittal']:
                num_frames = tot_dims[0]
            elif self.view in ['y', 'coronal']:
                num_frames = tot_dims[1]
            else:
                num_frames = tot_dims[2]
            frames = range(1, num_frames)
        
        self.ani = mpl_animation.FuncAnimation(fig=self.fig,
                                               func=self.update_frame,
                                               frames=frames,
                                               blit=True)
    
    def write_gif(self, fps: int = 15):
        """
        Writes the GIF to the output directory with filename ``{prefix}_view-{view}.gif``.
        
        Side Effects:
            - Creates a GIF file at the path specified by ``output_directory`` and ``prefix``
            - Closes the ``fig`` matplotlib figure
            
        """
        out_path = os.path.join(self.output_directory, f'{self.prefix}_view-{self.view}.gif')
        self.ani.save(f"{out_path}", fps=fps, writer='pillow', dpi=100)
        plt.close(self.fig)
        