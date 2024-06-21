"""
Module for generating quality control plots.
"""
import numpy as np
import pandas as pd
import seaborn as sns


def motion_plot(framewise_displacement: np.ndarray,
                output_plot: str=None):
    """
    Plots the quantity of motion estimated by :meth:`ants.motion_correction`.
    Takes the framewise displacement returned by ANTs, and plots this quantity
    against frame number.

    Args:
        framewise_displacement (np.ndarray): Total movement, or displacement,
            estimated between consecutive frames.
        output_plot (str): File to which plot is saved. If `None`, do not write
            plot to file. Default value `None`.
    
    Returns:
        movement_plot (sns.lineplot): Plot of total movement between frames.
    """
    movement_dataframe = pd.DataFrame(columns=['Frame','Framewise Displacement (mm)'])
    movement_dataframe['Frame'] = np.arange(len(framewise_displacement))
    movement_dataframe['Framewise Displacement (mm)'] = framewise_displacement
    movement_plot = sns.lineplot(data=movement_dataframe,x='Frame',y='Framewise Displacement (mm)')

    if output_plot is None:
        return movement_plot
    movement_plot.get_figure().savefig(output_plot)
    return movement_plot
