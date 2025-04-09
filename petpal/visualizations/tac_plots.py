"""
Module to plot TACs.

Key features:
    * :class:`TacFigure` Handles plotting TACs on a figure.
    * :class:`RegionalTacFigure` Extends :class:`TacFigure` to handle plotting TACs from a regional
        analysis.

"""
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from ..utils.time_activity_curve import MultiTACAnalysisMixin


class TacFigure:
    r"""
    A class for plotting Time Activity Curves (TACs) on linear and semi-logarithmic scales.

    This class simplifies the process of comparing TACs on different scales. It generates a
    side-by-side plot with a linear-linear scale for the first plot and a log-x scale for the
    second plot. Users can add TACs to the plots and optionally generate a legend.

    Attributes:
        fig (matplotlib.figure.Figure): The figure object that contains the plots.
        axes (ndarray of Axes): The axes objects where the TACs are plotted.

    Example:

    .. code-block:: python

        tac_plots = TacFigure()
        tac_plots.add_tac(tac_times_in_minutes, tac_vals, label='TAC 1', color='blue')
        tac_plots.add_tac(tac_times_2, tac_vals_2, label='TAC 2', color='red')
        tac_plots.gen_legend()
        plt.show()

    """
    def __init__(self,
                 figsize: tuple = (8, 4),
                 xlabel: str = r'$t$ [minutes]',
                 ylabel: str = r'TAC [$\mathrm{kBq/ml}$]',
                 plot_type: str='both'):
        r"""
        Initialize the TacFigure with two subplots, one with a linear scale and the other with a
        semi-logarithmic scale.

        Args:
            figsize (tuple): The total size of the figure. Defaults to an 8x4 inches figure.
            xlabel (str): The label for the x-axis. Defaults to '$t$ [minutes]'.
            ylabel (str): The label for the y-axis. Defaults to 'TAC [$\mathrm{kBq/ml}$]'.
            plot_type (str): Type of plot, with options 'linear', 'log', or 'both'.
        """
        if plot_type=='both':
            self.setup_linear_and_log_subplot(xlabel=xlabel, ylabel=ylabel, figsize=figsize)
        elif plot_type=='linear':
            self.setup_linear_subplot(xlabel=xlabel, ylabel=ylabel, figsize=figsize)
        elif plot_type=='log':
            self.setup_log_subplot(xlabel=xlabel, ylabel=ylabel, figsize=figsize)
        else:
            raise ValueError(f"Got unexpected plot_type {plot_type}, "
                             "expected one of: 'linear', 'log', 'both'.")

    def setup_linear_subplot(self, xlabel: str, ylabel: str, figsize: tuple):
        """
        Get the figure and axes objects for a 1x1 MatPlotLib subplot.

        Args:
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            figsize (tuple): Size of the figure.
        """
        subplot = plt.subplots(1, 1, sharey=True, constrained_layout=True, figsize=figsize)
        self.fig, self.axes = subplot
        self.fax = [self.axes]
        _xlabel_set = [ax.set(xlabel=xlabel) for ax in self.fax]
        self.fax[0].set(ylabel=ylabel, title='Linear')


    def setup_log_subplot(self, xlabel: str, ylabel: str, figsize: tuple):
        """
        Get the figure and axes objects for a 1x1 MatPlotLib subplot.

        Args:
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            figsize (tuple): Size of the figure.
        """
        subplot = plt.subplots(1, 1, sharey=True, constrained_layout=True, figsize=figsize)
        self.fig, self.axes = subplot
        self.fax = [self.axes]
        _xlabel_set = [ax.set(xlabel=xlabel) for ax in self.fax]
        self.fax[0].set(xscale='log',ylabel=ylabel, title='SemiLog-X')


    def setup_linear_and_log_subplot(self, xlabel: str, ylabel: str, figsize: tuple):
        """
        Get the figure and axes objects for a 1x2 MatPlotLib subplot.

        Args:
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            figsize (tuple): Size of the figure.
        """
        subplot = plt.subplots(1, 2, sharey=True, constrained_layout=True, figsize=figsize)
        self.fig, self.axes = subplot
        self.fax = self.axes.flatten()
        _xlabel_set = [ax.set(xlabel=xlabel) for ax in self.fax]
        self.fax[0].set(ylabel=ylabel, title='Linear')
        self.fax[1].set(xscale='log', title='SemiLog-X')


    def add_tac(self, tac_times: np.ndarray, tac_vals: np.ndarray, **kwargs):
        r"""
        Add a TAC to both subplots.

        Args:
            tac_times (np.ndarray): The time points for the TAC.
            tac_vals (np.ndarray): The corresponding values for the TAC.
            kwargs (dict): Additional keyword arguments for the plot() function.
        """
        return [ax.plot(tac_times, tac_vals, **kwargs) for ax in self.fax]


    def add_errorbar(self,
                     tac_times: np.ndarray,
                     tac_vals: np.ndarray,
                     uncertainty: np.ndarray,
                     **kwargs):
        """
        Add a lineplot with errorbars to the figure.

        Args:
            tac_times (np.ndarray): Array containing times at which TAC is sampled.
            tac_vals (np.ndarray): Array containing TAC activity concentration.
            uncertainty (np.ndarray): Array containing uncertainties in TAC measurements.
            kwargs (dict): Additional keyword arguments for the plt.errorbar() function.
        """
        return [ax.errorbar(tac_times, tac_vals, yerr=uncertainty, **kwargs) for ax in self.fax]


    def normalize_yaxis(self, **kwargs):
        r"""
        Set the y-axis lower limit to zero.

        Args:
            kwargs (dict): Additional keyword arguments for the set_ylim() function.
        """
        return [ax.set_ylim(0, None, **kwargs) for ax in self.fax]


    def gen_legend(self):
        r"""
        Generate a legend using the labels provided in the add_tac() method.

        Note:
            It is recommended to add all TACs before generating the legend. Any TACs added after
        the legend is generated will not be included in the legend.

        """
        handles, labels = self.fax[0].get_legend_handles_labels()
        if handles:
            self.fax[-1].legend(handles, labels, bbox_to_anchor=(1.0, 0.5), loc='center left')


    def write_fig(self, out_fig_path: str, **kwargs):
        """
        Write the figure to a common image file type, such as .png or .jpg. Applies
        :meth:`plt.savefig` to figure object.

        Args:
            out_fig_path (str): Path to the file the figure will be written to.
            kwargs (dict): Additional key word arguments passed to plt.savefig()
        """
        self.fig.savefig(fname=out_fig_path, **kwargs)


class RegionalTacFigure(TacFigure,MultiTACAnalysisMixin):
    """
    Handle plotting regional TACs generated with PETPAL.
    """
    def __init__(self,
                 tacs_dir: str,
                 figsize: tuple = (8, 4),
                 xlabel: str = r'$t$ [minutes]',
                 ylabel: str = r'TAC [$\mathrm{kBq/ml}$]',
                 plot_type='both'):
        MultiTACAnalysisMixin.__init__(self,input_tac_path='',tacs_dir=tacs_dir)
        TacFigure.__init__(self,figsize=figsize,xlabel=xlabel,ylabel=ylabel,plot_type=plot_type)


    @property
    def tacs_objects_dict(self):
        """
        A dictionary containing region name/TAC object pairs for all TAC files found in tacs_dir.
        """
        return self.get_tacs_objects_dict_from_dir(self.tacs_dir)


    def plot_tacs_in_regions_list(self,
                                  regions: list[str],
                                  show_legend: bool=True,
                                  colormap: str='Dark2',
                                  **kwargs):
        """
        Plot TACs for a list of provided regions. Region names correspond to abbreviated segment
        names in the dseg file used to generate the regions.

        Args:
            regions (list[str]): A list of region names whose TACs are plotted.
            show_legend (bool): Show the legend with region names in the resulting figure. Default
                True.
            colormap (str): A matplotlib color map used to select colors of different TAC plots.
                Default 'Dark2'.
            kwargs (dict): Additional keyword arguments for the plt.errorbar() function.
        """
        colors = colormaps[colormap].colors
        tacs_obj_dict = self.tacs_objects_dict
        for i, region in enumerate(regions):
            tac = tacs_obj_dict[region]
            self.add_errorbar(tac_times=tac.times,
                              tac_vals=tac.activity,
                              uncertainty=tac.uncertainty,
                              label=region,
                              color=colors[i%len(colors)],
                              **kwargs)
        if show_legend:
            self.gen_legend()
        self.normalize_yaxis()
        return self.fig


    def plot_all_regional_tacs(self,show_legend: bool=True, colormap='Dark2', **kwargs):
        """
        Plot TACs for all TACs found in a folder. Region names correspond to abbreviated segment
        names in the dseg file used to generate the regions.

        Args:
            show_legend (bool): Show the legend with region names in the resulting figure. Default
                True.
            colormap (str): A matplotlib color map used to select colors of different TAC plots.
                Default 'Dark2'.
            kwargs (dict): Additional keyword arguments for the plt.errorbar() function.
        """
        tacs_obj_dict = self.tacs_objects_dict
        regions = list(tacs_obj_dict.keys())
        self.plot_tacs_in_regions_list(regions=regions,
                                       show_legend=show_legend,
                                       colormap=colormap,
                                       **kwargs)
        return self.fig
