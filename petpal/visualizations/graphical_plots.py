"""
This module contains the plotting functionalities for the project. It contains various classes that handle different
types of plots.

Classes defined in this module are:
    - :class:`GraphicalAnalysisPlot`: It handles various parameters for graphical plot analysis. Provides
      functionalities for generating figures, adding plot data, shading, and more.
    - :class:`PatlakPlot`: This class is responsible for plotting functionalities specific to Patlak method. It includes
      methods for calculating indexes, generating labels, and more.
    - :class:`LoganPlot`: This class handles plotting functionalities for the Logan method. It also includes methods for
      calculating indexes, generating labels, etc.
    - :class:`AltLoganPlot`: This class is primarily for enhancing the LoganPlot with additional or alternate features.
    - :class:`Plot`: A general plot class that handles common plot factors such as output directory, method name, ROI
      TAC paths, etc. It includes utilities for validating file path, directory and more.

Functions and methods in this module use :mod:`matplotlib` and :mod:`numpy` packages for creating and manipulating
plots.

Please refer to the documentation of each class for more detailed information.
"""

from abc import ABC, abstractmethod
import os
from typing import Dict, Union, Type
from matplotlib import pyplot as plt
import numpy as np
from ..kinetic_modeling import graphical_analysis as pet_grph
from ..utils.image_io import safe_load_tac


class GraphicalAnalysisPlot(ABC):
    """
    This is an abstract base class designed for creating customizable plots for graphical analysis.
    It takes Time-Activity Curves (TACs) as input and creates various plots (x vs. y, fit lines, fit area
    shading plots, and fit points) based on user input. This class also calculates fit parameters for the plotting data
    and determines relevant indices based on given thresholds.

    Attributes:
        pTAC (np.ndarray): The Input/Plasma TAC, an array containing time points and corresponding activity.
        tTAC (np.ndarray): The Tissue/Region TAC, an array with time points and corresponding activity.
        t_thresh_in_mins (float): The threshold time, in minutes, to consider in the analysis. Points are fit after this
            threshold.
        fig (plt.Figure): A matplotlib Figure object where the plots will be drawn.
        ax_list (list): A list of matplotlib Axes associated with `fig` where the plots will be drawn.
        non_zero_idx (np.ndarray): Indexes of non-zero values in appropriate TAC (calculated in specific
            implementations).
        t_thresh_idx (int): The index at which the time threshold is crossed in the TACs (calculated in specific
            implementations).
        x (np.ndarray): The "x" valu0es for plotting (calculated in specific implementations).
        y (np.ndarray): The "y" values for plotting (calculated in specific implementations).
        fit_params (dict): The parameters fit to the data using least squares. Contains 'slope', 'intercept', and
            'r_squared' (calculated in specific implementations).

    .. important::

        This is an abstract class and should be inherited by a concrete class that implements the following methods:
            * :func:`calculate_valid_indicies_and_x_and_y`
            * :func:`generate_label_from_fit_params`
            * :func:`add_figure_axes_labels_and_legend`

    """

    def __init__(self, pTAC: np.ndarray, tTAC: np.ndarray, t_thresh_in_mins: float, figObj: plt.Figure = None):
        """
        Initialize an instance of the GraphicalAnalysisPlot class.

        The instance is initialized with two Time-Activity Curves (TACs), a threshold time, and an optional matplotlib
        Figure. It calculates valid indices (where the denominator is non-zero for the particular analysis), 'x' and 'y'
        values for plotting based on the TACs and the threshold time, and also analyzes the TACs to generate the fits.

        Args:
            pTAC (np.ndarray): The input Time Activity Curve, an array containing time points and corresponding activity.
            tTAC (np.ndarray): The Tissue or Region Time Activity Curve, an array with time points and corresponding
                activity.
            t_thresh_in_mins (float): The threshold time in minutes to consider when performing calculations for the
                plots.
            figObj (plt.Figure, optional): An optional matplotlib Figure object. If not provided, a new Figure object is
                created.

        Raises:
            matplotlib error: Error handling for the plot generation is managed by matplotlib. Any exceptions thrown
                during plotting are handled by the :mod:`matplotlib` internally.
        """
        self.pTAC: np.ndarray = pTAC[:]
        self.tTAC: np.ndarray = tTAC[:]
        self.t_thresh_in_mins: float = t_thresh_in_mins
        self.fig, self.ax_list = self.generate_figure_and_axes(figObj=figObj)
        self.non_zero_idx: np.ndarray = None
        self.t_thresh_idx: int = None
        self.x: np.ndarray = None
        self.y: np.ndarray = None
        self.fit_params: Dict = None
        self.calculate_valid_indicies_and_x_and_y()
        self.calculate_fit_params()

    @staticmethod
    def generate_figure_and_axes(figObj: plt.Figure = None):
        """
        Generate a matplotlib Figure and Axes for plotting.

        A new Figure and Axes are created if no Figure object is provided. If a Figure object is provided, the method
        retrieves the existing axes from the Figure object. In either case, the method returns the Figure and a list of
        its Axes.

        Args:
            figObj (plt.Figure, optional): An optional matplotlib Figure object. If not provided, a new Figure object is
                created with 2 subplots arranged in 1 row, a figure size of 8x4, line width of 3.0, and edge color 'k'.

        Returns:
            fig (plt.Figure): The resulting matplotlib Figure object.
            ax_list (list): A list of Axes objects associated with 'fig'.

        Raises:
            None
        """
        if figObj is None:
            fig, ax_list = plt.subplots(1, 2,
                                        constrained_layout=True, figsize=[8, 4],
                                        linewidth=3.0, edgecolor='k')
            ax_list = ax_list.flatten()
        else:
            fig = figObj
            ax_list = fig.get_axes()
        return fig, ax_list

    def add_data_plots(self, pl_kwargs: dict = None):
        """
        Add data plots to the Axes in the instance's Axes list.

        This method plots the instance's :math:`x` and :math:`y` values (from the particular analysis) on each Axes in
        the instance's Axes list. The style of the plotted points can be customized by passing a dictionary of keyword
        arguments for `matplotlib.pyplot.plot`.

        Args:
            pl_kwargs (dict, optional): A dictionary of keyword arguments to be passed to `matplotlib.pyplot.plot` for
                styling the points. If not provided, the points are plotted with ``linewidth=1``, ``alpha=0.9``,
                ``markersize=8``, ``markerstyle='.'``, ``zorder=1``, and ``color='black'``.

        Raises:
            ValueError: If pl_kwargs contains an argument not supported by ``matplotlib.pyplot.plot``.
        """
        if pl_kwargs is None:
            for ax in self.ax_list:
                ax.plot(self.x, self.y, lw=1, alpha=0.9, ms=8, marker='.', zorder=1, color='black')
        else:
            for ax in self.ax_list:
                ax.plot(self.x, self.y, **pl_kwargs)

    def add_shading_plots(self, pl_kwargs: dict = None):
        """
        Add shaded regions to the Axes in the instance's Axes list.

        This method shades a vertical span on each Axes in the instance's Axes list, from the x-value at the threshold
        index to the last x-value. The style of the shaded region can be customized by passing a dictionary of keyword
        arguments for :func:`matplotlib.pyplot.axvspan`.

        Args:
            pl_kwargs (dict, optional): A dictionary of keyword arguments to be passed to `matplotlib.pyplot.axvspan`
                for styling the shaded region. If not provided, the region is shaded with ``color='gray'``,
                ``alpha=0.2``, and ``zorder=0``.

        Raises:
            ValueError: If pl_kwargs contains an argument not supported by :func:`matplotlib.pyplot.axvspan`.
        """
        x_lo, x_hi = self.x[self.t_thresh_idx], self.x[-1]
        if pl_kwargs is None:
            for ax in self.ax_list:
                ax.axvspan(x_lo, x_hi, color='gray', alpha=0.2, zorder=0)
        else:
            for ax in self.ax_list:
                ax.axvspan(x_lo, x_hi, **pl_kwargs)

    def add_fit_points(self, pl_kwargs: dict = None):
        """
        Add fit points to the Axes in the instance's Axes list.

        This method plots the instance's :math:`x` and :math:`y` values that were used in fitting, i.e., the points past
        the threshold index, on each Axes in the instance's Axes list. The style of the plotted points can be customized
        by passing a dictionary of keyword arguments for :func:`matplotlib.pyplot.plot`.

        Args:
            pl_kwargs (dict, optional): A dictionary of keyword arguments to be passed to :func:`matplotlib.pyplot.plot`
                for styling the points. If not provided, the points are plotted as blue circles with ``alpha=0.9``,
                ``markersize=5``, and ``zorder=2``.

        Raises:
            ValueError: If pl_kwargs contains an argument not supported by :func:`matplotlib.pyplot.plot`.
        """
        t_thresh = self.t_thresh_idx
        if pl_kwargs is None:
            for ax in self.ax_list:
                ax.plot(self.x[t_thresh:], self.y[t_thresh:], 'o', alpha=0.9, ms='5', zorder=2, color='blue')
        else:
            for ax in self.ax_list:
                ax.plot(self.x[t_thresh:], self.y[t_thresh:], **pl_kwargs)

    def add_fit_lines(self, pl_kwargs: dict = None):
        """
        Add line of best fit to the Axes in the instance's Axes list.

        This method plots the line of best fit based on the instance's fit parameters (slope and intercept) on each
        Axes in the instance's Axes list. The style of the line can be customized by passing a dictionary of keyword
        arguments for :func:`matplotlib.pyplot.plot`.

        Args:
            pl_kwargs (dict, optional): A dictionary of keyword arguments to be passed to :func:`matplotlib.pyplot.plot`
                for styling of the line. If not provided, the line is plotted as an orange solid line with line
                ``linewidth=2.5``, ``zorder=3``, and labelled using the :meth:`generate_label_from_fit_params` method.

        Raises:
            ValueError: If pl_kwargs contains an argument not supported by :func:`matplotlib.pyplot.plot`.
        """
        y = self.x * self.fit_params['slope'] + self.fit_params['intercept']
        if pl_kwargs is None:
            for ax in self.ax_list:
                ax.plot(self.x, y, '-', color='orange', lw=2.5,
                        zorder=3, label=self.generate_label_from_fit_params())
        else:
            for ax in self.ax_list:
                ax.plot(self.x, y, **pl_kwargs)

    def add_plots(self,
                  plot_data: bool = True,
                  plot_fit_points: bool = True,
                  plot_fit_lines: bool = True,
                  fit_shading: bool = True,
                  data_kwargs: dict = None,
                  points_kwargs: dict = None,
                  line_kwargs: dict = None,
                  shading_kwargs: dict = None):
        """
        Generate and add various types of plots to the Axes in the instance's Axes list.

        Depending on the boolean values of ``plot_data``, ``plot_fit_points``, ``plot_fit_lines``, and ``fit_shading``,
        this method determines which types of plots to generate and include. This includes the data points plot,
        fitting points, line of best fit, and shading between the line of best fit and the x-axis.

        Args:
            plot_data (bool, optional): Determines if the analysis :math:`x` and :math:`y` points should be plotted.
            Defaults to True.
            plot_fit_points (bool, optional): Determines if the fit points should be plotted. Defaults to True.
            plot_fit_lines (bool, optional): Determines if the line of best fit should be plotted. Defaults to True.
            fit_shading (bool, optional): Determines if shading should be applied to the plots. Defaults to True.
            data_kwargs (dict, optional): Keyword arguments for styling the data plot.
            points_kwargs (dict, optional): Keyword arguments for styling the fit points plot.
            line_kwargs (dict, optional): Keyword arguments for styling the line of best fit.
            shading_kwargs (dict, optional): Keyword arguments for styling the shading.

        Raises:
            ValueError: If any of the keyword argument dictionaries contain unsupported arguments.

        See Also:
            * :meth:`add_data_plots`: Method used to plot the :math:`x` and :math:`y` data.
            * :meth:`add_fit_points`: Method used to plot the fit points.
            * :meth:`add_fit_lines`: Method used to plot the line of best fit.
            * :meth:`add_shading_plots`: Method used to add shading to the plots.

        """
        if plot_data:
            self.add_data_plots(pl_kwargs=data_kwargs)
        if plot_fit_points:
            self.add_fit_points(pl_kwargs=points_kwargs)
        if plot_fit_lines:
            self.add_fit_lines(pl_kwargs=line_kwargs)
        if fit_shading:
            self.add_shading_plots(pl_kwargs=shading_kwargs)

    def generate_figure(self,
                        plot_data: bool = True,
                        plot_fit_points: bool = True,
                        plot_fit_lines: bool = True,
                        fit_shading: bool = True,
                        data_kwargs: dict = None,
                        points_kwargs: dict = None,
                        line_kwargs: dict = None,
                        shading_kwargs: dict = None):
        """
        Generates the complete figure for graphical analysis.

        This function is the preferred way for a user to interact with this class. It sets up and adds the desired plots
        to the figure, adds the labels and legend, and sets the title and scale for the plots. By default, we generate
        a figure where each panel has the analysis :math:`x` and :math:`y` data, the points used for the fitting, a
        shading over the region for the fit, and the fit line.

        Args:
            plot_data (bool, optional): Determines if the :math:`x` and :math:`y` data points should be plotted.
            Defaults to True.
            plot_fit_points (bool, optional): Determines if the fit points should be plotted. Defaults to True.
            plot_fit_lines (bool, optional): Determines if the line of best fit should be plotted. Defaults to True.
            fit_shading (bool, optional): Determines if shading should be applied to the plots. Defaults to True.
            data_kwargs (dict, optional): Keyword arguments for styling the data plot.
            points_kwargs (dict, optional): Keyword arguments for styling the fit points plot.
            line_kwargs (dict, optional): Keyword arguments for styling the line of best fit.
            shading_kwargs (dict, optional): Keyword arguments for styling the shading.

        See Also:
            * :func:`add_plots`: Composite function that adds different types of plots.
            * :func:`add_figure_axes_labels_and_legend`: Adds labels and a legend to the figure.
              (To be implemented in a specific class)

        """
        self.add_plots(plot_data=plot_data,
                       plot_fit_points=plot_fit_points,
                       plot_fit_lines=plot_fit_lines,
                       fit_shading=fit_shading,
                       data_kwargs=data_kwargs,
                       points_kwargs=points_kwargs,
                       line_kwargs=line_kwargs,
                       shading_kwargs=shading_kwargs)
        self.add_figure_axes_labels_and_legend()
        self.ax_list[0].set_title("Linear Plot")
        self.ax_list[1].set_title("LogLog Plot")

        self.ax_list[1].set(yscale='log', xscale='log')

    def calculate_fit_params(self):
        """
        Calculates the parameters (slope, intercept, r_squared) for line fitting.

        This function fits a line to the :math:`x` and :math:`y` values that are generated, beyond the provided threshold.
        The fit line is computed using Least Square fitting with R-squared method.

        See Also:
            * :func:`calculate_valid_indicies_and_x_and_y`: Method used to generate the x and y values.
            * :func:`petpal.graphical_analysis.fit_line_to_data_using_lls_with_rsquared`: Method used to fit a line to
              data points using Least Square fitting with R-squared value.

        """
        t_thresh = self.t_thresh_idx
        fit_params = pet_grph.fit_line_to_data_using_lls_with_rsquared(xdata=self.x[t_thresh:], ydata=self.y[t_thresh:])

        fit_params = {
            'slope': fit_params[0],
            'intercept': fit_params[1],
            'r_squared': fit_params[2]
        }
        self.fit_params = fit_params

    @abstractmethod
    def calculate_valid_indicies_and_x_and_y(self) -> None:
        """
        Abstract method for calculating the :math:`x` and :math:`y` values given a particular analysis in a concrete
        class, valid indices where the denominator is 0, and the index corresponding to the provided threshold.

        This is a two-fold method:
        1. Calculates valid x and y values derived from the analysis where each concrete class represents a different
        method. Valid points are those where the denominator (which varies by analysis type) is non-zero.
        2. Saves the indices of the non-zero valid values and the indices that correspond to the given t_thresh_in_mins.

        .. important::
            This abstract method must be implemented in each subclass representing a different analysis method.

        Note:
            The implementation of this function in subclasses MUST calculate and assign values to ``x``, ``y``,
            ``t_thresh_idx``, and ``non_zero_idx`` attributes.

        Raises:
            NotImplementedError: This method needs to be implemented in a subclass.

        Example Implementation:
            :meth:`PatlakPlot.calculate_valid_indicies_and_x_and_y`: This class provides an example implementation of
            this method in a concrete subclass.

        """
        raise NotImplementedError("This method must be implemented in a concrete class.")

    @abstractmethod
    def generate_label_from_fit_params(self) -> str:
        """
        Abstract method to generate a label string from fitting parameters.

        This function reads the fitting parameters 'slope', 'intercept', and 'r_squared' from the class attribute
        'fit_params', and formats them into a string that can be used as a label on plots or reports.

        .. important::
            This abstract method must be implemented in each subclass representing a different analysis method.

        Raises:
            NotImplementedError: This method needs to be implemented in a subclass.

        Example Implementation:
            :meth:`PatlakPlot.generate_label_from_fit_params`: This method provides an example implementation of this
            abstract method in a concrete subclass. Similarly, :meth:`LoganPlot.generate_label_from_fit_params`.

        """
        raise NotImplementedError("This method must be implemented in a concrete class.")

    @abstractmethod
    def add_figure_axes_labels_and_legend(self):
        """
        Abstract method for adding analysis specific titles, axes labels and a legend to the figure.

        .. important::
            This abstract method must be implemented in each subclass representing a different analysis method.

        Raises:
            NotImplementedError: This method must be implemented in each subclass.

        Example Implementation:
            For an example of implementing this method, see :meth:`PatlakPlot.add_figure_axes_labels_and_legend`,
            or :meth:`LoganPlot.add_figure_axes_labels_and_legend`.

        """
        raise NotImplementedError("This method must be implemented in a concrete class.")


class PatlakPlot(GraphicalAnalysisPlot):
    r"""
    This class handles generation of Patlak plots for PET analysis.

    The :class:`PatlakPlot` class is designed to process PET data and display it in the form of Patlak plots. The class
    handles data processing including computation of valid indices and calculation of plot-specific parameters, setting
    of plot labels and legends, as well as the actual plotting of data.

    The processing steps involve computations such as handling of non-zero indices in the plasma time-activity curve
    (pTAC), calculation of x and y coordinates for the Patlak plot, and generation of plot labeling information.

    The class also provides a capability to add labels, legends and title to the figure  to present the data in a
    meaningful way.

    Note:
        The class inherits from the :class:`GraphicalAnalysisPlot` abstract base class.

    Example:
        In the proceeding examples, ``pTAC`` represent the plasma TAC (or input TAC) and ``tTAC`` represents the tissue TAC.

        For the quickest way to generate the Patlak Plot, we just instantiate the class and run the :meth:`generate_figure`
        method. This will generate 2 plots in a row. The first one having a linear-linear scale, and the second one
        having a log-log scale.

        .. code-block:: python

            from petpal.graphical_plots import PatlakPlot
            patlak_plot = PatlakPlot(tTAC=tTAC, pTAC=pTAC, t_thresh_in_mins=45.0)
            patlak_plot.generate_figure()
            plt.show() # Or use plt.savefig() to save the figure.

        If the default styling needs to be changed, we can pass keyword arguments to each of the plotting methods:

        .. code-block:: python

            from petpal.graphical_plots import PatlakPlot
            patlak_plot = PatlakPlot(tTAC=tTAC, pTAC=pTAC, t_thresh_in_mins=45.0)
            patlak_plot.generate_figure(line_kwargs=dict(lw=2, alpha=0.95, color='red', label=patlak_plot.generate_label_from_fit_params()),
                                        shading_kwargs=dict(color='palegreen', alpha=0.2),
                                        data_kwargs=dict(alpha=0.85, color='k', marker='.'))

    See Also:
            * :class:`LoganPlot`
            * :class:`AltLoganPlot`

    """

    def calculate_valid_indicies_and_x_and_y(self) -> None:
        r"""
        Calculates the valid indices along with :math:`x` and :math:`y` for Patlak plot analysis.

        This method performs the computation for the non-zero indices in the provided plasma time-activity curve (pTAC).
        It further calculates the values of :math:`x` and :math:`y` used in Patlak analysis based on these non-zero
        indices. This is done to avoid singularities caused by zero denominators. The Patlak :math:`x` and :math:`y`
        values are:

        .. math::

            y&= \frac{R(t)}{C_\mathrm{P}(t)}\\
            x&= \frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{C_\mathrm{P}(t)},

        where :math:`C_\mathrm{P}` is the input function and :math:`R(t)` is PET activity in the particular region of
        interest.

        The method updates the instance variables ``x``, ``y``, ``non_zero_idx``, and ``t_thresh_idx``.

        Returns:
            None
        """
        non_zero_indices = np.argwhere(self.pTAC[1] != 0.0).T[0]
        t_thresh = pet_grph.get_index_from_threshold(times_in_minutes=self.pTAC[0][non_zero_indices],
                                                     t_thresh_in_minutes=self.t_thresh_in_mins)

        x = pet_grph.cumulative_trapezoidal_integral(xdata=self.pTAC[0], ydata=self.pTAC[1])
        x = x[non_zero_indices] / self.pTAC[1][non_zero_indices]
        y = self.tTAC[1][non_zero_indices] / self.pTAC[1][non_zero_indices]

        self.x = x[:]
        self.y = y[:]
        self.non_zero_idx = non_zero_indices[:]
        self.t_thresh_idx = t_thresh
        return None

    def generate_label_from_fit_params(self) -> str:
        r"""
        Creates a label string from the fit parameters for graphical presentation.

        This method retrieves slope, intercept, and R-squared values from the instance's
        fit parameters, and then formats these values into a string that is LaTeX compatible
        for later rendering inside a plot's label. For example:

        .. math::

            K_{1}&=0.1\\
            V_{\mathrm{T}}&=0.2\\
            R^{2}&=0.95

        Returns:
            str: The created label. Each parameter is formatted as a separate line.
        """
        slope = self.fit_params['slope']
        intercept = self.fit_params['intercept']
        r_sq = self.fit_params['r_squared']

        return f"$K_1=${slope:<5.3f}\n$V_\mathrm{{T}}=${intercept:<5.3f}\n$R^2=${r_sq:<5.3f}"

    def add_figure_axes_labels_and_legend(self):
        r"""
        Adds labels and a legend to the axes of the figure.

        This method sets the `x_label` and `y_label` for all axes on the figure. It also adds a legend to the figure,
        which is anchored to the upper left corner. Lastly, we also give a title to the figure: Patlak Plots.

        The labels are set to:

        .. math::

           \begin{align*}
           y&= \frac{R(t)}{C_\mathrm{P}(t)}\\
           x&= \frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{C_\mathrm{P}(t)},
           \end{align*}

        where :math:`C_\mathrm{P}` is the input function and :math:`R(t)` is PET activity in the particular region of
        interest.

        See Also:
            * :meth:`calculate_valid_indicies_and_x_and_y` for the calculation implementation.

        """
        x_label = r"$\frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{C_\mathrm{P}(t)}$"
        y_label = r"$\frac{R(t)}{C_\mathrm{P}(t)}$"
        for ax in self.ax_list:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        self.fig.legend(*self.ax_list[0].get_legend_handles_labels(),
                        bbox_to_anchor=(1.0, 0.8),
                        loc='upper left',
                        title='Patlak Analysis')
        self.fig.suptitle("Patlak Plots")


class LoganPlot(GraphicalAnalysisPlot):
    r"""
    This class handles generation of Logan plots for PET analysis.

    The :class:`LoganPlot` class is designed to process PET data and display it in the form of Logan plots.
    The class handles data processing including computation of valid indices and calculation of plot-specific
    parameters, setting of plot labels and legends, as well as the actual plotting of data.

    The processing steps involve computations such as handling of non-zero indices in the plasma time-activity curve
    (pTAC), calculation of x and y coordinates for the Logan plot, and generation of plot labeling information.

    The class also provides a capability to add labels, legends and title to the figure to present the data in a
    meaningful way.

    Note:
        The class inherits from the :class:`GraphicalAnalysisPlot` abstract base class.

    Example:
        In the proceeding examples, ``pTAC`` represent the plasma TAC (or input TAC) and ``tTAC`` represents the tissue
        TAC.

        For the quickest way to generate the Logan Plot, we just instantiate the class and run the :meth:`generate_figure`
        method. This will generate 2 plots in a row. The first one having a linear-linear scale, and the second one
        having a log-log scale.

        .. code-block:: python

            from petpal.graphical_plots import LoganPlot
            logan_plot = LoganPlot(tTAC=tTAC, pTAC=pTAC, t_thresh_in_mins=45.0)
            logan_plot.generate_figure()
            plt.show() # Or use plt.savefig() to save the figure.

        If the default styling needs to be changed, we can pass keyword arguments to each of the plotting methods:

        .. code-block:: python

            from petpal.graphical_plots import LoganPlot
            logan_plot = LoganPlot(tTAC=tTAC, pTAC=pTAC, t_thresh_in_mins=45.0)
            logan_plot.generate_figure(line_kwargs=dict(lw=2, alpha=0.95, color='red', label=patlak_plot.generate_label_from_fit_params()),
                                        shading_kwargs=dict(color='palegreen', alpha=0.2),
                                        data_kwargs=dict(alpha=0.85, color='k', marker='.'))

    See Also:
        * :class:`PatlakPlot`
        * :class:`AltLoganPlot`

    """

    def calculate_valid_indicies_and_x_and_y(self) -> None:
        r"""Calculates the valid indices along with :math:`x` and :math:`y` for Logan plot analysis.

        This method performs the computation for the non-zero indices in the provided region time-activity curve (tTAC).
        It further calculates the values of :math:`x` and :math:`y` used in Logan analysis based on these non-zero
        indices. This is done to avoid singularities caused by zero denominators. The Logan :math:`x` and :math:`y`
        values are:

        .. math::

            y&= \frac{\int_{0}^{t}R(s)\mathrm{d}s}{R(t)}\\
            x&= \frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{R(t)},

        where :math:`C_\mathrm{P}` is the input function and :math:`R(t)` is PET activity in the particular region of
        interest.

        The method updates the instance variables ``x``, ``y``, ``non_zero_idx``, and ``t_thresh_idx``.

        Returns:
            None

        """
        non_zero_indices = np.argwhere(self.tTAC[1] != 0.0).T[0]
        t_thresh = pet_grph.get_index_from_threshold(times_in_minutes=self.pTAC[0][non_zero_indices],
                                                     t_thresh_in_minutes=self.t_thresh_in_mins)

        x = pet_grph.cumulative_trapezoidal_integral(xdata=self.pTAC[0], ydata=self.pTAC[1])
        y = pet_grph.cumulative_trapezoidal_integral(xdata=self.tTAC[0], ydata=self.tTAC[1])

        x = x[non_zero_indices] / self.tTAC[1][non_zero_indices]
        y = y[non_zero_indices] / self.tTAC[1][non_zero_indices]

        self.x = x[:]
        self.y = y[:]
        self.non_zero_idx = non_zero_indices[:]
        self.t_thresh_idx = t_thresh
        return None

    def generate_label_from_fit_params(self) -> str:
        r"""
        Creates a label string from the fit parameters for graphical presentation.

        This method retrieves slope, intercept, and R-squared values from the instance's fit parameters, and then
        formats these values into a string that is LaTeX compatible for later rendering inside a plot's label.
        For example:

        .. math::

            V_\mathrm{T}&=0.1\\
            b&=0.2\\
            R^{2}&=0.95

        Returns:
            str: The created label. Each parameter is formatted as a separate line.
        """
        slope = self.fit_params['slope']
        intercept = self.fit_params['intercept']
        r_sq = self.fit_params['r_squared']

        return f"$V_\mathrm{{T}}=${slope:<5.3f}\n$b=${intercept:<5.3f}\n$R^2=${r_sq:<5.3f}"

    def add_figure_axes_labels_and_legend(self):
        r"""
        Adds labels and a legend to the axes of the figure.

        This method sets the `x_label` and `y_label` for all axes on the figure. It also adds a legend to the figure,
        which is anchored to the upper left corner. Lastly, we also give a title to the figure: Logan Plots.

        The labels are set to:

        .. math::

            y&= \frac{\int_{0}^{t}R(s)\mathrm{d}s}{R(t)}\\
            x&= \frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{R(t)},

        where :math:`C_\mathrm{P}` is the input function and :math:`R(t)` is PET activity in the particular region of
        interest.

        See Also:
            * :meth:`calculate_valid_indicies_and_x_and_y` for the calculation implementation.

        """
        x_label = r"$\frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{R(t)}$"
        y_label = r"$\frac{\int_{0}^{t}R(s)\mathrm{d}s}{R(t)}$"
        for ax in self.ax_list:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        self.fig.legend(*self.ax_list[0].get_legend_handles_labels(), bbox_to_anchor=(1.0, 0.8), loc='upper left',
                        title='Logan Analysis')
        self.fig.suptitle("Logan Plots")


class AltLoganPlot(GraphicalAnalysisPlot):
    r"""
    This class handles generation of Alternative Logan plots (or "new plots") for PET analysis.

    The :class:`AltLoganPlot` class is designed to process PET data and display it in the form of Alt-Logan plots. The
    class handles data processing including computation of valid indices and calculation of plot-specific parameters,
    setting of plot labels and legends, as well as the actual plotting of data.

    The processing steps involve computations such as handling of non-zero indices in the plasma time-activity curve
    (pTAC), calculation of :math:`x` and :math:`y` coordinates for the Alt-Logan plot, and generation of plot labeling
    information.

    The class also provides a capability to add labels, legends and title to the figure to present the data in a
    meaningful way.

    Note:
        The class inherits from the :class:`GraphicalAnalysisPlot` abstract base class.

    Example:
        In the proceeding examples, ``pTAC`` represent the plasma TAC (or input TAC) and ``tTAC`` represents the tissue TAC.

        For the quickest way to generate the Alt-Logan Plot, we just instantiate the class and run the
        :meth:`generate_figure` method. This will generate 2 plots in a row. The first one having a linear-linear scale,
        and the second one having a log-log scale.

        .. code-block:: python

            from petpal.graphical_plots import AltLoganPlot
            alt_logan_plot = AltLoganPlot(tTAC=tTAC, pTAC=pTAC, t_thresh_in_mins=45.0)
            alt_logan_plot.generate_figure()
            plt.show() # Or use plt.savefig() to save the figure.

        If the default styling needs to be changed, we can pass keyword arguments to each of the plotting methods:

        .. code-block:: python

            from petpal.graphical_plots import AltLoganPlot
            alt_logan_plot = AltLoganPlot(tTAC=tTAC, pTAC=pTAC, t_thresh_in_mins=45.0)
            alt_logan_plot.generate_figure(line_kwargs=dict(lw=2, alpha=0.95, color='red', label=patlak_plot.generate_label_from_fit_params()),
                                        shading_kwargs=dict(color='palegreen', alpha=0.2),
                                        data_kwargs=dict(alpha=0.85, color='k', marker='.'))

    See Also:
            * :class:`PatlakPlot`
            * :class:`LoganPlot`

    """

    def calculate_valid_indicies_and_x_and_y(self) -> None:
        r"""Calculates the valid indices along with :math:`x` and :math:`y` for Alt-Logan plot analysis.

        This method performs the computation for the non-zero indices in the provided region time-activity curve (tTAC).
        It further calculates the values of :math:`x` and :math:`y` used in Logan analysis based on these non-zero
        indices. This is done to avoid singularities caused by zero denominators. The Alt-Logan :math:`x` and :math:`y`
        values are:

        .. math::

            y&= \frac{\int_{0}^{t}R(s)\mathrm{d}s}{C_\mathrm{P}(t)}\\
            x&= \frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{C_\mathrm{P}(t)},

        where :math:`C_\mathrm{P}` is the input function and :math:`R(t)` is PET activity in the particular region of
        interest. Compared to :class:`LoganPlot`, the denominator is the input function.

        The method updates the instance variables ``x``, ``y``, ``non_zero_idx``, and ``t_thresh_idx``.

        Returns:
            None

        """
        non_zero_indices = np.argwhere(self.pTAC[1] != 0.0).T[0]
        t_thresh = pet_grph.get_index_from_threshold(times_in_minutes=self.pTAC[0][non_zero_indices],
                                                     t_thresh_in_minutes=self.t_thresh_in_mins)

        x = pet_grph.cumulative_trapezoidal_integral(xdata=self.pTAC[0], ydata=self.pTAC[1])
        y = pet_grph.cumulative_trapezoidal_integral(xdata=self.tTAC[0], ydata=self.tTAC[1])

        x = x[non_zero_indices] / self.pTAC[1][non_zero_indices]
        y = y[non_zero_indices] / self.pTAC[1][non_zero_indices]

        self.x = x[:]
        self.y = y[:]
        self.non_zero_idx = non_zero_indices[:]
        self.t_thresh_idx = t_thresh
        return None

    def generate_label_from_fit_params(self) -> str:
        r"""
        Creates a label string from the fit parameters for graphical presentation.

        This method retrieves slope, intercept, and R-squared values from the instance's fit parameters, and then
        formats these values into a string that is LaTeX compatible for later rendering inside a plot's label.
        For example:

        .. math::

            V_\mathrm{T}&=0.1\\
            b&=0.2\\
            R^{2}&=0.95

        Returns:
            str: The created label. Each parameter is formatted as a separate line.
        """
        slope = self.fit_params['slope']
        intercept = self.fit_params['intercept']
        r_sq = self.fit_params['r_squared']

        return f"$m=${slope:<5.3f}\n$b=${intercept:<5.3f}\n$R^2=${r_sq:<5.3f}"

    def add_figure_axes_labels_and_legend(self):
        r"""
        Adds labels and a legend to the axes of the figure.

        This method sets the `x_label` and `y_label` for all axes on the figure. It also adds a legend to the figure,
        which is anchored to the upper left corner. Lastly, we also give a title to the figure: Alt-Logan Plots.

        The labels are set to:

        .. math::

            y&= \frac{\int_{0}^{t}R(s)\mathrm{d}s}{C_\mathrm{P}(t)}\\
            x&= \frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{C_\mathrm{P}},

        where :math:`C_\mathrm{P}` is the input function and :math:`R(t)` is PET activity in the particular region of
        interest.

        See Also:
            * :meth:`calculate_valid_indicies_and_x_and_y` for the calculation implementation.

        """
        x_label = r"$\frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{C_\mathrm{P}(t)}$"
        y_label = r"$\frac{\int_{0}^{t}R(s)\mathrm{d}s}{C_\mathrm{P}(t)}$"
        for ax in self.ax_list:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        self.fig.legend(*self.ax_list[0].get_legend_handles_labels(), bbox_to_anchor=(1.0, 0.8), loc='upper left',
                        title='Alt-Logan Analysis')
        self.fig.suptitle("Alt-Logan Plots")


class Plot:
    """
    A class used to generate and save graphical analysis plots of PET Time-Activity Curves (TACs).

    This class provides a simple and convenient interface to generate plots from TACs for the 'patlak', 'logan', or
    'alt-logan' methods and save them in the specified output directory. The plots are saved in both PNG and PDF formats.

    It is initialized with paths to the input TACs, the method name, the output directory, and an optional filename
    prefix. During initialization, it checks the validity of the file paths and directory, loads the TACs, and selects
    the appropriate figure class based on the given method. Later on, the `save_figure` method can be called to generate
    and save the plots.

    Attributes:
        input_tac_path (str): Path to the input TAC file.
        roi_tac_path (str): Path to the Region of Interest (ROI) TAC file.
        output_directory (str): Path to the directory where the plot output will be saved.
        method_name (str): Name of the method for generating the plot ('patlak', 'logan', 'alt-logan').
        thresh_in_mins (float): A threshold in minutes below which data points will be discarded from the plot.
        output_filename_prefix (str): An optional prefix for the output filenames.

    Example:
        An example of how to use this class to save a Patlak graphical analysis plot:


        .. code-block:: python

            import petpal.graphical_plots as pet_plt
            # ptac_path points to a plasma TAC (or input TAC)
            # ttac_path points to a ROI TAC.
            grph_plot = pet_plt.Plot(input_tac_path=ptac_path,
                                     roi_tac_path=ttac_path,
                                     threshold_in_mins=30.0,
                                     method_name='patlak',
                                     output_directory='./',
                                     output_filename_prefix="plot")
            grph_plot.save_figure()

    """

    def __init__(self,
                 input_tac_path: str,
                 roi_tac_path: str,
                 threshold_in_mins: float,
                 method_name: str,
                 output_directory: str,
                 output_filename_prefix: str = "") -> None:
        """
        Initializes the :class:`Plot` object given the paths and the method.

        During the initialization, it validates the given file paths and directory,  and selects the appropriate figure
        class based on the provided method name.

        Args:
            input_tac_path (str): Path to the input Time Activity Curve (TAC) file.
            roi_tac_path (str): Path to the Region of Interest (ROI) TAC file.
            threshold_in_mins (float): Threshold in minutes below which data points will be discarded.
            method_name (str): Name of the method to be used ('patlak', 'logan', or 'alt-logan').
            output_directory (str): Path to the directory where output files will be stored.
            output_filename_prefix (str, optional): Prefix to be appended to the output files. Defaults to an empty string.

        Raises:
            ValueError: If any of the file paths or the directory path does not exist, or if the method name is invalid.

        Note:
            This method does not return a value. It's an initializer of this class.

        Calls:
            * :meth:`_validate_filepath`
            * :meth:`_validate_directory`
            * :func:`safe_load_tac`
            * :meth:`_select_fig_class_based_on_method`

        """
        self.input_tac_path = os.path.abspath(input_tac_path)
        self.roi_tac_path = os.path.abspath(roi_tac_path)
        self.output_directory = os.path.abspath(output_directory)
        self._validate_filepath(self.input_tac_path)
        self._validate_filepath(self.roi_tac_path)
        self._validate_directory(self.output_directory)
        self.method_name = method_name
        self.thresh_in_mins = threshold_in_mins
        self.output_filename_prefix = output_filename_prefix
        self.fig_cls = self._select_fig_class_based_on_method(method_name=self.method_name)

    def save_figure(self):
        """
        Creates and saves the figure in both PNG and PDF formats.

        We first safely load the TACS. Then, this method generates the figure using the method selected during the
        initialization of the class. The figure is then saved inside the specified  output directory with a filename
        based on parameters provided at initialization.

        The saved files are in both PNG (with a resolution of 150 dpi) and PDF formats (with the PDF file being saved
        with transparency).

        Note:
            This method does not return any value. It is responsible for saving the generated figure to the specified
            output directory in PNG and PDF formats.

        Raises:
            Exception: An error occurred while loading the TACs from the input or ROI files.

        """
        p_tac = safe_load_tac(self.input_tac_path)
        t_tac = safe_load_tac(self.roi_tac_path)

        filename = f"{self.output_filename_prefix}_analysis-{self.method_name}"
        out_path = os.path.join(self.output_directory, filename)

        with plt.rc_context(rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.usetex': True}):
            fig_cls = self.fig_cls(pTAC=p_tac, tTAC=t_tac, t_thresh_in_mins=self.thresh_in_mins)
            fig_cls.generate_figure()
            plt.savefig(f"{out_path}.png", bbox_inches='tight', dpi=150)
            plt.savefig(f"{out_path}.pdf", bbox_inches='tight', transparent=True)
            plt.close()

    @staticmethod
    def _validate_filepath(filename: str) -> None:
        """
        Checks if the provided filename is a valid file.

        This is a private static method and should be called internally by the class while processing files.

        If the file is not valid, it raises a ValueError.

        Args:
            filename (str): The path to the file.

        Raises:
            ValueError: if the file at provided path does not exist.

        No return value.

        """
        if not os.path.isfile(filename):
            raise ValueError(f"Invalid file path: {filename}")

    @staticmethod
    def _validate_directory(directory: str) -> None:
        """
        Validates if the provided directory path is an existing directory.

        This is a private static method and should be called internally by the class while processing directories.

        If the directory does not exist, it raises a ValueError.

        Args:
            directory (str): Path to the directory.

        Raises:
            ValueError: if the provided directory path does not exist.

        No return value.

        """
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}.")

    @staticmethod
    def _select_fig_class_based_on_method(method_name: str) -> Union[
        Type[PatlakPlot], Type[LoganPlot], Type[AltLoganPlot]]:
        """
        Selects and returns the appropriate class based on the provided method name.

        This private static method returns the corresponding class object (`PatlakPlot`, `LoganPlot`, or `AltLoganPlot`)
        according to the provided method name ('patlak', 'logan', or 'alt-logan').

        Args:
            method_name (str): Name of the method.

        Returns:
            :class:`PatlakPlot` | :class:`LoganPlot` | :class:`AltLoganPlot`: Corresponding class.

        Raises:
            ValueError: The method name is invalid.

        See Also:
            * :class:`PatlakPlot`
            * :class:`LoganPlot`
            * :class:`AltLoganPlot`

        """
        if method_name == "patlak":
            return PatlakPlot
        elif method_name == "logan":
            return LoganPlot
        elif method_name == "alt-logan":
            return AltLoganPlot
        else:
            raise ValueError("Invalid method name. Please choose either 'patlak', 'logan', or 'alt-logan'.")
