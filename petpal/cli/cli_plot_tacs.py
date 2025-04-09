"""
CLI module to plot TACs from a TACs folder created by petpal function write-tacs.
"""
import argparse
from ..visualizations.tac_plots import TacFigure, RegionalTacFigure
from ..utils.time_activity_curve import TimeActivityCurve


_PLOT_EXAMPLES_ = r"""
Examples:
  - Plot a single TAC:
    petpal-plot-tacs --tac-files my_tac.tsv --out-fig-path tac.png
  - Plot two or more TACs:
    petpal-plot-tacs --tac-files my_tac_1.tsv my_tac_2.tsv --out-fig-path tac.png
  - Plot all the TACs in a directory:
    petpal-plot-tacs --tac-dir sub-001/tacs/ --out-fig-path tac.png
  - Plot specific regional TACs in a directory based on region names:
    petpal-plot-tacs --tac-files my_tac.tsv --regions RightPutamen LeftPutamen --out-fig-path tac.png
  - Set x-axis and y-axis units:
    petpal-plot-tacs --tac-files my_tac.tsv --yaxis-units cps --xaxis-units hours --out-fig-path tac.png
  - Plot the linear-linear plot only:
    petpal-plot-tacs --tac-files my_tac.tsv --plot-type linear --out-fig-path tac.png
  - Set the figure title to the participant name:
    petpal-plot-tacs --tac-files my_tac.tsv --participant sub-001 --out-fig-path tac.png
"""


def main():
    """
    CLI for tac plotting
    """

    parser = argparse.ArgumentParser(prog='petpal-plot-tacs',
                                     description='Command line interface for plotting TACs.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=_PLOT_EXAMPLES_)
    parser.add_argument('--tac-files',
                        required=False,
                        nargs='+',
                        help='Path to or more individual .tsv TAC files, separate paths with '
                             'spaces. At least one of: --tac-files, --tac-dir, are required.')
    parser.add_argument('--tac-dir',
                        required=False,
                        help='Path to a directory containing .tsv TAC files generated with PETPAL.'
                             ' At least one of: --tac-files, --tac-dir, are required.')
    parser.add_argument('--out-fig-path',
                        required=True,
                        help='Path to the file where the figure is saved.')
    parser.add_argument('--participant',
                        required=False,
                        help='Name of the participant the TAC or TACs belong to. Assigned to '
                             'figure title.')
    parser.add_argument('--regions',
                        required=False,
                        nargs='+',
                        help='If --tac-files is set, list the regions to be plotted. Separate '
                             'region names with spaces. Expecting TAC file names to follow the '
                             'convention: *seg-SegmentName* where SegmentName does not contain '
                             'special characters, especially - and _ which will conflict with '
                             'PETPAL code.')
    parser.add_argument('--plot-type',
                        required=False,
                        default='both',
                        choices=['linear','log','both'],
                        help='Set whether to plot the TACs as linear-linear, log-linear, or both.')
    parser.add_argument('--yaxis-units',
                        required=False,
                        default='Bq/mL',
                        choices=['Bq/mL','kBq/mL','cps'],
                        help='Set activity concentration unit label for the y-axis. Does not scale'
                             ' units, this only assigns the axis label name.')
    parser.add_argument('--xaxis-units',
                        required=False,
                        default='minutes',
                        choices=['minutes','seconds','hours'],
                        help='Set time units for the x-axis. Does not scale units, this only '
                             'assigns the axis label name.')
    parser.add_argument('--plot-style',
                        required=False,
                        default='lines',
                        choices=['lines','markers','both'],
                        help='Set style of TAC plots.')

    args = parser.parse_args()

    if args.tac_dir is None and args.tac_files is None:
        raise SystemExit('Both --tac-files and --tac-dir unset. Exiting.')

    if args.tac_dir is None:
        fig = TacFigure(plot_type=args.plot_type,
                        xlabel=fr'$t$ [{args.xaxis_units}]',
                        ylabel=fr'TAC [$\mathrm{{{args.yaxis_units}}}$]')
    else:
        fig = RegionalTacFigure(tacs_dir=args.tac_dir,
                                plot_type=args.plot_type,
                                xlabel=fr'$t$ [{args.xaxis_units}]',
                                ylabel=fr'TAC [$\mathrm{{{args.yaxis_units}}}$]')

    plot_style_opts = {'lines': '-', 'markers': 'o', 'both': '-o'}
    plot_style = plot_style_opts[args.plot_style]

    if args.tac_files is not None:
        for tac_file in args.tac_files:
            tac = TimeActivityCurve.from_tsv(filename=tac_file)
            fig.add_errorbar(*tac.tac_werr, fmt=plot_style)

    if args.tac_dir is not None:
        if args.regions is None:
            fig.plot_all_regional_tacs(fmt=plot_style)
        else:
            fig.plot_tacs_in_regions_list(regions=args.regions, fmt=plot_style)

    if args.participant is not None:
        fig.fig.suptitle(t=args.participant)

    fig.write_fig(out_fig_path=args.out_fig_path)


if __name__ == "__main__":
    main()
