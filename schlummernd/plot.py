import warnings

import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import binned_statistic_2d

__all__ = ["colored_corner"]


def colored_corner(
    X,
    scatter=True,
    color_by=None,
    labels=None,
    bins=64,
    statistic="mean",
    axes=None,
    add_colorbar=False,
    **style
):
    """
    TODO: this could become a PR to corner.py

    Parameters
    ----------
    X : array-like
        The data to plot. Should have shape ``(N, K)`` where ``N`` are the number of
        data points and ``K`` are the number of "features" (i.e. the number of elements
        on the diagonal of the corner plot).
    scatter : bool (optional)
        Controls whether to make a scatter plot or a set of images (binned statistics).
    color_by : array-like (optional)
        An array used to color the datapoints.
    labels : iterable (optional)
        A list of labels to add to the axes, one per "feature."
    bins : int, iterable (optional)
        If ``scatter=False``, the number of bins to use, or the bin edges for each
        feature.
    statistic : str, callable (optional)
        If ``scatter=False``, this is passed to ``scipy.stats.binned_statistic_2d`` to
        compute the colored 2D images based on the input data.
    axes : `matplotlib.Axes` (optional)
    add_colorbar : bool (optional)
        Controls whether to add a colorbar to the figure.
    **style
        Additional keyword arguments are passed to the plotting function; either
        ``plot()``, ``scatter()``, or ``pcolormesh()``.

    """
    X = np.asanyarray(X)

    if X.shape[1] > X.shape[0]:
        raise ValueError(
            "Number of data points is less than the number of features in your input "
            "data. Are you sure you don't need to transpose?"
        )

    if scatter:
        if color_by is None:
            plotfunc = "plot"
            style.setdefault("marker", "o")
            style.setdefault("mew", style.pop("markeredgewidth", 0))
            style.setdefault("ls", style.pop("linestyle", "none"))
            style.setdefault("ms", style.pop("markersize", 2.0))
        else:
            plotfunc = "scatter"
            style.setdefault("marker", "o")
            style.setdefault("lw", style.pop("linewidth", 0))
            style.setdefault("s", 5)
            style.setdefault("c", color_by)
    else:
        if color_by is None and statistic != 'count':
            raise ValueError(
                "If you would like to make images based on the input data, you must "
                "pass in an array for `color_by`."
            )

        try:
            bins = int(bins)
            # TODO: magic numbers 5 and 95
            bins = [
                np.linspace(*np.nanpercentile(X[:, k], [5, 95]), bins)
                for k in range(X.shape[1])
            ]
        except (ValueError, TypeError):
            bins = list(bins)

        if len(bins) != X.shape[1]:
            raise ValueError(
                "If specifying the bins explicitly, you must pass in an array of bin "
                "edges for each feature in the input data."
            )

        if color_by is not None:
            # TODO: magic numbers 5, 95
            style.setdefault('vmin', np.nanpercentile(color_by, 5))
            style.setdefault('vmax', np.nanpercentile(color_by, 95))

    nside = X.shape[1] - 1

    # Some magic numbers for pretty axis layout.
    # Stolen from corner.py!
    K = X.shape[1]
    factor = 2.0  # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    if axes is None:
        if add_colorbar:
            figsize = (dim + 1, dim)
        else:
            figsize = (dim, dim)

        fig, axes = plt.subplots(
            nside,
            nside,
            figsize=figsize,
            sharex="col",
            sharey="row",
            constrained_layout=True,
        )
    else:
        fig = axes.flat[0].figure

    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])

    cs = None
    for i in range(nside):
        for j in range(nside):
            ax = axes[i, j]
            if i < j:
                ax.set_visible(False)
            else:
                if scatter:
                    cs = getattr(ax, plotfunc)(X[:, j], X[:, i + 1], **style)
                else:
                    stat = binned_statistic_2d(
                        X[:, j],
                        X[:, i + 1],
                        color_by,
                        statistic=statistic,
                        bins=(bins[j], bins[i])
                    )
                    cs = ax.pcolormesh(
                        stat.x_edge, stat.y_edge, stat.statistic.T, **style
                    )

    if labels is not None:
        for i in range(nside):
            axes[i, 0].set_ylabel(labels[i + 1])

        for j in range(nside):
            axes[-1, j].set_xlabel(labels[j])

    return_stuff = [fig, axes]

    if add_colorbar and color_by is not None and cs is not None:
        cb = fig.colorbar(cs, ax=axes, aspect=30)
        return_stuff.append(cb)

    return return_stuff


# def plot_hr_cmd(
#     parent_stars, stars, idx0, other_idx, teff_logg_bins=None, cmd_bins=None
# ):

#     style_main = dict(
#         ls="none",
#         marker="o",
#         mew=0.6,
#         ms=6.0,
#         color="tab:blue",
#         zorder=100,
#         mec="gold",
#     )
#     style_neighbors = dict(
#         ls="none",
#         marker="o",
#         mew=0,
#         ms=2.0,
#         alpha=0.75,
#         color="tab:orange",
#         zorder=10,
#     )

#     if teff_logg_bins is None:
#         teff_logg_bins = (
#             np.linspace(3000, 9000, 128),
#             np.linspace(-0.5, 5.75, 128),
#         )

#     if cmd_bins is None:
#         cmd_bins = (np.linspace(-0.5, 2, 128), np.linspace(-6, 10, 128))

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     ax = axes[0]
#     ax.hist2d(
#         parent_stars["TEFF"],
#         parent_stars["LOGG"],
#         bins=teff_logg_bins,
#         norm=mpl.colors.LogNorm(),
#         cmap="Greys",
#     )

#     ax.plot(stars["TEFF"][idx0], stars["LOGG"][idx0], **style_main)

#     ax.plot(stars["TEFF"][other_idx], stars["LOGG"][other_idx], **style_neighbors)

#     ax.set_xlim(teff_logg_bins[0].max(), teff_logg_bins[0].min())
#     ax.set_ylim(teff_logg_bins[1].max(), teff_logg_bins[1].min())

#     ax.set_xlabel(r"$T_{\rm eff}$")
#     ax.set_ylabel(r"$\log g$")

#     # ---

#     ax = axes[1]

#     color = ("J", "K")
#     mag = "H"

#     (dist_mask, ) = np.where(
#         (parent_stars["GAIAEDR3_PARALLAX"] /
#          parent_stars["GAIAEDR3_PARALLAX_ERROR"]) > 5
#     )
#     plx = parent_stars["GAIAEDR3_PARALLAX"][dist_mask] * u.mas
#     distmod = coord.Distance(parallax=plx).distmod.value
#     ax.hist2d(
#         (parent_stars[color[0]] - parent_stars[color[1]])[dist_mask],
#         parent_stars[mag][dist_mask] - distmod,
#         bins=cmd_bins,
#         norm=mpl.colors.LogNorm(),
#         cmap="Greys",
#     )

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         distmod = coord.Distance(
#             parallax=stars["GAIAEDR3_PARALLAX"] * u.mas, allow_negative=True
#         ).distmod.value
#     ax.plot((stars[color[0]] - stars[color[1]])[idx0], (stars[mag] - distmod)[idx0],
#             **style_main)

#     ax.plot((stars[color[0]] - stars[color[1]])[other_idx],
#             (stars[mag] - distmod)[other_idx], **style_neighbors)

#     ax.set_xlim(cmd_bins[0].min(), cmd_bins[0].max())
#     ax.set_ylim(cmd_bins[1].max(), cmd_bins[1].min())

#     ax.set_xlabel("$J - K$")
#     ax.set_ylabel("$M_H$")

#     return fig
