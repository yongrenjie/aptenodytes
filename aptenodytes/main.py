"""
main.py
-------

All the functionality is in this one file. They are for personal use, they are
largely undocumented! Use at your own risk!
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Any, Union, Generator

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import penguins as pg
from penguins import dataset as ds  # for type annotations


# These are pure convenience routines for my personal use.
# Default save location for plots
dsl = Path("/Users/yongrenjie/Desktop/a_plot.png")
# Path to NMR spectra. The $nmrd environment variable should resolve to
# .../dphil/expn/nmr. On my Mac this is set to my SSD.
def __getenv(key):
    if os.getenv(key) is not None:
        x = Path(os.getenv(key))
        if x.exists():
            return x
    raise FileNotFoundError("$nmrd does not point to a valid location.")


def nmrd():
    return __getenv("nmrd")


# -- Seaborn plotting functions for SNR comparisons

def hsqc_stripplot(molecule: Any,
                   datasets: Union[ds.Dataset2D, Sequence[ds.Dataset2D]],
                   ref_dataset: ds.Dataset2D,
                   expt_labels: Union[str, Sequence[str]],
                   xlabel: str = "Experiment",
                   ylabel: str = "Intensity",
                   title: str = "",
                   edited: bool = False,
                   show_averages: bool = True,
                   ncol: int = 3,
                   loc: str = "upper center",
                   ax: Optional[Any] = None,
                   **kwargs: Any,
                   ) -> Tuple[Any, Any]:
    """
    Plot HSQC strip plots (i.e. plot relative intensities, split by
    multiplicity).

    Parameters
    ----------
    molecule : pg.private.Andrographolide or pg.private.Zolmitriptan
        The class from which the hsqc attribute will be taken from
    datasets : pg.Dataset2D or sequence of pg.Dataset2D
        Dataset(s) to analyse intensities of
    ref_dataset : pg.Dataset2D
        Reference dataset
    expt_labels : str or sequence of strings
        Labels for the analysed datasets
    xlabel : str, optional
        Axes x-label, defaults to "Experiment"
    ylabel : str, optional
        Axes y-label, defaults to "Intensity"
    title : str, optional
        Axes title, defaults to empty string
    edited : bool, default False
        Whether editing is enabled or not.
    show_averages : bool, default True
        Whether to indicate averages in each category using sns.pointplot.
    ncol : int, optional
        Passed to ax.legend(). Defaults to 4.
    loc : str, optional
        Passed to ax.legend(). Defaults to "upper center".
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on. If not provided, uses plt.gca().
    kwargs : dict, optional
        Keywords passed on to sns.stripplot().

    Returns
    -------
    (fig, ax).
    """
    # Stick dataset/label into a list if needed
    if isinstance(datasets, ds.Dataset2D):
        datasets = [datasets]
    if isinstance(expt_labels, str):
        expt_labels = [expt_labels]
    # Calculate dataframes of relative intensities.
    rel_ints_dfs = [molecule.hsqc.rel_ints_df(dataset=ds,
                                              ref_dataset=ref_dataset,
                                              label=label,
                                              edited=edited)
                    for (ds, label) in zip(datasets, expt_labels)]
    all_dfs = pd.concat(rel_ints_dfs)

    # Calculate the average integrals by multiplicity
    avgd_ints = pd.concat((df.groupby("mult").mean() for df in rel_ints_dfs),
                          axis=1)
    avgd_ints.drop(columns=["f1", "f2"], inplace=True)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    stripplot_alpha = 0.3 if show_averages else 0.8
    sns.stripplot(x="expt", y="int", hue="mult",
                  zorder=0, alpha=stripplot_alpha,
                  dodge=True, data=all_dfs, ax=ax, **kwargs)
    if show_averages:
        sns.pointplot(x="expt", y="int", hue="mult", zorder=1,
                      dodge=0.5, data=all_dfs, ax=ax, join=False,
                      markers='_', palette="dark", ci=None, scale=1.25)
    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    handles, _ = ax.get_legend_handles_labels()
    l = ax.legend(ncol=ncol, loc=loc,
                  markerscale=0.4,
                  handles=handles[0:3],
                  labels=["CH", r"CH$_2$", r"CH$_3$"])
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")
    # Set y-limits. We need to expand it by ~20% to make space for the legend,
    # as well as the averaged values.
    EXPANSION_FACTOR = 1.2
    ymin, ymax = ax.get_ylim()
    ymean = (ymin + ymax)/2
    ylength = (ymax - ymin)/2
    new_ymin = ymean - (EXPANSION_FACTOR * ylength)
    new_ymax = ymean + (EXPANSION_FACTOR * ylength)
    ax.set_ylim((new_ymin, new_ymax))
    # add the text
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), color) in enumerate(zip(expt_avgs.items(),
                                                  sns.color_palette("deep"))):
            ax.text(x=x-0.25+i*0.25, y=0.02, s=f"({avg:.2f})",
                    color=color, horizontalalignment="center",
                    transform=ax.get_xaxis_transform())
    pg.style_axes(ax, "plot")
    return plt.gcf(), ax


def cosy_stripplot(molecule: Any,
                   datasets: Union[ds.Dataset2D, Sequence[ds.Dataset2D]],
                   ref_dataset: ds.Dataset2D,
                   expt_labels: Union[str, Sequence[str]],
                   xlabel: str = "Experiment",
                   ylabel: str = "Intensity",
                   title: str = "",
                   ncol: int = 2,
                   separate_type: bool = True,
                   loc: str = "upper center",
                   ax: Optional[Any] = None,
                   **kwargs: Any,
                   ) -> Tuple[Any, Any]:
    """
    Plot COSY strip plots (i.e. plot relative intensities, split by peak type).

    Parameters
    ----------
    molecule : pg.private.Andrographolide or pg.private.Zolmitriptan
        The class from which the cosy attribute will be taken from
    datasets : pg.Dataset2D or sequence of pg.Dataset2D
        Dataset(s) to analyse intensities of
    ref_dataset : pg.Dataset2D
        Reference dataset
    expt_labels : str or sequence of strings
        Labels for the analysed datasets
    xlabel : str, optional
        Axes x-label, defaults to "Experiment"
    ylabel : str, optional
        Axes y-label, defaults to "Intensity"
    title : str, optional
        Axes title, defaults to empty string
    ncol : int, optional
        Passed to ax.legend(). Defaults to 4.
    loc : str, optional
        Passed to ax.legend(). Defaults to "upper center".
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on. If not provided, uses plt.gca().
    kwargs : dict, optional
        Keywords passed on to sns.stripplot().

    Returns
    -------
    (fig, ax).
    """
    # Stick dataset/label into a list if needed
    if isinstance(datasets, ds.Dataset2D):
        datasets = [datasets]
    if isinstance(expt_labels, str):
        expt_labels = [expt_labels]
    # Calculate dataframes of relative intensities.
    rel_ints_dfs = [molecule.cosy.rel_ints_df(dataset=ds,
                                              ref_dataset=ref_dataset,
                                              label=label)
                    for (ds, label) in zip(datasets, expt_labels)]
    if not separate_type:
        rel_ints_dfs = [rel_int_df.assign(type="cosy")
                        for rel_int_df in rel_ints_dfs]
    all_dfs = pd.concat(rel_ints_dfs)

    # Calculate the average integrals by type
    avgd_ints = pd.concat((df.groupby("type").mean() for df in rel_ints_dfs),
                          axis=1)
    avgd_ints.drop(columns=["f1", "f2"], inplace=True)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    sns.stripplot(x="expt", y="int", hue="type",
                  dodge=True, data=all_dfs, ax=ax,
                  palette=sns.color_palette("deep")[3:], **kwargs)
    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if separate_type:
        ax.legend(ncol=ncol, loc=loc,
                  labels=["diagonal", "cross"]).set(title=None)
    else:
        ax.legend().set_visible(False)
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")
    # Set y-limits. We need to expand it by ~20% to make space for the legend,
    # as well as the averaged values.
    EXPANSION_FACTOR = 1.2
    ymin, ymax = ax.get_ylim()
    ymean = (ymin + ymax)/2
    ylength = (ymax - ymin)/2
    new_ymin = ymean - (EXPANSION_FACTOR * ylength)
    new_ymax = ymean + (EXPANSION_FACTOR * ylength)
    ax.set_ylim((new_ymin, new_ymax))
    # add the text
    offset = -0.2 if separate_type else 0
    dx = 0.4 if separate_type else 1
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), color) in enumerate(zip(
                expt_avgs.items(), sns.color_palette("deep")[3:])):
            ax.text(x=x-offset+i*dx, y=0.02, s=f"({avg:.2f})",
                    color=color, horizontalalignment="center",
                    transform=ax.get_xaxis_transform())
    pg.style_axes(ax, "plot")
    return plt.gcf(), ax


def hsqc_cosy_stripplot(molecule: Any,
                        datasets: Sequence[ds.Dataset2D],
                        ref_datasets: Sequence[ds.Dataset2D],
                        xlabel: str = "Experiment",
                        ylabel: str = "Intensity",
                        title: str = "",
                        edited: bool = False,
                        show_averages: bool = True,
                        separate_mult: bool = True,
                        ncol: int = 4,
                        loc: str = "upper center",
                        ax: Optional[Any] = None,
                        font_kwargs: Optional[dict] = None,
                        **kwargs: Any,
                        ) -> Tuple[Any, Any]:
    """
    Plot HSQC and COSY relative intensities on the same Axes. HSQC peaks are
    split by multiplicity, COSY peaks are not split.

    Parameters
    ----------
    molecule : pg.private.Andrographolide or pg.private.Zolmitriptan
        The class from which the hsqc and cosy attributes will be taken from
    datasets : (pg.Dataset2D, pg.Dataset2D)
        HSQC and COSY dataset(s) to analyse intensities of
    ref_datasets : (pg.Dataset2D, pg.Dataset2D)
        Reference HSQC and COSY datasets
    xlabel : str, optional
        Axes x-label, defaults to "Experiment"
    ylabel : str, optional
        Axes y-label, defaults to "Intensity"
    title : str, optional
        Axes title, defaults to empty string
    edited : bool, default False
        Whether editing in the HSQC is enabled or not.
    show_averages : bool, default True
        Whether to indicate averages in each category using sns.pointplot.
    ncol : int, optional
        Passed to ax.legend(). Defaults to 4.
    loc : str, optional
        Passed to ax.legend(). Defaults to "upper center".
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on. If not provided, uses plt.gca().
    kwargs : dict, optional
        Keywords passed on to sns.stripplot().

    Returns
    -------
    (fig, ax).
    """
    # Set up default font_kwargs if not provided.
    font_kwargs = font_kwargs or {}
    # Calculate dataframes of relative intensities.
    hsqc_rel_ints_df = molecule.hsqc.rel_ints_df(dataset=datasets[0],
                                                 ref_dataset=ref_datasets[0],
                                                 edited=edited)
    # Rename mult -> type to match COSY
    hsqc_rel_ints_df = hsqc_rel_ints_df.rename(columns={"mult": "type"})
    # Remove multiplicity information if separation is not desired
    if not separate_mult:
        hsqc_rel_ints_df = hsqc_rel_ints_df.assign(type="hsqc")
    cosy_rel_ints_df = molecule.cosy.rel_ints_df(dataset=datasets[1],
                                                 ref_dataset=ref_datasets[1])
    cosy_rel_ints_df = cosy_rel_ints_df.assign(type="cosy")
    rel_ints_df = pd.concat((hsqc_rel_ints_df, cosy_rel_ints_df))

    # Calculate the average integrals by multiplicity
    avgd_ints = rel_ints_df.groupby("type").mean()
    # Fix the order if we need to (because by default it would be alphabetical)
    if not separate_mult:
        avgd_ints = avgd_ints.reindex(["hsqc", "cosy"])
    avgd_ints.drop(columns=["f1", "f2"], inplace=True)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    stripplot_alpha = 0.3 if show_averages else 0.8
    sns.stripplot(x="expt", y="int", hue="type",
                  zorder=0, alpha=stripplot_alpha,
                  dodge=True, data=rel_ints_df, ax=ax, **kwargs)
    if show_averages:
        dodge = 0.6 if separate_mult else 0.4
        sns.pointplot(x="expt", y="int", hue="type", zorder=1,
                      dodge=dodge, data=rel_ints_df, ax=ax, join=False,
                      markers='_', palette="dark", ci=None, scale=1.25)

    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[])
    # Setting the handles manually ensures that we get stripplot handles
    # rather than the pointplot ones (if present).
    handles, _ = ax.get_legend_handles_labels()
    l = ax.legend(ncol=ncol, loc=loc,
                  markerscale=0.4,
                  handles=handles[0:4],
                  labels=["HSQC CH", r"HSQC CH$_2$", r"HSQC CH$_3$", "COSY"])
    l.set(title=None)
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")
    # Set y-limits. We need to expand it by ~20% to make space for the legend,
    # as well as the averaged values.
    EXPANSION_FACTOR = 1.2
    ymin, ymax = ax.get_ylim()
    ymean = (ymin + ymax)/2
    ylength = (ymax - ymin)/2
    new_ymin = ymean - (EXPANSION_FACTOR * ylength)
    new_ymax = ymean + (EXPANSION_FACTOR * ylength)
    ax.set_ylim((new_ymin, new_ymax))

    # Add the text and averages
    x0 = -0.3 if separate_mult else -0.2
    dx = 0.2 if separate_mult else 0.4
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), deep) in enumerate(zip(expt_avgs.items(),
                                                 sns.color_palette("deep"))):
            ax.text(x=x+x0+i*dx, y=0.02, s=f"({avg:.2f})",
                    color=deep, horizontalalignment="center",
                    transform=ax.get_xaxis_transform(),
                    **font_kwargs)
    pg.style_axes(ax, "plot")
    return plt.gcf(), ax


def hsqcc_stripplot(molecule: Any,
                    datasets: Union[ds.Dataset2D, Sequence[ds.Dataset2D]],
                    ref_dataset: ds.Dataset2D,
                    expt_labels: Union[str, Sequence[str]],
                    xlabel: str = "Experiment",
                    ylabel: str = "Intensity",
                    title: str = "",
                    edited: bool = True,
                    show_averages: bool = True,
                    ncol: int = 2,
                    loc: str = "upper center",
                    ax: Optional[Any] = None,
                    **kwargs: Any,
                    ) -> Tuple[Any, Any]:
    """
    Plot HSQC-COSY strip plots (i.e. plot relative intensities, split by peak
    type).

    Parameters
    ----------
    molecule : pg.private.Andrographolide
        The class from which the hsqc attribute will be taken from
    datasets : pg.Dataset2D or sequence of pg.Dataset2D
        Dataset(s) to analyse intensities of
    ref_dataset : pg.Dataset2D
        Reference dataset
    expt_labels : str or sequence of strings
        Labels for the analysed datasets
    xlabel : str, optional
        Axes x-label, defaults to "Experiment"
    ylabel : str, optional
        Axes y-label, defaults to "Intensity"
    title : str, optional
        Axes title, defaults to empty string
    edited : bool, default False
        Whether editing is enabled or not.
    show_averages : bool, default True
        Whether to indicate averages in each category using sns.pointplot.
    ncol : int, optional
        Passed to ax.legend(). Defaults to 2.
    loc : str, optional
        Passed to ax.legend(). Defaults to "upper center".
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on. If not provided, uses plt.gca().
    kwargs : dict, optional
        Keywords passed on to sns.stripplot().

    Returns
    -------
    (fig, ax).
    """
    # Stick dataset/label into a list if needed
    if isinstance(datasets, ds.Dataset2D):
        datasets = [datasets]
    if isinstance(expt_labels, str):
        expt_labels = [expt_labels]
    # Calculate dataframes of relative intensities.
    rel_ints_dfs = [molecule.hsqc_cosy.rel_ints_df(dataset=ds,
                                                   ref_dataset=ref_dataset,
                                                   label=label,
                                                   edited=edited)
                    for (ds, label) in zip(datasets, expt_labels)]
    all_dfs = pd.concat(rel_ints_dfs)

    # Calculate the average integrals by multiplicity
    avgd_ints = pd.concat((df.groupby("type").mean() for df in rel_ints_dfs),
                          axis=1)
    avgd_ints.drop(columns=["f1", "f2"], inplace=True)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    stripplot_alpha = 0.3 if show_averages else 0.8
    sns.stripplot(x="expt", y="int", hue="type",
                  zorder=0, alpha=stripplot_alpha,
                  dodge=True, data=all_dfs, ax=ax, **kwargs)
    if show_averages:
        sns.pointplot(x="expt", y="int", hue="type", zorder=1,
                      dodge=0.4, data=all_dfs, ax=ax, join=False,
                      markers='_', palette="dark", ci=None, scale=1.25)
    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    handles, _ = ax.get_legend_handles_labels()
    l = ax.legend(ncol=ncol, loc=loc,
                  markerscale=0.4,
                  handles=handles[0:3],
                  labels=["direct", "indirect"])
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")
    # Set y-limits. We need to expand it by ~20% to make space for the legend,
    # as well as the averaged values.
    EXPANSION_FACTOR = 1.2
    ymin, ymax = ax.get_ylim()
    ymean = (ymin + ymax)/2
    ylength = (ymax - ymin)/2
    new_ymin = ymean - (EXPANSION_FACTOR * ylength)
    new_ymax = ymean + (EXPANSION_FACTOR * ylength)
    ax.set_ylim((new_ymin, new_ymax))
    # add the text
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), color) in enumerate(zip(expt_avgs.items(),
                                                  sns.color_palette("deep"))):
            ax.text(x=x-0.2+i*0.4, y=0.02, s=f"({avg:.2f})",
                    color=color, horizontalalignment="center",
                    transform=ax.get_xaxis_transform())
    pg.style_axes(ax, "plot")
    return plt.gcf(), ax


def generic_stripplot(experiment: Any,
                      datasets: Union[ds.Dataset2D, Sequence[ds.Dataset2D]],
                      ref_dataset: ds.Dataset2D,
                      expt_labels: Union[str, Sequence[str]],
                      xlabel: str = "Experiment",
                      ylabel: str = "Intensity",
                      title: str = "",
                      show_averages: bool = True,
                      ncol: int = 2,
                      loc: str = "upper center",
                      ax: Optional[Any] = None,
                      **kwargs: Any,
                      ) -> Tuple[Any, Any]:
    # Stick dataset/label into a list if needed
    if isinstance(datasets, ds.Dataset2D):
        datasets = [datasets]
    if isinstance(expt_labels, str):
        expt_labels = [expt_labels]
    # Calculate dataframes of relative intensities.
    rel_ints_dfs = [experiment.rel_ints_df(dataset=ds,
                                           ref_dataset=ref_dataset,
                                           label=label)
                    for (ds, label) in zip(datasets, expt_labels)]
    all_dfs = pd.concat(rel_ints_dfs)
    # Calculate the average integrals
    avgd_ints = pd.concat((df[["int"]].mean() for df in rel_ints_dfs), axis=1).transpose()
    avgd_ints.drop(columns=["f1", "f2"], inplace=True)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    sns.stripplot(x="expt", y="int",
                  dodge=True, data=all_dfs, ax=ax,
                  palette=sns.color_palette("deep"), **kwargs)
    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")
    # Set y-limits. We need to expand it by ~20% to make space for the legend,
    # as well as the averaged values.
    EXPANSION_FACTOR = 1.2
    ymin, ymax = ax.get_ylim()
    ymean = (ymin + ymax)/2
    ylength = (ymax - ymin)/2
    new_ymin = ymean - (EXPANSION_FACTOR * ylength)
    new_ymax = ymean + (EXPANSION_FACTOR * ylength)
    ax.set_ylim((new_ymin, new_ymax))
    # add the text
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), color) in enumerate(zip(
                expt_avgs.items(), sns.color_palette("deep"))):
            ax.text(x=x+i, y=0.02, s=f"({avg:.2f})",
                    color=color, horizontalalignment="center",
                    transform=ax.get_xaxis_transform())
    pg.style_axes(ax, "plot")
    return plt.gcf(), ax


def make_colorbar(cs, ax):
    """
    Quickly add a colour bar to a contour plot or similar.

    You can get the first argument as the return value of contour() or
    contourf(). imshow() also works. The second argument is the Axes.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)


def enzip(*iterables) -> Generator[tuple, None, None]:
    for i, t in enumerate(zip(*iterables)):
        yield (i, *t)


# -- Styling
def fira() -> None:
    plt.rcParams['font.family'] = 'Fira Sans'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Fira Sans'
    plt.rcParams['mathtext.it'] = 'Fira Sans:italic'
    plt.rcParams['font.size'] = 12
    plt.rcParams['savefig.dpi'] = 600

def source_serif() -> None:
    plt.rcParams['font.family'] = 'Source Serif Pro'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Source Serif Pro'
    plt.rcParams['mathtext.it'] = 'Source Serif Pro'
    plt.rcParams['font.size'] = 12
    plt.rcParams['savefig.dpi'] = 600
