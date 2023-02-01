"""
main.py
-------

All the functionality is in this one file. They are for personal use, they are
largely undocumented! Use at your own risk!
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Any, Union, Generator
import subprocess
from warnings import warn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import penguins as pg
from penguins import dataset as ds  # for type annotations


# These are pure convenience routines for my personal use.
# Default save 
def save_desktop(dpi=600):
    pg.savefig("/Users/yongrenjie/Desktop/a_plot.png", dpi=dpi)

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
                   palette: Optional[Any] = sns.color_palette("deep"),
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
        for i, ((_, avg), color) in enumerate(zip(expt_avgs.items(), palette)):
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
                   palette: Optional[Any] = sns.color_palette("deep"),
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
                  palette=palette[3:], **kwargs)
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
                expt_avgs.items(), palette[3:])):
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
                        legend: bool = True,
                        title: str = "",
                        edited: bool = False,
                        show_averages: bool = True,
                        separate_mult: bool = True,
                        show_categories: bool = False,
                        ncol: int = 4,
                        loc: str = "upper center",
                        palette: Optional[Any] = sns.color_palette("deep"),
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
    # fix the legend
    if legend:
        # Setting the handles manually ensures that we get stripplot handles
        # rather than the pointplot ones (if present).
        handles, _ = ax.get_legend_handles_labels()
        l = ax.legend(ncol=ncol, loc=loc,
                      markerscale=0.4,
                      handles=handles[0:4],
                      labels=["HSQC CH", r"HSQC CH$_2$", r"HSQC CH$_3$", "COSY"])
        l.set(title=None)
    else:
        ax.get_legend().remove()

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
    categories = ["CH", "CH$_2$", "CH$_3$", "COSY"]
    x0 = -0.3 if separate_mult else -0.2
    dx = 0.2 if separate_mult else 0.4
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), clr, cat) in enumerate(zip(expt_avgs.items(),
                                                      palette, categories)):
            ax.text(x=x+x0+i*dx, y=0.02, s=f"({avg:.2f})",
                    color=clr, horizontalalignment="center",
                    transform=ax.get_xaxis_transform(),
                    **font_kwargs)
            if show_categories:
                ax.text(x=x+x0+i*dx, y=0.09, s=cat,
                        color=clr, horizontalalignment="center",
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
                    palette: Optional[Any] = sns.color_palette("deep"),
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
        for i, ((_, avg), color) in enumerate(zip(expt_avgs.items(), palette)):
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
                      legend: bool = True,
                      palette: Optional[Any] = sns.color_palette("deep"),
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

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    sns.stripplot(x="expt", y="int",
                  dodge=True, data=all_dfs, ax=ax,
                  palette=palette, **kwargs)
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
                expt_avgs.items(), palette)):
            ax.text(x=x+i, y=0.02, s=f"({avg:.2f})",
                    color=color, horizontalalignment="center",
                    transform=ax.get_xaxis_transform())
    if legend:
        ax.legend()
    else:
        ax.get_legend().remove()
    pg.style_axes(ax, "plot")
    return plt.gcf(), ax


def sscc_stripplot(molecule: Any,
                   datasets: Union[ds.Dataset2D, Sequence[ds.Dataset2D]],
                   ref_datasets: ds.Dataset2D,
                   expt_labels: Union[str, Sequence[str]],
                   experiments: Any = None,
                   xlabel: str = "Experiment",
                   ylabel: str = "Intensity",
                   title: str = "",
                   show_averages: bool = True,
                   legend: bool = True,
                   ncol: int = 2,
                   loc: str = "upper center",
                   palette: Optional[Any] = sns.color_palette("deep"),
                   ax: Optional[Any] = None,
                   **kwargs: Any,
                   ) -> Tuple[Any, Any]:
    # Calculate dataframes of relative intensities.
    if experiments is None:
        experiments = [molecule.hsqc, molecule.hsqc, molecule.cosy]
    if len(experiments) != 3:
        raise ValueError('Three experiments must be supplied')
    rel_ints_dfs = [expt.rel_ints_df(dataset=ds, ref_dataset=ref_ds,
                                     label=label)
                    for (ds, ref_ds, expt, label)
                    in zip(datasets, ref_datasets, experiments, expt_labels)]
    # Add an extra label for the hue parameter
    for i, df in enumerate(rel_ints_dfs):
        df['category'] = str(i)
    all_dfs = pd.concat(rel_ints_dfs)

    # Calculate the average integrals by peak type
    avgd_ints = pd.concat((df.mean() for df in rel_ints_dfs), axis=1)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    stripplot_alpha = 0.3 if show_averages else 0.8
    sns.stripplot(x="expt", y="int", hue="category",
                  zorder=0, alpha=stripplot_alpha,
                  dodge=True, data=all_dfs, ax=ax, **kwargs)
    if show_averages:
        sns.pointplot(x="expt", y="int", hue="category", zorder=1,
                      dodge=0.55, data=all_dfs, ax=ax, join=False,
                      markers='_', palette="dark", ci=None, scale=1.25)
    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        l = ax.legend(ncol=ncol, loc=loc,
                      markerscale=0.4,
                      handles=handles[0:3],
                      labels=["direct", "indirect"])
    else:
        ax.get_legend().remove()
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
    for i, color, (_, expt_avgs) in enzip(palette, avgd_ints.items()):
        ax.text(x=-0.26+i*0.26, y=0.02, s=f"({expt_avgs['int']:.2f})",
                color=color, ha="center",
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


# matplotlib 3+2 grid layout
def subplots_2d_32(width=12, height=8):
    fig = pg.figure(figsize=(width, height))
    gspec = gs.GridSpec(ncols=6, nrows=2, figure=fig)
    axs = []
    axs.extend((fig.add_subplot(gspec[0:1, 0:2]),
                fig.add_subplot(gspec[0:1, 2:4]),
                fig.add_subplot(gspec[0:1, 4:6]),
                fig.add_subplot(gspec[1:2, 1:3]),
                fig.add_subplot(gspec[1:2, 3:5])))
    return fig, axs

def subplots2d(nrows=1, ncols=1, scale=None, **kwargs):
    if scale is not None:
        figsize = (ncols * 4 * scale, nrows * 4 * scale)
        kwargs['figsize'] = figsize
    return pg.subplots2d(nrows, ncols, **kwargs)

# -- Styling
def fira() -> None:
    plt.rcParams['font.family'] = 'Fira Sans'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Fira Sans'
    plt.rcParams['mathtext.it'] = 'Fira Sans:italic'
    plt.rcParams['font.size'] = 12
    plt.rcParams['savefig.dpi'] = 600

def source_serif() -> None:
    plt.rcParams['font.family'] = 'Source Serif 4'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Source Serif 4'
    plt.rcParams['mathtext.it'] = 'Source Serif 4:italic'
    plt.rcParams['font.size'] = 12
    plt.rcParams['savefig.dpi'] = 600

def source_sans() -> None:
    plt.rcParams['font.family'] = 'Source Sans 3'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Source Sans 3'
    plt.rcParams['mathtext.it'] = 'Source Sans 3:italic'
    plt.rcParams['font.size'] = 12
    plt.rcParams['savefig.dpi'] = 600

__bright = pg.color_palette('bright')
PAL = list(__bright)
PAL[2] = pg.color_palette('deep')[2]

def thesis() -> None:
    pg.set_palette(PAL)
    # same as fira but with smaller font size
    plt.rcParams['font.family'] = 'Fira Sans'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Fira Sans'
    plt.rcParams['mathtext.it'] = 'Fira Sans:italic'
    plt.rcParams['font.size'] = 10
    plt.rcParams['savefig.dpi'] = 600

# Convenience functions
def label_axes_def(axs, **kwargs):
    return pg.label_axes(axs, fstr='({})', fontweight='semibold', fontsize=10, **kwargs)

def show():
    return plt.show()

def save(fname, svg=False, dpi=600, trim=True):
    """
    Designed to be called as save(__file__). The parameter is assumed to be a
    string (or maybe pathlib.Path) with the absolute path

    Trimming whitespace around the picture requires the `convert` executable
    (from imagemagick) to be installed. Also, it's only applied for png files.
    """
    if svg:
        output_fname = str(fname).replace('.py', '.svg')
        result = plt.savefig(output_fname, transparent=True)
    else:  # assume png
        output_fname = str(fname).replace('.py', '.png')
        result = plt.savefig(output_fname, dpi=dpi, transparent=True)
        if trim:
            try:
                subprocess.run(f'convert "{output_fname}" -trim "{output_fname}"',
                               shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                warn(f"\n\nA CalledProcessError was raised when calling "
                     f"`convert` (exit code: {e.returncode}).\n"
                     f"Please make sure you have installed imagemagick "
                     f"(e.g. using `brew install imagemagick`).\n"
                     f"Captured stderr:\n{e.stderr}")

    return result

def tl():
    return plt.tight_layout()
