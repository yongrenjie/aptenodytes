import os
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Any, Union, Generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import penguins as pg
from penguins import dataset as ds  # for type annotations


class Experiment:
    """
    Generic interface for experiments.
    """
    default_margin = (0.5, 0.02)   # applicable for 13C experiments
    # This is overridden in subclasses.
    #            use (0.02, 0.02) for 1H experiments
    #            use (0.4, 0.05) for 15N experiments
    def __init__(self,
                 peaks: List[Tuple[float, float]],
                 margin: Optional[Tuple[float, float]] = None,
                 ):
        self.peaks = peaks
        self.margin = margin or self.default_margin

    def integrate(self,
                  dataset: ds.Dataset2D,
                  ) -> np.ndarray:
        # Get absolute peak intensities for a given dataset.
        return np.array([dataset.integrate(peak=peak,
                                           margin=self.margin,
                                           mode="max")
                         for peak in self.peaks])

    def show_peaks(self, ax=None, **kwargs) -> None:
        """
        Draw red crosses corresponding to each peak on an existing Axes
        instance. Useful for checking whether the peaks actually line up with
        the spectrum.

        If 'ax' is not provided, defaults to currently active Axes.

        Other kwargs are passed to ax.scatter().
        """
        if ax is None:
            ax = plt.gca()

        scatter_kwargs = {"color": pg.color_palette("bright")[3],
                          "marker": "+", "zorder": 2}
        scatter_kwargs.update(kwargs)
        ax.scatter([p[1] for p in self.peaks], [p[0] for p in self.peaks],
                   **scatter_kwargs)


    @property
    def df(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing all the peaks. This DF has
        columns "f1" and "f2".
        """
        return pd.DataFrame.from_records(self.peaks, columns=("f1", "f2"))

    def rel_ints_df(self,
                    dataset: ds.Dataset2D,
                    ref_dataset: ds.Dataset2D,
                    label: str = "",
                    ) -> pd.DataFrame:
        """
        Construct a dataframe of relative intensities vs a reference
        dataset.

        This DataFrame will have columns "f1", "f2", "expt", and "int".
        """
        df = pd.DataFrame()
        df["int"] = self.integrate(dataset) / self.integrate(ref_dataset)
        df["expt"] = label
        df["f1"] = self.df["f1"]
        df["f2"] = self.df["f2"]
        return df


class Hmbc(Experiment):
    """
    For 13C HMBC experiments. Just call hmbc(peaks, margin) to instantiate.
    """
    default_margin = (0.5, 0.02)


class NHsqc(Experiment):
    """
    For 15N HSQC experiments. Just call nhsqc(peaks, margin) to instantiate.
    """
    default_margin = (0.4, 0.05)


class Hsqc(Experiment):
    """
    For 13C HSQC experiments. The variables ch, ch2, and ch3 should be
    lists of 2-tuples (f1_shift, f2_shift) which indicate, well, CH, CH2,
    and CH3 peaks respectively.

    None of the methods from Experiment are actually inherited.
    """
    def __init__(self,
                 ch: List[Tuple[float, float]],
                 ch2: List[Tuple[float, float]],
                 ch3: List[Tuple[float, float]],
                 margin: Optional[Tuple[float, float]] = (0.5, 0.02),
                 ):
        self.ch = ch
        self.ch2 = ch2
        self.ch3 = ch3
        self.margin = margin

    @property
    def peaks(self) -> List[Tuple[float, float]]:
        """
        Returns a list of all peaks.
        """
        return self.ch + self.ch2 + self.ch3

    @property
    def df(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing all the peaks. This DF has
        columns "f1", "f2", and "mult".
        """
        _chdf, _ch2df, _ch3df = (
            pd.DataFrame.from_records(peaklist, columns=("f1", "f2"))
            for peaklist in (self.ch, self.ch2, self.ch3)
        )
        _chdf["mult"] = "ch"
        _ch2df["mult"] = "ch2"
        _ch3df["mult"] = "ch3"
        return pd.concat((_chdf, _ch2df, _ch3df), ignore_index=True)

    def integrate(self,
                  dataset: ds.Dataset2D,
                  edited: bool = False,
                  ) -> np.ndarray:
        """
        Calculates the absolute integral of each peak in the HSQC. Assumes that
        CH/CH3 is phased to negative and CH2 to positive.
        """
        if edited:
            # We need self.df here as it contains multiplicity information.
            return np.array([dataset.integrate(peak=(peak.f1, peak.f2),
                                               margin=self.margin,
                                               mode=("max"
                                                     if peak.mult == "ch2"
                                                     else "min"))
                             for peak in self.df.itertuples()])
        else:
            return np.array([dataset.integrate(peak=peak,
                                               margin=self.margin,
                                               mode=("max"))
                             for peak in self.peaks])

    def rel_ints_df(self,
                    dataset: ds.Dataset2D,
                    ref_dataset: ds.Dataset2D,
                    label: str = "",
                    edited: bool = False,
                    ) -> pd.DataFrame:
        """
        Construct a dataframe of relative intensities vs a reference
        dataset.

        This DataFrame will have columns (f1, f2, mult) just like self.df,
        but will also have "expt" which is a string indicating the type of
        experiment being ran, and "int" which is the relative integral vs a
        reference dataset.
        """
        df = pd.DataFrame()
        df["int"] = (self.integrate(dataset, edited=edited) /
                     self.integrate(ref_dataset, edited=edited))
        df["expt"] = label
        df["mult"] = self.df["mult"]
        df["f1"] = self.df["f1"]
        df["f2"] = self.df["f2"]
        return df


class HsqcCosy(Experiment):
    """
    For 13C HSQC-COSY experiments. The variables hsqc and cosy should be lists
    of 2-tuples (f1_shift, f2_shift) which indicate the direct (HSQC) and
    indirect (HSQC-COSY) responses respectively.

    None of the methods from Experiment are actually inherited.
    """
    def __init__(self,
                 hsqc: List[Tuple[float, float]],
                 cosy: List[Tuple[float, float]],
                 margin: Optional[Tuple[float, float]] = (0.5, 0.02),
                 ):
        self.hsqc = hsqc
        self.cosy = cosy
        self.margin = margin

    @property
    def peaks(self) -> List[Tuple[float, float]]:
        """
        Returns a list of all peaks.
        """
        return self.hsqc + self.cosy

    @property
    def df(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing all the peaks. This DF has
        columns "f1", "f2", and "type".
        """
        hsqc_df, cosy_df = (
            pd.DataFrame.from_records(peaklist, columns=("f1", "f2"))
            for peaklist in (self.hsqc, self.cosy)
        )
        hsqc_df["type"] = "hsqc"
        cosy_df["type"] = "cosy"
        return pd.concat((hsqc_df, cosy_df), ignore_index=True)

    def integrate(self,
                  dataset: ds.Dataset2D,
                  edited: bool = True,
                  ) -> np.ndarray:
        """
        Calculates the absolute integral of each peak in the HSQC. If editing
        is enabled, assumes that HSQC peaks are positive and HSQC-COSY peaks
        negative.
        """
        if edited:
            # We need self.df here as it contains multiplicity information.
            return np.array([dataset.integrate(peak=(peak.f1, peak.f2),
                                               margin=self.margin,
                                               mode=("max"
                                                     if peak.type == "hsqc"
                                                     else "min"))
                             for peak in self.df.itertuples()])
        else:
            return np.array([dataset.integrate(peak=peak,
                                               margin=self.margin,
                                               mode=("max"))
                             for peak in self.peaks])

    def rel_ints_df(self,
                    dataset: ds.Dataset2D,
                    ref_dataset: ds.Dataset2D,
                    label: str = "",
                    edited: bool = True,
                    ) -> pd.DataFrame:
        """
        Construct a dataframe of relative intensities vs a reference
        dataset.

        This DataFrame will have columns (f1, f2, mult) just like self.df,
        but will also have "expt" which is a string indicating the type of
        experiment being ran, and "int" which is the relative integral vs a
        reference dataset.
        """
        df = pd.DataFrame()
        df["int"] = (self.integrate(dataset, edited=edited) /
                     self.integrate(ref_dataset, edited=edited))
        df["expt"] = label
        df["type"] = self.df["type"]
        df["f1"] = self.df["f1"]
        df["f2"] = self.df["f2"]
        return df


class Cosy(Experiment):
    """
    For COSY experiments. The variables diagonal and cross_half should be
    lists of 2-tuples (f1_shift, f2_shift). cross_half should only contain
    half the peaks, i.e. only at (f1, f2) and not at (f2, f1). These will
    be automatically reflected.
    
    Only integrate() is actually inherited from Experiment.
    """
    def __init__(self,
                 diagonal: List[Tuple[float, float]],
                 cross_half: List[Tuple[float, float]],
                 margin: Optional[Tuple[float, float]] = (0.02, 0.02),
                 ):
        self.diagonal = diagonal
        self.cross_half = cross_half
        self.margin = margin

    @property
    def cross(self) -> List[Tuple[float, float]]:
        cross_otherhalf = [(t[1], t[0]) for t in self.cross_half]
        # All crosspeaks
        return self.cross_half + cross_otherhalf

    @property
    def peaks(self) -> List[Tuple[float, float]]:
        return self.diagonal + self.cross

    @property
    def df(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing all the peaks. This DF has
        columns "f1", "f2", and "type".
        """
        _diagdf, _crossdf = (
            pd.DataFrame.from_records(peaklist, columns=("f1", "f2"))
            for peaklist in (self.diagonal, self.cross)
        )
        _diagdf["type"] = "diagonal"
        _crossdf["type"] = "cross"
        return pd.concat((_diagdf, _crossdf), ignore_index=True)

    def rel_ints_df(self,
                    dataset: ds.Dataset2D,
                    ref_dataset: ds.Dataset2D,
                    label: str = "",
                    ) -> pd.DataFrame:
        """
        Construct a dataframe of relative intensities vs a reference
        dataset.

        This DataFrame will have columns (f1, f2, type) just like self.df,
        but will also have "expt" which is a string indicating the type of
        experiment being ran, and "int" which is the relative integral vs a
        reference dataset.
        """
        df = pd.DataFrame()
        df["int"] = self.integrate(dataset) / self.integrate(ref_dataset)
        df["expt"] = label
        df["type"] = self.df["type"]
        df["f1"] = self.df["f1"]
        df["f2"] = self.df["f2"]
        return df


class Tocsy(Cosy):
    """
    For TOCSY experiments. The variables diagonal and cross_half should be
    lists of 2-tuples (f1_shift, f2_shift). cross_half should only contain
    half the peaks, i.e. only at (f1, f2) and not at (f2, f1). These will
    be automatically reflected.

    The code is derived from the Cosy code, but for Tocsy classes there is
    additionally a `mixing_time` attribute which gives the TOCSY mixing time in
    milliseconds.
    """
    def __init__(self,
                 diagonal: List[Tuple[float, float]],
                 cross_half: List[Tuple[float, float]],
                 mixing_time: float,
                 margin: Optional[Tuple[float, float]] = (0.02, 0.02),
                 ):
        super().__init__(diagonal, cross_half, margin)
        self.mixing_time = mixing_time
