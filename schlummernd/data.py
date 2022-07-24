"""
Currently a bag of helper functionality.
TODO: split up and turned into a proper package.
"""

# Standard library
import pathlib

# Third-party
import numpy as np
from pyia import GaiaData
from gala.util import atleast_2d

__all__ = ["load_data", "Features", "Labels"]

# ====
# Data
# ----

# The output of notebooks/Assemble-data.ipynb
default_data_path = pathlib.Path(__file__).absolute().parent / "../data"
default_data_path = (
    default_data_path.resolve() / "apogee-dr17-x-gaia-dr3-xp.fits"
)


def load_data(filters="default", data_path=None):
    """
    Load the APOGEE DR17 x Gaia DR3 cross-match parent sample.

    By default, this function returns a subset of the data that matches our fiducial set
    of quality cuts and sample criteria. To disable this (i.e. to get the full
    cross-matched sample), set `filters=None`. Alternatively, you can pass in a
    dictionary of key-value pairs where the keys are column names and the values are
    ranges to subselect the data to. So, for example, you could pass in
    `filters={'TEFF': (3500, 4500)}` to subselect to only stars with APOGEE TEFF between
    3500 and 4500 K. Pass ``None`` as a value if you want to only set a lower or upper
    bound.

    Parameters
    ----------
    filters : str, dict-like, ``None`` (optional)
        See description above for ways to use this parameter.
    data_path : path-like (optional)
        To override the default data path location, pass a full file path here.
    """

    if data_path is None:
        data_path = default_data_path
    gall = GaiaData(data_path)

    if filters is None or filters is False:
        # Disable filters: return
        return gall

    elif filters == "default":
        # Our default dataset is the upper red giant branch (above the red clump)
        filters = dict(TEFF=(3000, 5200), LOGG=(-0.5, 2.2))
        return load_data(filters, data_path)

    else:
        return gall.filter(**filters)


class Features:
    """
    TODO: currently ignoring uncertainties. In principle, we should also keep track of
    covariance matrices or at least variances for the features.
    """

    def __init__(
        self,
        bp_bp0=None,
        bp_bp0_err=None,
        rp_rp0=None,
        rp_rp0_err=None,
        **other_features,
    ):
        if bp_bp0 is None:
            bp_bp0 = []
            bp_bp0_err = []
        self.bp = np.asarray(atleast_2d(bp_bp0, insert_axis=1))
        self.bp_err = np.asarray(atleast_2d(bp_bp0_err, insert_axis=1))

        if rp_rp0 is None:
            rp_rp0 = []
            rp_rp0_err = []
        self.rp = np.asarray(atleast_2d(rp_rp0, insert_axis=1))
        self.rp_err = np.asarray(atleast_2d(rp_rp0_err, insert_axis=1))

        self._bp_names = np.array(
            [f"BP[{i}]/BP[0]" for i in range(1, self.bp.shape[1] + 1)]
        )
        self._rp_names = np.array(
            [f"RP[{i}]/RP[0]" for i in range(1, self.rp.shape[1] + 1)]
        )

        self._features = {}
        self._features_err = {}
        for k, v in other_features.items():
            self._features[k] = np.asarray(atleast_2d(v[0], insert_axis=1))
            self._features_err[k] = np.asarray(atleast_2d(v[1], insert_axis=1))

        # HACK: assumption that in the neighborhoods, we only use coeffs
        self.X_tree = np.hstack((self.bp, self.rp))

        X = np.hstack((self.bp, self.rp) + tuple(self._features.values()))
        self.X = X

        Xerr = np.hstack(
            (self.bp_err, self.rp_err) + tuple(self._features_err.values())
        )
        self.X_err = Xerr

        self.names = np.concatenate(
            (self._bp_names, self._rp_names, list(self._features.keys()))
        )

    @classmethod
    def from_gaiadata(cls, g, n_bp=None, n_rp=None, **other_features):

        if n_bp is None:
            n_bp = 1000  # arbitrary big number
        if n_rp is None:
            n_rp = 1000  # arbitrary big number

        n_xp = 0
        if n_bp == 0:
            bp = None
            bp_err = None
        else:
            j = min(g.bp.shape[1], n_bp + 1)
            bp = g.bp[:, 1:j] / g.bp[:, 0:1]
            bp_err = (
                np.sqrt(
                    (g.bp_err[:, 1:j] / g.bp[:, 1:j]) ** 2
                    + (g.bp_err[:, 0:1] / g.bp[:, 0:1]) ** 2
                )
                * np.abs(bp)
            )
            n_xp += bp.shape[1]

        if n_rp == 0:
            rp = None
            rp_err = None
        else:
            j = min(g.rp.shape[1], n_rp + 1)
            rp = g.rp[:, 1:j] / g.rp[:, 0:1]
            rp_err = (
                np.sqrt(
                    (g.rp_err[:, 1:j] / g.rp[:, 1:j]) ** 2
                    + (g.rp_err[:, 0:1] / g.rp[:, 0:1]) ** 2
                )
                * np.abs(rp)
            )
            n_xp += rp.shape[1]

        return cls(
            bp_bp0=bp,
            bp_bp0_err=bp_err,
            rp_rp0=rp,
            rp_rp0_err=rp_err,
            **other_features,
        )

    def slice_bp(self, K):
        return self.__class__(
            self.bp[:, :K],
            self.bp_err[:, :K],
            self.rp,
            self.rp_err,
            **self._features,
        )

    def slice_rp(self, K):
        return self.__class__(
            self.bp,
            self.bp_err,
            self.rp[:, :K],
            self.rp_err[:, :K],
            **self._features,
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = slice(slc, slc + 1)

        return self.__class__(
            self.bp[slc],
            self.bp_err[slc],
            self.rp[slc],
            self.rp_err[slc],
            **{k: (v[0][slc], v[1][slc]) for k, v in self._features.items()},
        )


class Labels:

    def __init__(self):
        self.vals = {}
        self.errs = {}
        self.labels = {}
        self._shape = None
        self._percs = None
        self._y = None
        self._y_err = None

    def add_label(self, name, value, err, label=None):
        if label is None:
            label = name

        value = np.array(value)
        err = np.array(err)

        if len(self.vals) == 0:
            self._shape = value.shape

        if value.shape != self._shape or err.shape != self._shape:
            raise ValueError("Invalid shape.")

        self.vals[name] = value
        self.errs[name] = err
        self.labels[name] = label

    def _transform(self, vals, err=False):
        if self._percs is None:
            self._percs = {}
            for name, val in self.vals.items():
                self._percs[name] = np.nanpercentile(val, [16, 50, 84])

        new_vals = {}
        for name, perc in self._percs.items():
            scale = perc[2] - perc[0]
            new_vals[name] = vals[name] / scale
            if not err:
                new_vals[name] = new_vals[name] - perc[1] / scale

        return new_vals

    def _untransform(self, ys, err=False):
        if self._percs is None:
            self._percs = {}
            for name, val in self.vals.items():
                self._percs[name] = np.nanpercentile(val, [16, 50, 84])

        new_vals = {}
        for i, (name, perc) in enumerate(self._percs.items()):
            scale = perc[2] - perc[0]
            new_vals[name] = ys[:, i] * scale
            if not err:
                new_vals[name] = new_vals[name] + perc[1]

        return new_vals

    @property
    def y(self):
        if self._y is None or self._y.shape[1] != len(self.vals):
            new_vals = self._transform(self.vals)
            self._y = np.stack(list(new_vals.values())).T.astype(np.float64)
        return self._y

    @property
    def y_err(self):
        if self._y_err is None or self._y_err.shape[1] != len(self.vals):
            new_errs = self._transform(self.errs, err=True)
            self._y_err = np.abs(np.stack(list(new_errs.values())).T).astype(np.float64)
        return self._y_err

    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = slice(slc, slc + 1)

        new_obj = self.__class__()
        for name in self._ys:
            new_obj.add_label(
                name,
                self.vals[name][slc],
                self.errs[name][slc],
                self.labels[name]
            )
