# Third-party
import numpy as np
from gala.util import atleast_2d

__all__ = ["Features", "Labels"]


class Features:
    """
    TODO: currently ignoring uncertainties. In principle, we should also keep track of
    covariance matrices or at least variances for the features.
    """

    def __init__(
        self,
        bp=None,
        bp_err=None,
        bp_scale=1.0,
        rp=None,
        rp_err=None,
        rp_scale=1.0,
        apply_scale=False,
        **other_features,
    ):
        # TODO: This interface sucks, with the scales and shit. Also it doesn't error
        # propagate the uncertainty on the scale factor (typically rp[0])
        if bp is None:
            bp = []
            bp_err = []
        self.bp_scale = np.array(bp_scale)
        self.bp = np.asarray(atleast_2d(bp, insert_axis=1))
        if self.bp_scale.shape == ():
            self.bp_scale = np.full_like(self.bp, self.bp_scale)
        elif self.bp_scale.ndim != 2 and self.bp_scale.shape[0] == self.bp.shape[0]:
            self.bp_scale = self.bp_scale[:, None]
        self.bp_err = np.asarray(atleast_2d(bp_err, insert_axis=1))
        if apply_scale:
            self.bp = self.bp / self.bp_scale
            self.bp_err = self.bp_err / self.bp_scale

        if rp is None:
            rp = []
            rp_err = []
        self.rp_scale = np.array(rp_scale)
        self.rp = np.asarray(atleast_2d(rp, insert_axis=1))
        if self.rp_scale.shape == ():
            self.rp_scale = np.full_like(self.rp, self.rp_scale)
        elif self.rp_scale.ndim != 2 and self.rp_scale.shape[0] == self.rp.shape[0]:
            self.rp_scale = self.rp_scale[:, None]
        self.rp_err = np.asarray(atleast_2d(rp_err, insert_axis=1))
        if apply_scale:
            self.rp = self.rp / self.rp_scale
            self.rp_err = self.rp_err / self.rp_scale

        word = " scaled" if not np.all(np.atleast_1d(self.bp_scale) == 1.0) else ""
        self._bp_names = np.array(
            [f"BP[{i}]{word}" for i in range(1, self.bp.shape[1] + 1)]
        )
        word = " scaled" if not np.all(np.atleast_1d(self.rp_scale) == 1.0) else ""
        self._rp_names = np.array(
            [f"RP[{i}]{word}" for i in range(1, self.rp.shape[1] + 1)]
        )

        self._features = {}
        self._features_err = {}
        for k, v in other_features.items():
            self._features[k] = np.asarray(v[0])
            self._features_err[k] = np.asarray(v[1])

        # HACK: assumption that in the neighborhoods, we only use coeffs
        self.X_tree = np.hstack((self.bp, self.rp))

        X = np.hstack(
            [self.bp, self.rp]
            + [atleast_2d(x, insert_axis=1) for x in self._features.values()]
        )
        self.X = X

        Xerr = np.hstack(
            [self.bp_err, self.rp_err]
            + [atleast_2d(x, insert_axis=1) for x in self._features_err.values()]
        )
        self.X_err = Xerr

        self.names = np.concatenate(
            (self._bp_names, self._rp_names, list(self._features.keys()))
        )

    @classmethod
    def from_gaiadata(cls, g, n_bp=None, n_rp=None, **other_features):
        """
        TODO: describe in more detail

        This scales all coefficients by rp0
        """

        if n_bp is None:
            n_bp = 1000  # arbitrary big number
        if n_rp is None:
            n_rp = 1000  # arbitrary big number

        bp_scale = g.rp[:, 0]  # TODO: HARDCODED
        rp_scale = g.rp[:, 0]  # TODO: HARDCODED

        n_xp = 0
        if n_bp == 0:
            bp = None
            bp_err = None
        else:
            j = min(g.bp.shape[1], n_bp + 1)
            bp = g.bp[:, 0:j]
            bp_err = g.bp_err[:, 0:j]
            n_xp += bp.shape[1]

        if n_rp == 0:
            rp = None
            rp_err = None
        else:
            j = min(g.rp.shape[1], n_rp)
            rp = g.rp[:, 1:j]
            rp_err = g.rp_err[:, 1:j]
            n_xp += rp.shape[1]

        return cls(
            bp=bp,
            bp_err=bp_err,
            rp=rp,
            rp_err=rp_err,
            bp_scale=bp_scale,
            rp_scale=rp_scale,
            **other_features,
            apply_scale=True,
        )

    def slice_bp(self, K):
        return self.__class__(
            self.bp[:, :K],
            self.bp_err[:, :K],
            self.bp_scale,
            self.rp,
            self.rp_err,
            self.rp_scale,
            **{k: (self._features[k], self._features_err[k]) for k in self._features},
            apply_scale=False,
        )

    def slice_rp(self, K):
        return self.__class__(
            self.bp,
            self.bp_err,
            self.bp_scale,
            self.rp[:, :K],
            self.rp_err[:, :K],
            self.rp_scale,
            **{k: (self._features[k], self._features_err[k]) for k in self._features},
            apply_scale=False,
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = slice(slc, slc + 1)

        return self.__class__(
            bp=self.bp[slc],
            bp_err=self.bp_err[slc],
            bp_scale=self.bp_scale[slc],
            rp=self.rp[slc],
            rp_err=self.rp_err[slc],
            rp_scale=self.rp_scale[slc],
            **{
                k: (self._features[k][slc], self._features_err[k][slc])
                for k in self._features
            },
            apply_scale=False,
        )

    def X_to_features(self, X):
        features = {}

        i = 0
        if self.bp is not None:
            features["bp"] = X[:, i : i + self.bp.shape[1]] * self.bp_scale
            i += self.bp.shape[1]

        if self.rp is not None:
            features["rp"] = X[:, i : i + self.rp.shape[1]] * self.rp_scale
            i += self.rp.shape[1]

        print(i)
        for j, name in enumerate(self._features.keys(), start=i):
            features[name] = X[:, j]

        return features


class Labels:
    def __init__(self):
        self._vals = {}
        self.plot_labels = {}
        self._icov_parts = []
        self._y = None
        self._y_Cinv = None

        # value percentiles used to scale the label values below
        self._percs = None

        # number of stars
        self.N = None
        self.Q = 0

    def add_label(self, name, value, var, plot_label=None):
        """
        Parameters
        ----------
        name : str
            the name of the label (e.g., '[Fe/H]' or 'logg')
        value : array-like
            an array of values for the label
        var : array-like
            an array of error variances (square of the 'error')
        plot_label : str (optional)
            used to label axes when making plots of the label values
        """
        if plot_label is None:
            plot_label = name

        value = np.array(value)
        var = np.array(var)

        if self.N is None:
            self.N = len(value)

        if value.ndim != 1 or value.shape[0] != self.N or var.shape != value.shape:
            raise ValueError("Invalid shape for input `value` or `var`.")

        self._vals[name] = value
        self._icov_parts.append(1 / var)
        self.plot_labels[name] = plot_label
        self.Q += 1

    def add_label_group(self, names, values, cov, plot_labels=None):
        """
        Parameters
        ----------
        names : array-like of str
            the names of the labels
        values : array-like
            an array of values for the labels, with shape `(N, Q)`
        var : array-like
            an array of covariance matrices for the labels, with shape `(N, Q, Q)`
        plot_labels : str (optional)
            used to label axes when making plots of the label values
        """
        if plot_labels is None:
            plot_labels = names

        values = np.array(values)
        cov = np.array(cov)

        if values.ndim == 1:
            raise RuntimeError("Use .add_label() for adding 1D labels")

        if self.N is None:
            self.N = len(values)

        if (
            values.ndim != 2
            or values.shape[0] != self.N
            or cov.shape[:2] != values.shape
            or cov.shape[1] != cov.shape[2]
            or len(names) != values.shape[1]
        ):
            raise ValueError("Invalid shape for input `value` or `cov`.")

        for i, name in enumerate(names):
            self._vals[name] = values[:, i]
            self.plot_labels[name] = plot_labels[i]

        self._icov_parts.append(np.linalg.inv(cov))
        self.Q += len(names)

    def _make_icov(self):
        Cinv = np.zeros((self.N, self.Q, self.Q))

        i = 0
        for part in self._icov_parts:
            K = part.shape[1] if part.ndim > 1 else 1
            Cinv[:, i : i + K, i : i + K].flat = part
            i += K

        return Cinv

    def _transform(self, vals, icov=False):
        if self._percs is None:
            self._percs = {}
            for name, val in self._vals.items():
                self._percs[name] = np.nanpercentile(val, [50, 16, 84])

        if not icov:
            new_vals = {}
            for name, perc in self._percs.items():
                new_vals[name] = (vals[name] - perc[0]) / (perc[2] - perc[1])
        else:
            # Scale the inverse variance matrix
            Minv = np.diag([(perc[2] - perc[1]) for perc in self._percs.values()])
            Cinv = self._make_icov()
            new_vals = np.einsum("ik,nkl,lj->nij", Minv, Cinv, Minv)

        return new_vals

    def _untransform(self, ys, icov=False):
        if self._percs is None:
            self._percs = {}
            for name, val in self.vals.items():
                self._percs[name] = np.nanpercentile(val, [50, 16, 84])

        if not icov:
            new_vals = {}
            for i, (name, perc) in enumerate(self._percs.items()):
                new_vals[name] = ys[:, i] * (perc[2] - perc[1]) + perc[0]

        else:
            # Scale the inverse variance matrix
            Minv = np.diag([1 / (perc[2] - perc[1]) for perc in self._percs.values()])
            Cinv = self._make_icov()
            new_vals = np.einsum("ik,nkl,lj->nij", Minv, Cinv, Minv)

        return new_vals

    @property
    def y(self):
        if self._y is None or self._y.shape[1] != len(self._vals):
            new_vals = self._transform(self._vals)
            self._y = np.stack(list(new_vals.values())).T.astype(np.float64)
        return self._y

    @property
    def y_Cinv(self):
        if self._y_Cinv is None or self._y_Cinv.shape[1] != len(self._vals):
            new_Cinv = self._transform(self._make_icov(), icov=True)
            self._y_Cinv = new_Cinv.astype(np.float64)
        return self._y_Cinv

    # def __getitem__(self, slc):
    #     if isinstance(slc, int):
    #         slc = slice(slc, slc + 1)

    #     new_obj = self.__class__()
    #     for name in self._vals:
    #         new_obj.add_label(
    #             name, self.vals[name][slc], self.errs[name][slc], self.plot_labels[name]
    #         )
