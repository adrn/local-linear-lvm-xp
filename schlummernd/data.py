# Third-party
import astropy.table as at
import numpy as np

__all__ = ["Features", "Labels"]


class BaseData:
    def __init__(self):
        # raw data as inputted:
        self._raw_data = {}
        self._raw_icov = {}

        # value percentiles used to scale the input values below
        self._scales = {}
        self._offsets = {}

        # number of stars
        self.N = None

        # TODO: how to handle plot labels?
        self.plot_labels = {}

    def __len__(self):
        return self.N

    @property
    def names(self):
        return list(self._raw_data.keys())

    def _transform_data(self, vals, inv=False):
        new_vals = {}
        for n, v in vals.items():
            if not inv:
                new_vals[n] = (v - self._offsets[n]) / self._scales[n]
            else:
                new_vals[n] = v * self._scales[n] + self._offsets[n]
        return new_vals

    def _transform_icov(self, vals, inv=False):
        new_vals = {}
        for n, icov in vals.items():
            if not inv:
                M = np.diag(self._scales[n])
            else:
                M = np.diag(1 / self._scales[n])
            new_vals[n] = np.einsum("ik,nkl,lj->nij", M, icov, M)
        return new_vals

    def add(
        self,
        name,
        value,
        err=None,
        cov=None,
        icov=None,
        scale=None,
        offset=None,
        plot_label=None,
    ):
        """
        Pass data values with `value` and either pass in errors (stddev) with `err`, a
        covariance matrix or variance values with `cov`, or inverse-variance matrix with
        `icov`.

        Parameters
        ----------
        name : str
            The name of the feature or label
        value : array-like
            The data values as a numpy array or other data container
        err : array-like (optional)
            Array of standard deviations / errors. Only pass one of err, cov, icov.
        cov : array-like (optional)
            Array of variances or array of covariance matrices. Only pass one of err,
            cov, icov.
        icov : array-like (optional)
            Array of inverse variances or array of inverse-variance matrices. Only pass
            one of err, cov, icov.
        scale : numeric, array-like (optional)
            Either a single value, or an array of scale values per feature/label.
            Default is to take the 84th-16th percentile values for each feature/label.
        offset : numeric, array-like (optional)
            Either a single value, or an array of offset values per feature/label.
            Default is to take the median values for each feature/label.
        """
        name = str(name)
        value = np.array(value)
        if value.ndim > 2:
            raise ValueError("Invalid shape for input `value`")
        elif value.ndim == 1:
            value = value.reshape(-1, 1)
        N, P = value.shape

        if self.N is None:
            self.N = N
        if self.N != N:
            raise ValueError(
                f"Number of data points for input data {N} does not match shape of "
                f"existing data {self.N}"
            )

        if scale is None:
            tmp = np.nanpercentile(value, [16, 84], axis=0)
            scale = tmp[1] - tmp[0]
        self._scales[name] = scale

        if offset is None:
            offset = np.nanpercentile(value, 50, axis=0)
        self._offsets[name] = offset

        if plot_label is None:
            plot_label = name
        self.plot_labels[name] = plot_label

        # First, validate that only one of [err, cov, icov] passed in:
        n_passed = sum([x is None for x in [err, cov, icov]])
        if n_passed < 2:
            raise ValueError(
                "Only one of `err`, `cov`, or `icov` can be passed in at the same time"
            )

        if err is not None:
            # TODO: this should really be a sparse matrix!
            icov = np.zeros((N, P, P))
            err = np.array(err).reshape((N, P))
            icov[:, np.arange(P), np.arange(P)] = 1 / err[:, np.arange(P)] ** 2
        elif cov is not None:
            icov = np.linalg.inv(cov).reshape((N, P, P))
        else:
            icov = np.array(icov).reshape((N, P, P))

        self._raw_data[name] = value
        self._raw_icov[name] = icov

    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = slice(slc, slc + 1)

        elif isinstance(slc, str):
            return np.squeeze(self._raw_data[slc])

        obj = self.__class__()
        for name in self.names:
            new_data = self._raw_data[name][slc]
            new_icov = self._raw_icov[name][slc]
            obj.add(
                name=name,
                value=new_data,
                icov=new_icov,
                plot_label=self.plot_labels[name],
            )

        return obj

    def _make_helper(self, icov=True):
        """
        Helper to create transformed data blocks
        """
        trans_data = self._transform_data(self._raw_data)
        trans_data = np.hstack(list(trans_data.values()))

        if not icov:
            return trans_data

        # TODO: should use sparse matrices
        tmp = self._transform_icov(self._raw_icov)
        full_K = sum([v.shape[1] for v in tmp.values()])

        trans_icov = np.zeros((self.N, full_K, full_K))
        i = 0
        for icov in tmp.values():
            K = icov.shape[1]
            trans_icov[:, i : i + K, i : i + K] = icov
            i += K

        return trans_data, trans_icov

    def from_transformed(self, data, icov=None):
        # expand block array into dict:
        data_dict = {}
        icov_dict = {}

        i = 0
        for name, vals in self._raw_data.items():
            K = vals.shape[1]
            data_dict[name] = data[:, i : i + K]
            if icov is not None:
                icov_dict[name] = icov[:, i : i + K, i : i + K]

        data_dict = self._transform_data(data_dict, inv=True)

        data_tbl = at.Table(data_dict)
        if icov is None:
            return data_tbl

        else:
            icov_dict = self._transform_icov(icov_dict, inv=True)
            return data_tbl, at.Table(icov_dict)


class Features(BaseData):
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

    def make_X(self, icov=True):
        return self._make_helper(icov=icov)

    # TODO: still need these?
    # def slice_bp(self, K):
    #     return self.__class__(
    #         self.bp[:, :K],
    #         self.bp_err[:, :K],
    #         self.bp_scale,
    #         self.rp,
    #         self.rp_err,
    #         self.rp_scale,
    #         **{k: (self._features[k], self._features_err[k]) for k in self._features},
    #         apply_scale=False,
    #     )

    # def slice_rp(self, K):
    #     return self.__class__(
    #         self.bp,
    #         self.bp_err,
    #         self.bp_scale,
    #         self.rp[:, :K],
    #         self.rp_err[:, :K],
    #         self.rp_scale,
    #         **{k: (self._features[k], self._features_err[k]) for k in self._features},
    #         apply_scale=False,
    #     )


class Labels(BaseData):
    def make_y(self, icov=True):
        return self._make_helper(icov=icov)
