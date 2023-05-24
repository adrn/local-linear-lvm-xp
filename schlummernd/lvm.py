# Standard library
from dataclasses import dataclass
from typing import Annotated, Any, Dict, get_args

# Third-party
import jax.numpy as jnp
import numpy as np


def _model_linear(mu, A, x):
    return mu[None] + x @ A.T


@dataclass
class ParameterState:
    sizes: Dict
    A: Annotated[Any, ("R", "D")]  # noqa
    B: Annotated[Any, ("Q", "D")]  # noqa
    z: Annotated[Any, ("N", "D")]  # noqa
    mu_X: Annotated[Any, ("R",)]  # noqa
    mu_y: Annotated[Any, ("Q",)]  # noqa

    def __post_init__(self):
        for name, field in self.__dataclass_fields__.items():
            if name == "sizes":
                continue

            runtime_shape = get_args(field.type)[1]
            shape = tuple([self.sizes[x] for x in runtime_shape])

            got_shape = getattr(self, name).shape
            if got_shape != shape:
                raise ValueError(
                    f"Invalid shape for {name}: expected {runtime_shape}={shape}, got "
                    f"{got_shape}"
                )

    @property
    def names(self):
        return [x for x in self.__dataclass_fields__.keys() if x != "sizes"]


class LinearLVM:
    def __init__(
        self,
        X,
        y,
        X_icov,
        y_icov,
        n_latents,
        alpha,
        beta,
        verbose=False,
        rng=None,
    ):
        """
        N - number of stars
        R - XP spectra (BP/RP spectral coefficients, photometry, etc.), aka "features"
        Q - stellar parameters (Teff, logg, [Fe/H], [alpha/Fe], etc.), aka "labels"
        D - number of latent variables in the model

        Parameters
        ----------
        X : array-like
            shape `(N, R)` array of training features
        y : array-like
            shape `(N, Q)` array of training labels
        X_icov : array-like
            shape `(N, R, R)` inverse-variance matrix array, or `(N, R)` array of error
            inverse variances for the features (i.e. NOT THE STANDARD DEVIATIONS)
        y_icov : array-like
            shape `(N, Q, Q)` inverse-variance matrix array, or `(N, Q)` array of error
            inverse variances for the labels (i.e. NOT THE STANDARD DEVIATIONS)
        n_latents : int
            sets the number of latent variables, i.e., `D` in the definitions above
        alpha : numeric
            regularization strength for elements of matrix `A`
        beta : numeric
            regularization strength for unconstrained parts of matrix `B`
        """
        self.verbose = verbose
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.X = jnp.array(X)
        self.y = jnp.array(y)
        self.X_icov = jnp.array(X_icov)
        self.y_icov = jnp.array(y_icov)

        self.sizes = {}
        self.sizes["N"], self.sizes["R"] = self.X.shape
        self.sizes["Q"] = self.y.shape[1]

        shp_msg = "Invalid shape for {object_name}: got {got}, expected {expected})"
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                shp_msg.format(
                    object_name="training labels y",
                    got=self.y.shape[0],
                    expected=self.X.shape[0],
                )
            )

        for K, name in zip(["R", "Q"], ["X", "y"]):
            expected_cov_shape = getattr(self, name).shape + (self.sizes[K],)
            val_shape = getattr(self, name).shape
            err_shape = getattr(self, f"{name}_icov").shape
            if len(err_shape) == 2 and val_shape != err_shape:
                raise ValueError(
                    shp_msg.format(
                        object_name=f"{name}_icov", got=err_shape, expected=val_shape
                    )
                )
            elif len(err_shape) == 3 and err_shape != expected_cov_shape:
                raise ValueError(
                    shp_msg.format(
                        object_name=f"{name}_icov",
                        got=err_shape,
                        expected=expected_cov_shape,
                    )
                )
            elif len(err_shape) not in [2, 3]:
                raise ValueError(
                    shp_msg.format(
                        object_name=f"{name}_icov",
                        got=err_shape,
                        expected=f"{val_shape} or {expected_cov_shape}",
                    )
                )

        # First square block of B is the identity matrix, so some latents are
        # semi-interpretable:
        # (B turned into a Jax array below)
        self.sizes["D"] = int(n_latents)
        B = np.zeros((self.sizes["Q"], self.sizes["D"]))
        B[: self.sizes["Q"], : self.sizes["Q"]] = np.eye(self.sizes["Q"])
        self._z_fit_mask = np.arange(self.sizes["Q"], self.sizes["D"])

        self._B_init = B
        if verbose:
            print(f"B = {B}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        if np.any(np.array([self.alpha, self.beta]) < 0):
            raise ValueError("You must regularize positively.")

        self.par_state = self.initialize_par_state()

    def _chi2_X(self, mu_X, A, z):
        dX = self.X - _model_linear(mu_X, A, z)
        if self.X_icov.ndim == 3:
            return jnp.einsum("ni,nij,nj->n", dX, self.X_icov, dX)
        else:
            return jnp.sum(dX**2 * self.X_icov, axis=1)

    def _chi2_y(self, mu_y, B, z):
        dy = self.y - _model_linear(mu_y, B, z)
        if self.y_icov.ndim == 3:
            return jnp.einsum("ni,nij,nj->n", dy, self.y_icov, dy)
        else:
            return jnp.sum(dy**2 * self.y_icov, axis=1)

    def chi2(self, params):
        p = ParameterState(self.sizes, **params)
        chi2 = (
            self._chi2_X(p.mu_X, p.A, p.z)
            + self._chi2_y(p.mu_y, p.B, p.z)
            + self.alpha * jnp.sum(p.A**2)
            + self.beta * jnp.sum(p.B[:, self._z_fit_mask] ** 2)
        )
        return chi2

    def chi(self, params):
        return jnp.sqrt(self.chi2(params))

    def objective(self, params):
        return 0.5 * jnp.sum(self.chi2(params))

    def __call__(self, params):
        return self.objective(params)

    def initialize_par_state(self, **state):
        """
        N - stars
        R - features
        Q - labels
        D - latents

        mu_X : (R, )
        mu_y : (Q, )
        z : (N, D)
        A : (R, D)
        B : (Q, D)

        """

        # Initialize the means using invvar weighted means
        # TODO: could do sigma-clipping here to be more robust
        if "mu_X" not in state:
            X_ivar = self.X_icov if self.X_icov.ndim == 2 else np.diag(self.X_icov)
            state["mu_X"] = np.sum(self.X * X_ivar, axis=0) / np.sum(X_ivar, axis=0)

        if "mu_y" not in state:
            y_ivar = self.y_icov if self.y_icov.ndim == 2 else np.diag(self.y_icov)
            state["mu_y"] = np.sum(self.y * y_ivar, axis=0) / np.sum(y_ivar, axis=0)

        if "z" not in state:
            # First hack: Start with the pseudo-inverse of `B`.
            state["z"] = np.zeros((self.sizes["N"], self.sizes["D"]))

            # TODO: assume icov comes in
            y_err = (
                np.sqrt(1 / np.diag(self.y_var))
                if self.y_var.ndim == 3
                else np.sqrt(self.y_var)
            )
            chi = (self.y - state["mu_y"][None]) / y_err
            for n in range(self.sizes["N"]):
                state["z"][n] = np.linalg.lstsq(
                    self._B_init / self.y_err[n][:, None], chi[n], rcond=None
                )[0].T

            # Second hack: Add some noise to unconstrained z components
            # TODO: I think this is adding noise to the *constrained* z components
            sigma = np.std(state["z"][:, ~self._z_fit_mask], axis=0)
            scale = 0.1  # TODO: MAGIC NUMBER
            state["z"][:, ~self._z_fit_mask] += self.rng.normal(
                0, scale * sigma, size=(self.sizes["N"], (~self._z_fit_mask).sum())
            )

            state["z"][:, self._z_fit_mask] = self.rng.normal(
                0,
                scale * np.mean(sigma),
                size=(self.sizes["N"], self._z_fit_mask.sum()),
            )

        if "A" not in state:
            state["A"] = np.zeros((self.sizes["R"], self.sizes["D"]))

            # TODO: should I do this in a less hacky way?
            X_err = (
                np.sqrt(1 / np.diag(self.X_var))
                if self.X_var.ndim == 3
                else np.sqrt(self.X_var)
            )
            chi = (self.X - state["mu_X"][None]) / X_err
            for r in range(self.sizes["R"]):
                state["A"][r] = np.linalg.lstsq(
                    state["z"] / X_err[:, r : r + 1], chi[:, r], rcond=None
                )[0]

        renorm = np.sqrt(np.sum(state["A"][:, self._z_fit_mask] ** 2, axis=0))
        state["A"][:, self._z_fit_mask] = (
            state["A"][:, self._z_fit_mask] / renorm[None, :]
        )
        state["z"][:, self._z_fit_mask] = (
            state["z"][:, self._z_fit_mask] * renorm[None, :]
        )

        if "B" not in state:
            state["B"] = self._B_init.copy()
            chi = np.sqrt(self._chi2_y(state["mu_y"], state["B"], state["z"]))

            for i in self._B_fit_idx_cols:
                state["B"][:, i] = np.linalg.lstsq(state["z"], chi[:, i], rcond=None)[0]

        return ParameterState(sizes=self.sizes, **state)

    def predict_y(self, X, X_icov, par_state=None):
        if par_state is None:
            par_state = self.par_state

        # should this use the regularization matrix? Hogg thinks not.
        M = X.shape[0]
        if X.shape[1] != self.sizes["R"]:
            raise ValueError("Invalid shape for input feature matrix X")

        y_hat = np.zeros((M, self.sizes["Q"]))
        chi = np.einsum("nij,ni->nj", X_icov, (X - par_state.mu_X[None]))
        for i, dX in enumerate(chi):
            M = X_icov[i] @ par_state.A
            z = np.linalg.lstsq(M, dX, rcond=None)[0]
            y_hat[i] = par_state.mu_y + par_state.B @ z

        return y_hat

    def predict_X(self, y, y_icov, par_state=None):
        if par_state is None:
            par_state = self.par_state

        M = y.shape[0]
        if y.shape[1] != self.sizes["Q"]:
            raise ValueError("Invalid shape for input label array y")

        X_hat = np.zeros((M, self.sizes["R"]))
        chi = np.einsum("nij,ni->nj", y_icov, (y - par_state.mu_y[None]))
        for i, dy in enumerate(chi):
            M = y_icov[i] @ par_state.B
            z = np.linalg.lstsq(M, dy, rcond=None)[0]
            X_hat[i] = par_state.mu_X + par_state.A @ z

        return X_hat
