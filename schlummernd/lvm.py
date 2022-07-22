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
    A: Annotated[Any, ('R', 'D')]  # noqa
    B: Annotated[Any, ('Q', 'D')]  # noqa
    z: Annotated[Any, ('N', 'D')]  # noqa
    mu_X: Annotated[Any, ('R', )]  # noqa
    mu_y: Annotated[Any, ('Q', )]  # noqa

    def __post_init__(self):
        for name, field in self.__dataclass_fields__.items():
            if name == 'sizes':
                continue

            runtime_shape = get_args(field.type)[1]
            shape = tuple([self.sizes[x] for x in runtime_shape])

            got_shape = getattr(self, name).shape
            if got_shape != shape:
                raise ValueError(
                    f'Invalid shape for {name}: expected {runtime_shape}={shape}, got {got_shape}'
                )

    @property
    def names(self):
        return [x for x in self.__dataclass_fields__.keys() if x != 'sizes']


class LinearLVM:

    def __init__(self, X, y, X_err, y_err, B, alpha, beta,
                 verbose=False, rng=None):
        """
        N - stars
        R - features
        Q - labels
        D - latents

        Parameters
        ----------
        X : array-like
            shape `(N, R)` array of training features
        y : array-like
            shape `(N, Q)` array of training labels
        X_err : array-like
            shape `(N, R)` array of errors (standard deviations) for the features
        y_err : array-like
            shape `(N, Q)` array of errors (standard deviations) for the labels
        B : array-like
            shape `(Q, D)` matrix translating latents to labels.
        alpha : numeric
            regularization strength; use the source, Luke.
        beta : numeric
            burp.
        """
        self.verbose = verbose
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.X = jnp.array(X)
        self.y = jnp.array(y)
        self.X_err = jnp.array(X_err)
        self.y_err = jnp.array(y_err)

        self.sizes = {}
        self.sizes['N'], self.sizes['R'] = self.X.shape
        self.sizes['Q'] = self.y.shape[1]

        shp_msg = "Invalid shape for {object_name}: got {got}, expected {expected})"
        if self.y.shape[0] != self.sizes['N']:
            shp_msg.format(
                object_name="training labels y",
                got=self.y.shape[0],
                expected=self.sizes['N']
            )
        if self.X_err.shape != self.X.shape:
            shp_msg.format(
                object_name="X_err",
                got=self.X_err.shape,
                expected=self.X.shape
            )
        if self.y_err.shape != self.y.shape:
            shp_msg.format(
                object_name="y_err",
                got=self.y_err.shape,
                expected=self.y.shape
            )

        self._X_ivar = 1 / self.X_err**2
        self._y_ivar = 1 / self.y_err**2

        # B turned into a Jax array below
        B = np.array(B, copy=True)
        _, self.sizes['D'] = B.shape
        if B.shape[0] != self.sizes['Q']:
            shp_msg.format(
                object_name="B",
                got=B.shape[0],
                expected=self.sizes['Q']
            )

        # Elements of B that we will fit for should be set to nan in the input B array
        self._B_fit_mask = jnp.isnan(B)
        if not np.any(self._B_fit_mask) and verbose:
            print("no free elements of B")
        elif np.any(self._B_fit_mask):
            B[self._B_fit_mask] = 0.
            if verbose:
                print(f"using {self._B_fit_mask.sum()} free elements of B")
        self.B = jnp.array(B)
        if verbose:
            print(f"B = {B}")
            print(f"B fit elements = {self._B_fit_mask}")

        # Now assess which latents to fit:
        self._z_fit_mask = jnp.all(self.B == 0, axis=0)
        if verbose:
            print(
                f"using {self._z_fit_mask.sum()} unconstrained elements of z, "
                f"out of {self.sizes['D']} latents"
            )

        self.alpha = float(alpha)
        self.beta = float(beta)

        # Regularization matrix:
        self.Lambda = self.alpha * np.diag(self._z_fit_mask.astype(int))
        if verbose:
            print(f"Lambda = {self.Lambda}")
        assert self.alpha > 0., "You must regularize, and strictly positively."

        # TODO:
        self.par_state = self.initialize_par_state()

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
        if 'mu_X' not in state:
            state['mu_X'] = (
                np.sum(self.X * self._X_ivar, axis=0) /
                np.sum(self._X_ivar, axis=0)
            )

        if 'mu_y' not in state:
            state['mu_y'] = (
                np.sum(self.y * self._y_ivar, axis=0) /
                np.sum(self._y_ivar, axis=0)
            )

        if 'z' not in state:
            # First hack: Start with the pseudo-inverse of `B`.
            state['z'] = np.zeros((self.sizes['N'], self.sizes['D']))
            chi = (self.y - state['mu_y'][None]) / self.y_err
            for n in range(self.sizes['N']):
                state['z'][n] = np.linalg.lstsq(
                    self.B / self.y_err[n][:, None],
                    chi[n],
                    rcond=None
                )[0].T

            # Second hack: Add some noise to unconstrained z components
            sigma = np.std(state['z'][:, ~self._z_fit_mask], axis=0)
            scale = 0.1  # MAGIC NUMBER
            state['z'][:, ~self._z_fit_mask] += self.rng.normal(
                0,
                scale * sigma,
                size=(self.sizes['N'], (~self._z_fit_mask).sum())
            )

            state['z'][:, self._z_fit_mask] = self.rng.normal(
                0,
                scale * np.mean(sigma),
                size=(self.sizes['N'], self._z_fit_mask.sum())
            )

        if 'A' not in state:
            state['A'] = np.zeros((self.sizes['R'], self.sizes['D']))
            chi = self._chi_X(state['mu_X'], state['A'], state['z'])

            for r in range(self.sizes['R']):
                state['A'][r] = np.linalg.lstsq(
                    state['z'] / self.X_err[:, r:r+1],
                    chi[:, r],
                    rcond=None
                )[0]

        renorm = np.sqrt(np.sum(state['A'][:, self._z_fit_mask]**2, axis=0))
        state['A'][:, self._z_fit_mask] = state['A'][:, self._z_fit_mask] / renorm[None, :]
        state['z'][:, self._z_fit_mask] = state['z'][:, self._z_fit_mask] * renorm[None, :]

        if 'B' not in state:
            # TODO: implement this
            state['B'] = self.B

        return ParameterState(sizes=self.sizes, **state)

    def _chi_X(self, mu_X, A, z):
        return (self.X - _model_linear(mu_X, A, z)) / self.X_err

    def _chi_y(self, mu_y, B, z):
        return (self.y - _model_linear(mu_y, B, z)) / self.y_err

    def unpack_p(self, p):
        """
        TODO: deal with some of B is frozen
        """
        i = 0
        state = {}
        for name in self.par_state.names:
            if name == 'B':
                # TODO: see note above
                state['B'] = self.par_state.B
                continue

            val = getattr(self.par_state, name)
            state[name] = p[i:i+val.size].reshape(val.shape)
            i += val.size
        return ParameterState(sizes=self.sizes, **state)

    def pack_p(self, par_state=None):
        """
        TODO: deal with some of B is frozen
        """
        if par_state is None:
            par_state = self.par_state

        arrs = []
        for name in par_state.names:
            if name == 'B':
                # TODO: deal with note above
                continue

            val = getattr(par_state, name).flatten()
            arrs.append(val)
        return jnp.concatenate(arrs)

    def cost(self, p):
        """
        TODO: Regularization term is totally wrong.
        """
        pars = self.unpack_p(p)
        # TODO: set par_state??

        chi_X = self._chi_X(pars.mu_X, pars.A, pars.z)
        chi_y = self._chi_y(pars.mu_y, pars.B, pars.z)

        return 0.5 * (
            jnp.sum(chi_X ** 2) +
            jnp.sum(chi_y ** 2) +
            self.alpha * jnp.sum(pars.z[:, self._z_fit_mask] ** 2) +
            self.beta * jnp.sum(pars.A[:, self._z_fit_mask] ** 2)
        )

    def __call__(self, p):
        val = self.cost(p)
        return val

    def predict_y(self, X, X_err, par_state=None):
        if par_state is None:
            par_state = self.par_state

        # should this use the regularization matrix? Hogg thinks not.
        M = X.shape[0]
        if X.shape[1] != self.sizes['R']:
            raise ValueError("Invalid shape for input feature matrix X")

        y_hat = np.zeros((M, self.sizes['Q']))

        chi = (X - par_state.mu_X[None]) / X_err
        for i, dx in enumerate(chi):
            M = par_state.A / X_err[i][:, None]
            z = np.linalg.lstsq(M, dx, rcond=None)[0]
            y_hat[i] = par_state.mu_y + par_state.B @ z

        return y_hat
