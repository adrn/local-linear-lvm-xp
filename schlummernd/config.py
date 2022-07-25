# Standard library
import pathlib
from typing import Union

# Third-party
from pydantic import BaseModel
from pyia import GaiaData
import yaml

__all__ = ['Config']


class Config(BaseModel, validate_assignment=True):

    # The paths to write any output (cache files and plots)
    output_path: Union[pathlib.Path, str]
    data_path: Union[pathlib.Path, str] = None  # default: output_path / 'data'
    plot_path: Union[pathlib.Path, str] = None  # default: output_path / 'plots'

    # The number of PCA components to use when defining neighborhoods
    n_neighborhood_pca_components: int

    # The maximum size of a neighborhood used for training
    max_neighborhood_size: int = 4096

    # The number of K-fold splits to do within each neighborhood for assessing
    # uncertainty
    Kfold_K: int = 8

    # Random number seed
    seed: int = 42

    @classmethod
    def parse_yaml(cls, filename):
        """Parse and load a YAML config file.

        Parameters
        ----------
        filename : path-like
            The full path to a YAML config file.
        """
        filename = pathlib.Path(filename).expanduser().resolve()
        if not filename.exists():
            raise IOError(f"Config file {filename!s} does not exist.")

        with open(filename, 'r') as f:
            vals = yaml.safe_load(f.read())

        vals.setdefault('output_path', filename.parent / 'output')
        return cls(**vals)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.plot_path is None:
            self.plot_path = self.output_path / 'plots'

        if self.data_path is None:
            self.data_path = self.output_path / 'data'

        # Normalize paths:
        for k, v in self.dict().items():
            if isinstance(v, pathlib.Path):
                setattr(self, k, v.expanduser().resolve())

                # Make sure path exists:
                v.mkdir(exist_ok=True)

    def load_training_data(self, filters="default", filename=None):
        """
        Load the APOGEE DR17 x Gaia DR3 cross-match parent sample.

        By default, this function returns a subset of the data that matches our fiducial
        set of quality cuts and sample criteria. To disable this (i.e. to get the full
        cross-matched sample), set `filters=None`. Alternatively, you can pass in a
        dictionary of key-value pairs where the keys are column names and the values are
        ranges to subselect the data to. So, for example, you could pass in
        `filters={'TEFF': (3500, 4500)}` to subselect to only stars with APOGEE TEFF
        between 3500 and 4500 K. Pass ``None`` as a value if you want to only set a
        lower or upper bound.

        Parameters
        ----------
        filters : str, dict-like, ``None`` (optional)
            See description above for ways to use this parameter.
        filename : path-like (optional)
            To override the default data file location, pass a full file path here.
        """

        if filename is None:
            filename = self.data_path / "apogee-dr17-x-gaia-dr3-xp.fits"
        g = GaiaData(filename)

        if filters is None or filters is False:
            # Disable filters: return
            return g

        elif filters == "default":
            # Our default dataset are things with measured APOGEE stellar parameters
            filters = dict(
                TEFF=(2500, 10000),
                LOGG=(-0.6, 6),
                M_H=(-2.5, 1)
            )
            return self.load_training_data(filters, filename=filename)

        else:
            return g.filter(**filters)
