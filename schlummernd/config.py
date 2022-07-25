# Standard library
import pathlib
from typing import Union

# Third-party
from pydantic import BaseModel
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

    # The number of K-fold splits to do within each neighborhood for assessing uncertainty
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
        filename = pathlib.Path(filename).expanduser().absolute()
        if not filename.exists():
            raise IOError(f"Config file {filename!s} does not exist.")

        with open(filename, 'r') as f:
            vals = yaml.safe_load(f.read())

        vals.setdefault('output_path', filename.parent)
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
                setattr(self, k, v.expanduser().absolute())

                # Make sure path exists:
                v.mkdir(exist_ok=True)
