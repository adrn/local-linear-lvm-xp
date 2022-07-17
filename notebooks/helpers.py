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


# ====
# Data
# ----

# The output of notebooks/Assemble-data.ipynb
default_data_path = pathlib.Path(__file__).absolute().parent / '../data'
default_data_path = default_data_path.resolve() / 'apogee-dr17-x-gaia-dr3-xp.fits'


def load_data(filters='default', data_path=None):
    """
    Load the APOGEE DR17 x Gaia DR3 cross-match parent sample.
    
    By default, this function returns a subset of the data that matches our fiducial set of quality cuts and sample criteria. To disable this (i.e. to get the full cross-matched sample), set `filters=None`. Alternatively, you can pass in a dictionary of key-value pairs where the keys are column names and the values are ranges to subselect the data to. So, for example, you could pass in `filters={'TEFF': (3500, 4500)}` to subselect to only stars with APOGEE TEFF between 3500 and 4500 K. Pass ``None`` as a value if you want to only set a lower or upper bound.
    
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
    
    elif filters == 'default':
        # Our default dataset is the upper red giant branch (above the red clump)
        filters = dict(
            TEFF=(3000, 5200),
            LOGG=(-0.5, 2.2)
        )
        return load_data(filters, data_path)
    
    else:
        return gall.filter(**filters)
    

class Features:
    """
    TODO: currently ignoring uncertainties. In principle, we should also keep track of 
    covariance matrices or at least variances for the features.
    """
    
    def __init__(self, bp_bp0=None, rp_rp0=None, **other_features):
        if bp_bp0 is None:
            bp_bp0 = []
        self.bp = np.asarray(atleast_2d(bp_bp0, insert_axis=1))
        
        if rp_rp0 is None:
            rp_rp0 = []
        self.rp = np.asarray(atleast_2d(rp_rp0, insert_axis=1))
        
        self._bp_names = np.array([
            f'BP[{i}]/BP[0]' 
            for i in range(1, self.bp.shape[1] + 1)
        ])
        self._rp_names = np.array([
            f'RP[{i}]/RP[0]' 
            for i in range(1, self.rp.shape[1] + 1)
        ])
        
        self._features = {}
        for k, v in other_features.items():
            self._features[k] = np.asarray(atleast_2d(v, insert_axis=1))
        
        X = np.hstack((self.bp, self.rp) + tuple(self._features.values()))
        self.X = X
        self.names = np.concatenate((
            self._bp_names, self._rp_names, list(self._features.keys())
        ))
    
    @classmethod
    def from_gaiadata(cls, g, n_bp=None, n_rp=None, **other_features):
        
        if n_bp is None:
            n_bp = 1000  # arbitrary big number
        if n_rp is None:
            n_rp = 1000  # arbitrary big number

        n_xp = 0
        if n_bp == 0:
            bp = None
        else:
            j = min(g.bp.shape[1], n_bp + 1)
            bp = g.bp[:, 1:j] / g.bp[:, 0:1]
            n_xp += bp.shape[1]

        if n_rp == 0:
            rp = None
        else:
            j = min(g.rp.shape[1], n_rp + 1)
            rp = g.rp[:, 1:j] / g.rp[:, 0:1]
            n_xp += rp.shape[1]

        return cls(bp_bp0=bp, rp_rp0=rp, **other_features)
    
    def slice_bp(self, K):
        return self.__class__(self.bp[:, :K], self.rp, **self._features)
    
    def slice_rp(self, K):
        return self.__class__(self.bp, self.rp[:, :K], **self._features)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = slice(slc, slc + 1)
            
        return self.__class__(
            self.bp[slc], 
            self.rp[slc], 
            **{k: v[slc] for k, v in self._features.items()}
        )
  