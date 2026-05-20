from datasets.adapters.cmapss import make_cmapss_loaders, load_cmapss_as_arrays
from datasets.adapters.ecoating import make_ecoating_loaders, load_ecoating_as_arrays
from datasets.adapters.hydraulic import make_hydraulic_loaders, load_hydraulic_as_arrays
from datasets.adapters.cwru import make_cwru_loaders, load_cwru_as_arrays

__all__ = [
    "make_cmapss_loaders",
    "load_cmapss_as_arrays",
    "make_ecoating_loaders",
    "load_ecoating_as_arrays",
    "make_hydraulic_loaders",
    "load_hydraulic_as_arrays",
    "make_cwru_loaders",
    "load_cwru_as_arrays",
]
