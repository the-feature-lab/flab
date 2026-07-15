# flab

Library for Feature Lab empirics.

For local development:
```
# Run this when starting a new project with uv
uv add "flab @ git+https://github.com/the-feature-lab/flab"
# Run these when the flab repo gets updated -- otherwise uv's local copy doesn't sync.
uv lock --upgrade-package flab
uv sync
```

For colab:
```
!pip install git+https://github.com/the-feature-lab/flab
```

Then the following should work:
```
from flab.empirics import rcsetup
from flab.mupify import mupify
from flab.prismatic.hea import generate_hea_monomials
# etc
```

## Structure

```
flab/
├── data/               # Dataset utilities
│   ├── hermite.py          # Synthetic data generation (power-law, polynomial/Hermite targets)
│   └── vision.py           # Image dataset loading, downloading, and preprocessing
│
├── prismatic/          # Tools for polynomial/Hermite feature analysis
│   ├── hea.py              # Monomial representation, HEA eigenvalue computation, and coeff loading
│   ├── deprecated.py       # Deprecated monomial target/grouping helpers
│   └── utils.py            # Lower-level array/tensor utilities and seeding
│
├── devices.py          # Device management and tensor/array conversion
├── empirics.py         # Plotting, experiment measurements (ExptTrace), and file I/O (FileManager)
├── krr.py              # Kernel ridge regression and eigenlearning theory
├── models.py           # Neural network architectures (MLP, CNN, ExpanderMLP)
└── mupify.py           # Maximal Update Parametrization (muP) utilities
```
