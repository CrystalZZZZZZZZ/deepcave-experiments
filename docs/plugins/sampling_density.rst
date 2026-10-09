Sampling Density
================

The Sampling Density plugin shows how successful configurations are distributed in the original
hyperparameter value space at a selected budget. Repeated evaluations of the same configuration
are folded before the density is calculated.

Visualization
-------------

The selected hyperparameter is displayed as a one-dimensional marginal density, with an optional
uniform baseline, sample rug marks, and the incumbent value.

Categorical and integer hyperparameters are shown as discrete frequency masses. Continuous
hyperparameters use a bounded histogram with at most 200 bins. All calculations use the original
configuration values rather than encoded values.

Filters
-------

The display filters toggle the uniform baseline and one-dimensional rug marks. Changing these
filters only redraws the figure; it does not recalculate the density.