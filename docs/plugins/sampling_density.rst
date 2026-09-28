Sampling Density
================

The Sampling Density plugin shows how successful configurations are distributed in the encoded
hyperparameter space at a selected budget. Repeated evaluations of the same configuration are
folded before the density is calculated.

Modes
-----

* **Single hyperparameter** displays a one-dimensional marginal density, a uniform-space baseline,
  sample rug marks, and the incumbent value.
* **Hyperparameter pair** displays a two-dimensional histogram with evaluated points and the
  incumbent configuration.

Categorical and integer hyperparameters are shown as discrete frequency masses. Continuous
hyperparameters use a bounded histogram with at most 200 bins in one dimension and 80 bins per
axis in two dimensions. All calculations use DeepCAVE's encoded configuration values.

Filters
-------

The display filters toggle the uniform baseline, one-dimensional rug marks, and two-dimensional
evaluated points. Changing these filters only redraws the figure; it does not recalculate the
density.