This is not an officially supported Google product.

In MMD-MA ([Liu et al. 2019, Jointly Embedding Multiple
Single-Cell Omics Measurements](https://pubmed.ncbi.nlm.nih.gov/34632462/)), we project two sets of points, from two different spaces endowed
with a positive definite kernel, to a shared Euclidean space of dimension
`low_dim`. The mappings from high to low dimensional space are
obtained using functions belonging to the respective RKHS. To obtain the
mappings, we minimise a loss function that is composed of three terms:
- an MMD term between the low dimensional representations of the two views,
which encourages them to have the same distribution.
- two non-collapsing penalty terms (corresponding to the `pen_dual` or
`pen_primal` functions), one for each view. These terms ensure that
the low dimensional representations are mutually orthogonal, preventing
collapsing.
- two distortion penalties (corresponding to the `dis_dual` or
`dis_primal` functions), one for each view. These terms encourage the
low dimensional representation to obtain the same pairwise structure as
the original views.

MMD-MA can be formulated using either the primal (when we use the linear
kernel in the input spaces) or the dual problem. Each has
advantages or disadvantages depending on the input data. In each view, when the
number of features is larger than the number of samples
p_featureX >> n_sampleX, then the dual formulation is beneficial in terms
of runtime and memory, while if n_sampleX >> p_sampleX, the primal
formulation is favorable.

## Example

Work in progress. See main.py for usage.

