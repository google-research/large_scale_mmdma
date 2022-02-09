# Large-Scale MMD-MA

[**Installation**](#installation)
| [**Examples**](https://github.com/google-research/large-scale-mmdma/tree/master/examples)

The objective of [MMD-MA](https://pubmed.ncbi.nlm.nih.gov/34632462/) is to
match points coming from two different spaces in a lower dimensional space. To
this end, two sets of points are projected, from two different spaces endowed
with a positive definite kernel, to a shared Euclidean space of lower dimension
`low_dim`. The mappings from high to low dimensional space are
obtained using functions belonging to the respective RKHS. To obtain the
mappings, we minimise a [loss function](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/train.py) that is composed of three terms:
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

## Installation<a id="installation"></a>

To install the latest release of lsmmdma, use the following command:

```bash
$ pip install lsmmdma
```

To install the **development** version, use the following command instead:

```bash
$ pip install git+https://github.com/google-research/lsmmdma
```

Alternatively, it can be installed from sources with the following command:

```bash
$ python setup.py install
```

In Google Colab, use the following command:
```bash
$ !pip install lsmmdma
```

## Disclaimer

This is not an officially supported Google product.
