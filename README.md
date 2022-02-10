# Large-Scale MMD-MA

[**Installation**](#installation)
| [**Examples**](https://github.com/google-research/large_scale_mmdma/blob/master/examples/tutorial101.ipynb)
| [**Command line instructions**](#commandline)

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
advantages or disadvantages depending on the input data. For each view,
when the number of features p is larger than the number of samples n
p >> n, then the dual formulation is beneficial in terms
of runtime and memory, while if n >> p, the primal
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

The KeOps library might require to be installed separately in advance, according
to the given [instructions](http://www.kernel-operations.io/keops/python/installation.html).

## Command line instructions<a id="commandline"></a>

1. To run the algorithm on simulated data from [data_pipeline.py](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/data/data_pipeline.py):

python3 -m lsmmdma.main --output_dir outdir \
--data branch --n 300 --p 400 \
--k 4 --ns 5 \
--e 1001 --d 5 --nr 100 --ne 100 --keops True --m dual --pca 100 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.0 --init 'uniform,0,0.1'


2. To run the algorithm on user input data, in the form n_sample x p_feature.
--data should be '' (default value) and --kernel should be False. The
argument --keops can be True or False, --mode can be 'dual' or 'primal'.

python3 -m lsmmdma.main --input_dir datadir --output_dir outdir \
--input_fv my_data_1 --input_sv my_data_2 --kernel False \
--k 4 --ns 5 \
--e 1001 --d 5 --nr 100 --ne 100 --keops True --m dual --pca 100 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.0 --init 'uniform,0,0.1'


3. To run the algorithm on user kernel data, in the form n_sample x n_sample.
--data should be '' (default value) and --kernel should be True. The
argument --keops can be True or False, --mode can only be `dual`.

python3 -m lsmmdma.main --inputdir datadir --output_dir outdir \
--input_fv my_data_1 --input_sv my_data_2 --kernel True \
--k 4 --ns 5 \
--e 1001 --d 5 --nr 100 --ne 100 --keops True --m dual --pca 100 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.0 --init 'uniform,0,0.1'


## Disclaimer

This is not an officially supported Google product.
