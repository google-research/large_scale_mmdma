# Large-Scale MMD-MA

[**Installation**](#installation)
| [**Examples**](https://github.com/google-research/large_scale_mmdma/blob/master/examples/tutorial101.ipynb)
| [**Command line instructions**](#commandline)
| [**Output**](#output)

The objective of [MMD-MA](https://pubmed.ncbi.nlm.nih.gov/34632462/) is to
match points coming from two different spaces in a lower dimensional space. To
this end, two sets of points are projected, from two different spaces endowed
with a positive definite kernel, to a shared Euclidean space of lower dimension
`low_dim`. The mappings from high to low dimensional space are
obtained using functions belonging to the respective RKHS. To obtain the
mappings, we minimise a [loss function](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/train.py) that is composed of three terms:
- an MMD term between the low dimensional representations of the two views,
which encourages them to have the same distribution. The RBF kernel is used
to compute MMD.
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
when the number of features `p` is larger than the number of samples `n`
`p >> n`, then the dual formulation is beneficial in terms
of runtime and memory, while if `n >> p`, the primal
formulation is favorable.

Additionally, in order to scale the computation of MMD to a large number of
samples, we propose to use the library
[KeOps](https://www.kernel-operations.io/keops/index.html) which
lets you compute large kernel operations on GPUs without memory overflow.

## Installation<a id="installation"></a>

To install the latest release of lsmmdma, use the following command:

```bash
$ pip install lsmmdma
```

To install the **development** version, use the following command instead:

```bash
$ pip install git+https://github.com/google-research/large-scale-mmdma
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

The algorithm can be run with a command line using:

```bash
python3 -m lsmmmda.main [flags]
```

A set of flags is available to determine the IO, the model, the hyperparameters
and the seed.


**Input/Output** It is possible to give as input either a path and filenames pointing to the
user input or to choose to generate data with [data_pipeline.py](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/data/data_pipeline.py). In the
case one wants to generate simulation data, the input flags are:
- `--data`: str, it can be either 'branch', 'triangle' or None (default). The
simulated data is described in the pydoc of [data_pipeline.generate_data](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/data/data_pipeline.py). None
means that simulated data is not used.
- `--n`: int (default None), number of samples for both views.
- `--p`: int (default None), number of features for both views.

For data given by the user, the input flags are:
- `--input_dir`: str (default None), input directory.
- `--input_fv`: str (default None), filename of the array (n1 x p1 or n1 x n1)
that serves as first set of points.
- `--input_sv`: str (default None), filename of the array (n2 x p2 or n2 x n2)
that serves as second set of points.
- `--rd_vec`: str (default None), filename of the vector that contains the indices
of the samples from the first view that match (ground truth) the samples from
the second view. This is only used at evaluation time. If `--rd_vec` is not
used, we assume that the samples of both views follow the same ordering.

In both cases, two other flags are also available:
- `--kernel`: bool (default False), whether the input data given by the user is
a kernel (n x n instead of n x p). This parameter can only be set to True when
`--mode` is 'dual'.
- `--output_dir`: str (default None), output directory.

**Model** The flags allow you to choose between four algorithms, using either
the 'primal' or 'dual' formulation, and using KeOps or not.
- `--mode`: str, either 'primal' or 'dual' (default).
- `--keops`: bool, either True (default) or False.

**Seeds** The seed for the training phase, and for generating the data when
`--data` is not None, is fixed with the flag `--seed` (int, default value is 0).
If one wishes to use multipe starts when training (X seeds and selection
of the best one based on the value of the loss), it is possible to also define
the number of seeds to try with: `--ns` (int, default value is 1).

**Model hyperparameters** Several hyperparameters ought to be fixed in advance:
- `--d`: int (default 5), dimension of the latent space.
- `--init`: str (default 'uniform,0.,0.1'), initialisation for the learned
parameters. It can be sampled from a 'uniform', 'normal', 'xavier_uniform' or
'xavier_normal' distributions. The parameters of the initialisation functions
are passed to the same flag separated by a coma. See [initializers.py](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/initializers.py) and [train.py](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/train.py).
- `--l1`: float (default 1e-4), hyperparameter in front of both penalty terms.
- `--l2`: float (default 1e-4), hyperparameter in front of both distortion terms.
- `--lr`: float (default 1e-5), learning rate.
- `--s`: float (default 1.0), scale parameter of the RBF kernel in MMD.

**Training and evaluation** Several flags structure the training loop:
- `--e`: int (default 5001), number of epochs for the training process.
- `--ne`: int (default 100), regular interval at which the evaluation (call to
[metrics.SupervisedEvaluation](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/metrics.py))
is done, every 'ne' epochs. 0 means that the results are never evaluated.
- `--nr`: int (default 100), regular interval at which the loss and components
of the loss are recorded, every 'ne' epochs. 0 means that the loss and its
components are never recorded.
- `--pca`: int (default 100), regular interval at which PCA is performed on the
embeddings, every 'pca' epochs. 0 means that PCA is not used on the output.

**Timing** Timing the method is possible when the flag `--time` is set to True
(default False).

We show now three examples of usage of the command line to run the algorithm.

1. To run the algorithm on simulated data from [data_pipeline.py](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/data/data_pipeline.py),
a minimal set of commands is:

```bash
python3 -m lsmmdma.main --output_dir outdir \
--data branch --n 300 --p 400 \
--e 1001 --d 5 --keops False --m dual
```

2. To run the algorithm on simulated data from [data_pipeline.py](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/data/data_pipeline.py), one can
also choose when to record the intermediate results, the hyperparameters and to
allow for 5 multiple starts:

```bash
python3 -m lsmmdma.main --output_dir outdir \
--data branch --n 300 --p 400 \
--k 4 --ns 5 \
--e 1001 --d 5 --nr 100 --ne 100 --keops False --m dual --pca 100 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.0 --init 'uniform,0,0.1'
```

3. To run the algorithm on user input data, in the form n_sample x p_feature.
`--data` should be '' (default) and `--kernel` should be False (default). The
argument `--keops` can be True or False, `--mode` can be 'dual' or 'primal'.

```bash
python3 -m lsmmdma.main --input_dir datadir --output_dir outdir \
--input_fv my_data_1 --input_sv my_data_2 --kernel False \
--k 4 --ns 5 \
--e 1001 --d 5 --nr 100 --ne 100 --keops True --m dual --pca 100 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.0 --init 'uniform,0,0.1'
```

4. To run the algorithm on user kernel data, in the form n_sample x n_sample.
`--data` should be '' (default) and `--kernel` should be True. The
argument `--keops` can be True or False, `--mode` can only be 'dual'.

```bash
python3 -m lsmmdma.main --input_dir datadir --output_dir outdir \
--input_fv my_data_1 --input_sv my_data_2 --kernel True \
--k 4 --ns 5 \
--e 1001 --d 5 --nr 100 --ne 100 --keops True --m dual --pca 100 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.0 --init 'uniform,0,0.1'
```

## Output<a id="output"></a>

When one uses [main.py](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/main.py),
several files are saved in the output directory
`FLAGS.output_dir`:
- `[key:val].tsv`: results of the model at the last epoch.
- `[key:val]_tracking.json`: loss and its components during training,
evaluation metrics during training, seed, number of epochs.
- `[key:val]_model.json`: model and optimiser state dictionaries,
loss, number of epochs, seed.
- `[key:val]_pca.npy`: 2D representation obtained with PCA on the
embeddings during training.
- `[key:val]_embeddings.npy`: embeddings during training.
- `generated_data_X`: first view, second view and `rd_vec`
generated by [data_pipeline.generate_data](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/data/data_pipeline.py).

`[key:val]` represent a list of key:value as determined by the flags and written
in [main.py](https://github.com/google-research/large_scale_mmdma/blob/master/lsmmdma/main.py).

## Disclaimer

This is not an officially supported Google product.
