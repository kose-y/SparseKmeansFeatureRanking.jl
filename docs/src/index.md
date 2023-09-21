# SparseKmeansFeatureRanking.jl

Sparse K-means via Feature Ranking

This software package contains an efficient multi-threaded implementation of sparse K-means via feature ranking proposed by [Zhang, Lange, and Xu (2020)](https://proceedings.neurips.cc//paper/2020/file/735ddec196a9ca5745c05bec0eaa4bf9-Paper.pdf). The code is based on the [original github repository](https://github.com/ZhiyueZ/SKFR). The authors of the original code have kindly agreed to redistribute the derivative of their code on this repository under the MIT License. 

## Installation

This package requires Julia v1.6 or later, which can be obtained from
<https://julialang.org/downloads/> or by building Julia from the sources in the
<https://github.com/JuliaLang/julia> repository.

The package can be installed by running the following code:
```julia
using Pkg
pkg"add https://github.com/kose-y/SparseKmeansFeatureRanking.jl"
```
For running the examples below, the following are also necessary. 
```julia
pkg"add Random Clustering SnpArrays"
```


```julia
versioninfo()
```

    Julia Version 1.7.1
    Commit ac5cc99908 (2021-12-22 19:35 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin19.5.0)
      CPU: Intel(R) Core(TM) i7-7820HQ CPU @ 2.90GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-12.0.1 (ORCJIT, skylake)


## Basic Usage

First, let us initialize the random number generator for reproducibility. We use `MersenneTwister` to obtain the same result with different number of threads.

!!! Since Julia 1.7, the default random number generator depends on thread launches.


```julia
using Random, SparseKmeansFeatureRanking
rng = MersenneTwister(7542)
```




    MersenneTwister(7542)



Then, we generate a random data with 300 samples and 100 features. For the first 33 features, we add `1.0` to samples `101:200` and `2.0` to samples `201:300` to give a cluster structure. 


```julia
(features, cases) = (100, 300);
(classes, sparsity)  = (3, 33);
X = randn(features, cases);
(m, n) = (div(features, 3), 2 * div(features, 3));
(r, s) = (div(cases, 3) + 1, 2 * div(cases, 3));
X[1:m, r:s] = X[1:m, r:s] .+ 1.0;
X[1:m, s + 1:end] = X[1:m, s + 1:end] .+ 2.0;
```

`ImputedMatrix` is the basic data structure for the SKFR algorithm with k-POD imputation. This can be generated using the function `get_imputed_matrix()`. The second argument `3` is the number of clusters, and the optional keyword argument `rng` determines the status of the random number generator to be used. 


```julia
IM = SparseKmeansFeatureRanking.get_imputed_matrix(collect(transpose(X)), 3; rng=rng);
```

The function `sparsekmeans1()` selects `sparsity` most informative features globally. 


```julia
(classout1, center1, selectedvec1, WSSval1, TSSval1) = SparseKmeansFeatureRanking.sparsekmeans1(IM, sparsity);
```

`classout1` is the class labels, `center1` contains cluster centers, `selectedvec1` contains selected informative features. `WSSval1` shows within-cluster sum of squares value, and `TSSval1` contains total sum of squares. 


```julia
using Clustering
randindex(classout1,[repeat([1], 100); repeat([2], 100); repeat([3], 100)])[1]
```

Checking the rand index gives the value `1.0`, meaning perfect clustering. As expected, 


```julia
all(sort(selectedvec1).== 1:33)
```

Also, the first 33 features are selected, as expected. 

The function `sparsekmeans2()` selects `sparsity` most informative feature for *each cluster*. 


```julia
IM = SparseKmeansFeatureRanking.get_imputed_matrix(collect(transpose(X)), 3; rng=rng)
(classout2, center2, selectedvec2, WSSval2, TSSval2) = SparseKmeansFeatureRanking.sparsekmeans2(IM, sparsity);
```


```julia
randindex(classout2,[repeat([1], 100); repeat([2], 100); repeat([3], 100)])[1]
```

Selected feature does not necessarily match 1:33 for all the clusters, as seen below. 


```julia
selectedvec2
```

## Matrices with missing entries

We can apply the SKFR algorithm on the dataset with missing values, denoted by `NaN`s. Below, we put 10% of the values in the data matrix as `NaN`. 


```julia
using StatsBase
missingix=sample(1:features*cases,Int(features*cases*0.1),replace=false)
X_missing = deepcopy(X)
X_missing[CartesianIndices(X_missing)[missingix]] .= NaN;
```

Then, we run the SKFR functions just as above. For each iteration, missing values are imputed by the center of current cluster centers, as suggested by the k-POD imputation method. 


```julia
IM = SparseKmeansFeatureRanking.get_imputed_matrix(collect(transpose(X_missing)), 3; rng=rng);
```


```julia
(classout1, center1, selectedvec1, WSSval1, TSSval1) = SparseKmeansFeatureRanking.sparsekmeans1(IM, sparsity);
```

If we check the rand index, we see that the clustering result is a little bit noisy, as one may expect.


```julia
randindex(classout1,[repeat([1], 100); repeat([2], 100); repeat([3], 100)])[1]
```

The set of selected features is still the same as before. 


```julia
length(union(selectedvec1, 1:33))
```

## `SnpArray` Usage

The SKFR algorithm can also be applied to the PLINK 1 BED-formatted data through `SnpArrays.jl`. This can be considered a case of unsupervised ancestry informative marker (AIM) selection. 


```julia
using SnpArrays
X = SnpArray(SnpArrays.datadir("EUR_subset.bed"))
nclusters = 3;
```


```julia
IM = SparseKmeansFeatureRanking.get_imputed_matrix(X, nclusters; rng=rng)
(classout, center, selectedvec, WSSval, TSSval) = SparseKmeansFeatureRanking.sparsekmeans1(IM, 1000);
```
