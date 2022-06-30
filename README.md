# SKFR.jl
[![](https://img.shields.io/badge/docs-dev-blue.svg)

Sparse K-means via Feature Ranking

This software package contains an efficient multi-threaded implementation of sparse K-means via feature ranking proposed by [Zhang, Lange, and Xu (2020)](https://proceedings.neurips.cc//paper/2020/file/735ddec196a9ca5745c05bec0eaa4bf9-Paper.pdf). The code is based on the [original github repository](https://github.com/ZhiyueZ/SKFR). The authors of the original code have kindly agreed to redistribute the derivative of their code on this repository under the MIT License. 

## Installation

This package requires Julia v1.6 or later, which can be obtained from
<https://julialang.org/downloads/> or by building Julia from the sources in the
<https://github.com/JuliaLang/julia> repository.

The package can be installed by running the following code:
```julia
using Pkg
pkg"add https://github.com/kose-y/SKFR.jl"
```
