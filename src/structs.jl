abstract type AbstractImputedMatrix{T} <: AbstractMatrix{T} end

"""
    ImputedMatrix{T}(data::AbstractMatrix{T}, k::Int; 
    renormalize=true, 
    initclass=true, 
    rng=Random.GLOBAL_RNG,
    fixed_normalization=true)

Structure for SKFR. 

# Input
- `data`: raw data matrix 
- `k`: number of clusters
- `renormalize`: whether to compute normalization lazily
- `initclass`: whether to initialize the clusters on its creation
- `rng`: random number generator.
- `fixed_normalization`: If `true`, means and standard deviations 
    for each feature is pre-computed based on the values disregarding missing values. 
    Normalization is not to be updated.
"""
mutable struct ImputedMatrix{T} <: AbstractImputedMatrix{T}
    data::Matrix{T} # Data matrix (n x p) with `NaN`s for missing values
    clusters::Vector{Int} # cluster labels (length-n Int)
    clusters_tmp::Vector{Int} # temp storage for clusters
    clusters_stable::Vector{Int} # "stable" cluster assignment for imputation
    centers::Matrix{T} # cluster centers
    centers_stable::Matrix{T} # "stable" cluster centers for imputation
    avg::T  # average of all the entries of the input matrix, disregarding missing values.
    bestclusters::Vector{Int} # best cluster assignment for `sparsekmeans_repeat`.
    bestcenters::Matrix{T} # best cluster centers for `sparsekmeans_repeat`
    centers_tmp::Matrix{T} # all the cluster centers, no variables zeroed out. 
    members::Vector{Int} # number of members for each cluster
    criterion::Vector{T} # information criterion for each variable
    distances::Matrix{T} # distance to cluster centers
    distances_tmp::Array{T, 3} # temporary storage for distance computation
    μ::Vector{T} # variable means for normalization
    σ::Vector{T} # variable stds for normalization
    switched::BitVector # boolean vector for each feature
    renormalize::Bool # whether to compute normalization on-the-fly
    fixed_normalization::Bool # if true, do not recompute `μ` and `σ`.
end

mutable struct ImputedSnpMatrix{T} <: AbstractImputedMatrix{T}
    data::SnpArray
    model::Union{Val{1}, Val{2}, Val{3}}
    clusters::Vector{Int}
    clusters_tmp::Vector{Int}
    clusters_stable::Vector{Int}
    centers::Matrix{T}
    centers_stable::Matrix{T} 
    avg::T
    bestclusters::Vector{Int}
    bestcenters::Matrix{T}
    centers_tmp::Matrix{T}
    members::Vector{Int}
    criterion::Vector{T}
    distances::Matrix{T}
    distances_tmp::Array{T, 3}
    μ::Vector{T}
    σ::Vector{T}
    switched::BitVector
    renormalize::Bool
    fixed_normalization::Bool
end

mutable struct ImputedStackedSnpMatrix{T} <: AbstractImputedMatrix{T}
    data::StackedSnpArray
    model::Union{Val{1}, Val{2}, Val{3}}
    clusters::Vector{Int}
    clusters_tmp::Vector{Int}
    clusters_stable::Vector{Int}
    centers::Matrix{T}
    centers_stable::Matrix{T} 
    avg::T
    bestclusters::Vector{Int}
    bestcenters::Matrix{T}
    centers_tmp::Matrix{T}
    members::Vector{Int}
    criterion::Vector{T}
    distances::Matrix{T}
    distances_tmp::Array{T, 3}
    μ::Vector{T}
    σ::Vector{T}
    switched::BitVector
    renormalize::Bool
    fixed_normalization::Bool
end

const AbstractImputedSnpMatrix{T} = Union{ImputedSnpMatrix{T}, ImputedStackedSnpMatrix{T}}

@inline function Base.size(x::AbstractImputedMatrix)
    return size(x.data)
end

@inline function classes(x::AbstractImputedMatrix)
    return size(x.centers, 2)
end

function get_imputed_matrix(data, k::Int; renormalize=true,
    initclass=true, 
    rng=Random.GLOBAL_RNG,
    fixed_normalization=true, T=Float64)
    if typeof(data) <: AbstractSnpArray
        ImputedSnpMatrix{T}(data, k; renormalize=renormalize, initclass=initclass, 
            rng=rng, fixed_normalization=fixed_normalization)
    else
        ImputedMatrix{T}(data, k; renormalize=renormalize, initclass=initclass, 
            rng=rng, fixed_normalization=fixed_normalization)
    end
end

function ImputedMatrix{T}(data::AbstractMatrix{T}, k::Int;  
    renormalize=true, 
    initclass=true, 
    rng=Random.GLOBAL_RNG,
    fixed_normalization=true, 
    μ=nothing, σ=nothing) where {T <: Real}
    n, p = size(data)
    clusters = Vector{Int}(undef, n)
    clusters_tmp = Vector{Int}(undef, n)
    centers = zeros(T, p, k)
    centers_stable = zeros(T, p, k)
    centers_tmp = zeros(T, p, k)
    bestclusters = Vector{Int}(undef, n)
    bestcenters = zeros(T, p, k)

    # set up mean imputation
    fill!(clusters, 1)
    clusters_stable = copy(clusters)
    s_ = zeros(T, nthreads())
    cnt_ = zeros(Int, nthreads())
    @threads for t in 1:nthreads()
        j = t 
        @inbounds while j <= p
            for i in 1:n
                if isnan(data[i, j])
                    continue
                end
                s_[t] += data[i, j]
                cnt_[t] += 1
            end
            j += nthreads()
        end
    end
    s = sum(s_)
    cnt = sum(cnt_)
    
    avg = s / cnt

    centers .= avg
    centers_stable .= centers
    # Initialization step
    members = zeros(Int, k)
    criterion = zeros(T, p)
    distances = zeros(T, n, k)
    distances_tmp = zeros(T, n, k, nthreads())

    _μ = zeros(T, p)
    _σ = zeros(T, p)
    switched = falses(n)

    r = ImputedMatrix{T}(data, clusters, clusters_tmp, clusters_stable, centers, centers_stable, avg,
        bestclusters, bestcenters, centers_tmp, members, criterion, distances, distances_tmp, _μ, _σ, switched, renormalize, fixed_normalization)
    if initclass # initialize clusters
        r.clusters = initclass!(r.clusters, r, k; rng=rng)
    end
    # populate μ and σ for on-the-fly normalization
    if μ === nothing
        compute_μ_σ!(r)
    else
        r.μ .= μ
        r.σ .= σ
    end
    get_centers!(r)

    r.renormalize = renormalize
    return r
end

function ImputedSnpMatrix{T}(data::AbstractSnpArray, k::Int; 
        renormalize=true, initclass=true, 
        fixed_normalization=true,
        rng=Random.GLOBAL_RNG,
        model=ADDITIVE_MODEL,
        μ=nothing, σ=nothing) where {T <: Real}
    n, p = size(data)
    clusters = Vector{Int}(undef, n)
    clusters_tmp = Vector{Int}(undef, n)
    centers = zeros(T, p, k)
    centers_stable = zeros(T, p, k)
    bestclusters = Vector{Int}(undef, n)
    bestcenters = zeros(T, p, k)
    centers_tmp = zeros(T, p, k)

    # set up mean imputation
    fill!(clusters, 1)
    clusters_stable = copy(clusters)

    s = zero(T)
    cnt = zero(Int)
    # s_ = zeros(T, nthreads())
    # cnt_ = zeros(Int, nthreads())

    @tturbo for j in 1:p, i in 1:n
        ip3 = i + 3
        v = ((data.data)[ip3 >> 2, j] >> ((ip3 & 0x03) << 1)) & 0x03
        nanv = (v == 0x01)
        v = (v > 0x01) ? T(v - 0x01) : T(v)
        s += !nanv * v
        cnt += !nanv * 1
    end

    avg = s / cnt
    centers .= avg
    centers_stable .= centers

    # Initialize structure
    members = zeros(Int, k)
    criterion = zeros(T, p)
    distances = zeros(T, n, k)
    distances_tmp = zeros(T, n, k, nthreads())

    _μ = zeros(T, p)
    _σ = ones(T, p)
    switched = falses(n)
    
    MatrixType = typeof(data) <: SnpArray ? ImputedSnpMatrix : ImputedStackedSnpMatrix
    r = MatrixType{T}(data, model, clusters, clusters_tmp, clusters_stable, centers, centers_stable, avg,
        bestclusters, bestcenters, centers_tmp, members, criterion, distances, distances_tmp, _μ, _σ, switched, renormalize, fixed_normalization)
    if initclass # initialize clusters
        initclass!(r.clusters, r, k; rng=rng)
    end
    if μ === nothing
        compute_μ_σ!(r)
    else
        r.μ .= μ
        r.σ .= σ
    end
    get_centers!(r)


    r.renormalize = renormalize
    return r
end

function ImputedSnpMatrix{T}(path::AbstractString, k::Int; renormalize=true, initclass=true, 
    fixed_normalization=true,
    rng=Random.GLOBAL_RNG,
    model=ADDITIVE_MODEL) where {T <: Real}
    ImputedSnpMatrix{T}(SnpArray(path), k; renormalize=renormalize, initclass=initclass,
        fixed_normalization=fixed_normalization,
        rng=Random.GLOBAL_RNG,
        model=ADDITIVE_MODEL)
end

"""
    reinitialize!(X; rng=Random.GLOBAL_RNG)
Reinitialize cluster assignments.
"""
function reinitialize!(X::AbstractImputedMatrix{T}; rng=Random.GLOBAL_RNG) where T
    n, p = size(X)
    X.centers_stable .= X.avg
    # clusters_stable does not matter, as long as this is total mean imputation. 
    k = classes(X)
    initclass!(X.clusters, X, k; rng=rng)
    get_centers!(X)
    return X
end

"""
    Base.getindex(X, i, j)
Compute the value of `X[i, j]`. 
"""
@inline function Base.getindex(A::AbstractImputedMatrix{T}, i::Int, j::Int)::T where {T}
    r = getindex_raw(A, i, j)
    if isnan(r)
        r = A.centers_stable[j, A.clusters_stable[i]]
    end
    if A.renormalize
        r -= A.μ[j]
        if A.σ[j] > eps()
            r /= A.σ[j]
        end
    end
    r
end

"""
    Base.setindex!(X, v, i, j)
Directly set the value in `X.data[i, j]`.
"""
@inline function Base.setindex!(A::AbstractImputedMatrix{T}, v::T, i::Int, j::Int) where T
    A.data[i, j] = v
end

@inline function Base.setindex!(A::AbstractImputedSnpMatrix{T}, v::T, i::Int, j::Int) where T
    @error "setindex!() on ImputedSnpMatrix is not allowed."
end

"""
    getindex_raw(X, i, j)
Get the raw data in the index `(i, j)`.
"""
@inline function getindex_raw(A::AbstractImputedMatrix{T}, i::Int, j::Int)::T where T
    Base.getindex(A.data, i, j)
end    

@inline function getindex_raw(A::AbstractImputedSnpMatrix{T}, 
    i::Int, j::Int)::T where T
    SnpArrays.convert(T, getindex(A.data, i, j), A.model)
end  
