abstract type AbstractImputedMatrix{T} <: AbstractMatrix{T} end

"""
    ImputedMatrix{T}(data, k)

Structure for SKFR. 

# members

- `data`: Data matrix (n x p) with `NaN`s for missing values
- `clusters`: cluster labels (length-n Int)
- `centers`: cluster centers (p x k)
- `members`: Number of members of each cluster (length-k Int)
- `criterion`: cluster criterion (length-p T)
"""
mutable struct ImputedMatrix{T} <: AbstractImputedMatrix{T}
    data::Matrix{T}
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

function ImputedMatrix{T}(data::AbstractMatrix{T}, k::Int; renormalize=true, 
    initclass=true, 
    rng=Random.GLOBAL_RNG,
    fixed_normalization=true) where {T <: Real}
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

    μ = zeros(T, p)
    σ = zeros(T, p)
    switched = falses(n)

    r = ImputedMatrix{T}(data, clusters, clusters_tmp, clusters_stable, centers, centers_stable, avg,
        bestclusters, bestcenters, centers_tmp, members, criterion, distances, distances_tmp, μ, σ, switched, renormalize, fixed_normalization)
    if initclass
        r.clusters = initclass!(r.clusters, r, k; rng=rng)
    end
    compute_μ_σ!(r)
    get_centers!(r)

    r.renormalize = renormalize
    return r
end

function ImputedSnpMatrix{T}(data::AbstractSnpArray, k::Int; renormalize=true, initclass=true, 
        fixed_normalization=true,
        rng=Random.GLOBAL_RNG,
        model=ADDITIVE_MODEL) where {T <: Real}
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

    s_ = zeros(T, nthreads())
    cnt_ = zeros(Int, nthreads())
    @threads for t in 1:nthreads()
        j = t 
        @inbounds while j <= p
            for i in 1:n
                v = SnpArrays.convert(T, getindex(data, i, j), model)
                if isnan(v)
                    continue
                end
                s_[t] += v
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

    μ = zeros(T, p)
    σ = ones(T, p)
    switched = falses(n)
    
    MatrixType = typeof(data) <: SnpArray ? ImputedSnpMatrix : ImputedStackedSnpMatrix
    r = MatrixType{T}(data, model, clusters, clusters_tmp, clusters_stable, centers, centers_stable, avg,
        bestclusters, bestcenters, centers_tmp, members, criterion, distances, distances_tmp, μ, σ, switched, renormalize, fixed_normalization)
    if initclass
        initclass!(r.clusters, r, k; rng=rng)
    end
    compute_μ_σ!(r)
    get_centers!(r)


    r.renormalize = renormalize
    return r
end

function reinitialize!(X::AbstractImputedMatrix{T}; rng=Random.GLOBAL_RNG) where T
    n, p = size(X)
    X.centers_stable .= X.avg
    # clusters_stable does not matter, as long as this is total mean imputation. 
    k = classes(X)
    initclass!(X.clusters, X, k; rng=rng)
    get_centers!(X)
    return X
end

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

@inline function Base.setindex!(A::AbstractImputedMatrix{T}, v::T, i::Int, j::Int) where T
    A.data[i, j] = v
end

@inline function Base.setindex!(A::AbstractImputedSnpMatrix{T}, v::T, i::Int, j::Int) where T
    @error "setindex!() on ImputedSnpMatrix is not allowed."
end

@inline function getindex_raw(A::AbstractImputedMatrix{T}, i::Int, j::Int)::T where T
    Base.getindex(A.data, i, j)
end    

@inline function getindex_raw(A::AbstractImputedSnpMatrix{T}, 
    i::Int, j::Int)::T where T
    SnpArrays.convert(T, getindex(A.data, i, j), A.model)
end  
