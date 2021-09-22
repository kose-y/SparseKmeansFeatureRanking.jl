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
    clusters_stable::Vector{Int}
    centers::Matrix{T}
    centers_stable::Matrix{T}    
    bestclusters::Vector{Int}
    bestcenters::Matrix{T}
    centers_tmp::Matrix{T}
    members::Vector{Int}
    criterion::Vector{T}
    distances::Matrix{T}
    μ::Vector{T}
    σ::Vector{T}
    renormalize::Bool
    fixed_normalization::Bool
end

mutable struct ImputedSnpMatrix{T} <: AbstractImputedMatrix{T}
    data::SnpArray
    model::Union{Val{1}, Val{2}, Val{3}}
    clusters::Vector{Int}
    clusters_stable::Vector{Int}
    centers::Matrix{T}
    centers_stable::Matrix{T} 
    bestclusters::Vector{Int}
    bestcenters::Matrix{T}
    centers_tmp::Matrix{T}
    members::Vector{Int}
    criterion::Vector{T}
    distances::Matrix{T}
    μ::Vector{T}
    σ::Vector{T}
    renormalize::Bool
    fixed_normalization::Bool
end

@inline function Base.size(x::AbstractImputedMatrix)
    return size(x.data)
end

@inline function classes(x::AbstractImputedMatrix)
    return size(x.centers, 2)
end

function ImputedMatrix{T}(data::AbstractMatrix{T}, k::Int; renormalize=true, initclass=true, fixed_normalization=true) where {T <: Real}
    n, p = size(data)
    clusters = Vector{Int}(undef, n)
    centers = zeros(T, p, k)
    centers_stable = zeros(T, p, k)
    centers_tmp = zeros(T, p, k)
    bestclusters = Vector{Int}(undef, n)
    bestcenters = zeros(T, p, k)
    # set up mean imputation
    fill!(clusters, 1)
    clusters_stable = copy(clusters)
    s = zero(T)
    cnt = 0
    @inbounds for j in 1:p
        for i in 1:n
            if isnan(data[i, j])
                continue
            end
            s += data[i, j]
            cnt += 1
        end
    end
    avg = s / cnt
    centers .= avg
    centers_stable .= centers
    # Initialization step
    members = zeros(Int, k)
    criterion = zeros(T, p)
    distances = zeros(T, n, k)

    μ = zeros(T, p)
    σ = zeros(T, p)

    r = ImputedMatrix{T}(data, clusters, clusters_stable, centers, centers_stable, 
        bestclusters, bestcenters, centers_tmp, members, criterion, distances, μ, σ, renormalize, fixed_normalization)
    if initclass
        r.clusters = initclass!(r.clusters, r, k)
    end
    compute_μ_σ!(r)
    get_centers!(r)

    r.renormalize = renormalize
    return r
end

function ImputedSnpMatrix{T}(data::SnpArray, k::Int; renormalize=true, initclass=true, 
        fixed_normalization=true,
        model=ADDITIVE_MODEL) where {T <: Real}
    n, p = size(data)
    clusters = Vector{Int}(undef, n)
    centers = zeros(T, p, k)
    centers_stable = zeros(T, p, k)
    bestclusters = Vector{Int}(undef, n)
    bestcenters = zeros(T, p, k)
    centers_tmp = zeros(T, p, k)
    # set up mean imputation
    fill!(clusters, 1)
    clusters_stable = copy(clusters)
    s = zero(T)
    cnt = 0
    @inbounds for j in 1:p
        for i in 1:n
            v = SnpArrays.convert(T, getindex(data, i, j), model)
            if isnan(v)
                continue
            end
            s += v
            cnt += 1
        end
        # avg = s / cnt
        # centers[j, :] .= avg
    end
    avg = s / cnt
    centers .= avg
    centers_stable .= centers
    # Initialization step
    members = zeros(Int, k)
    criterion = zeros(T, p)
    distances = zeros(T, n, k)

    μ = zeros(T, p)
    σ = ones(T, p)

    r = ImputedSnpMatrix{T}(data, model, clusters, clusters_stable, centers, centers_stable, 
        bestclusters, bestcenters, centers_tmp, members, criterion, distances, μ, σ, renormalize, fixed_normalization)
    if initclass
        initclass!(r.clusters, r, k)
    end
    compute_μ_σ!(r)
    get_centers!(r)


    r.renormalize = renormalize
    return r
end

function reinitialize!(X::AbstractImputedMatrix)
    @inbounds for j in 1:p
        for i in 1:n
            v = SnpArrays.convert(T, getindex(data, i, j), model)
            if isnan(v)
                continue
            end
            s += v
            cnt += 1
        end
        # avg = s / cnt
        # centers[j, :] .= avg
    end
    avg = s / cnt
    X.centers_stable .= avg
    k = classes(X)
    initclass!(X.clusters, X, k)
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

@inline function Base.setindex!(A::ImputedSnpMatrix{T}, v::T, i::Int, j::Int) where T
    @error "setindex!() on ImputedSnpMatrix is not allowed."
end

@inline function getindex_raw(A::AbstractImputedMatrix{T}, i::Int, j::Int)::T where T
    Base.getindex(A.data, i, j)
end    

@inline function getindex_raw(A::ImputedSnpMatrix{T}, i::Int, j::Int)::T where T
    SnpArrays.convert(T, getindex(A.data, i, j), A.model)
end  
