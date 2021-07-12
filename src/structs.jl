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
    centers::Matrix{T}
    centers_tmp::Matrix{T}
    members::Vector{Int}
    criterion::Vector{T}
    distances::Matrix{T}
    μ::Vector{T}
    σ::Vector{T}
    renormalize::Bool
end

mutable struct ImputedSnpMatrix{T} <: AbstractImputedMatrix{T}
    data::SnpArray
    model::Union{Val{1}, Val{2}, Val{3}}
    clusters::Vector{Int}
    centers::Matrix{T}
    centers_tmp::Matrix{T}
    members::Vector{Int}
    criterion::Vector{T}
    distances::Matrix{T}
    μ::Vector{T}
    σ::Vector{T}
    renormalize::Bool
end

@inline function Base.size(x::AbstractImputedMatrix)
    return size(x.data)
end

@inline function classes(x::AbstractImputedMatrix)
    return size(x.centers, 2)
end

function ImputedMatrix{T}(data::AbstractMatrix{T}, k::Int; renormalize=true, initclass=true) where {T <: Real}
    n, p = size(data)
    clusters = Vector{Int}(undef, n)
    centers = zeros(T, p, k)
    centers_tmp = zeros(T, p, k)
    # set up column imputation
    fill!(clusters, 1)
    @inbounds for j in 1:p
        s = zero(T)
        cnt = 0
        for i in 1:n
            if isnan(data[i, j])
                continue
            end
            s += data[i, j]
            cnt += 1
        end
        avg = s / cnt
        centers[j, :] .= avg
    end
    # Initialization step
    members = zeros(Int, k)
    criterion = zeros(T, p)
    distances = zeros(T, n, k)

    μ = zeros(T, p)
    σ = zeros(T, p)

    r = ImputedMatrix{T}(data, clusters, centers, centers_tmp, members, criterion, distances, μ, σ, false)
    if initclass
        r.clusters = initclass!(r.clusters, r, k)
    end
    get_centers!(r)
    compute_μ_σ!(r)
    r.renormalize = renormalize
    return r
end

function ImputedSnpMatrix{T}(data::SnpArray, k::Int; renormalize=true, initclass=true, 
        model=ADDITIVE_MODEL) where {T <: Real}
    n, p = size(data)
    clusters = Vector{Int}(undef, n)
    centers = zeros(T, p, k)
    centers_tmp = zeros(T, p, k)
    # set up column imputation
    fill!(clusters, 1)
    @inbounds for j in 1:p
        s = zero(T)
        cnt = 0
        for i in 1:n
            v = SnpArrays.convert(T, getindex(s, i, j), model)
            if isnan(v)
                continue
            end
            s += v
            cnt += 1
        end
        avg = s / cnt
        centers[j, :] .= avg
    end
    # Initialization step
    members = zeros(Int, k)
    criterion = zeros(T, p)
    distances = zeros(T, n, k)

    μ = zeros(T, p)
    σ = zeros(T, p)

    r = ImputedSnpMatrix{T}(data, model, clusters, centers, centers_tmp, members, criterion, distances, μ, σ, false)
    if initclass
        r.clusters = initclass!(r.clusters, r, k)
    end
    get_centers!(r)
    compute_μ_σ!(r)
    r.renormalize = renormalize
    return r
end

@inline function Base.getindex(A::AbstractImputedMatrix{T}, i::Int, j::Int)::T where {T}
    r = getindex_raw(A, i, j)
    if A.renormalize
        r -= A.μ[j]
        r /= A.σ[j]
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
    r = Base.getindex(A.data, i, j)
    if isnan(r)
        r = A.centers[j, A.clusters[i]]
    end
    return r
end    

@inline function getindex_raw(A::ImputedSnpMatrix{T}, i::Int, j::Int)::T where T
    r = SnpArrays.convert(T, getindex(A.data, i, j), A.model)
    if isnan(r)
        r = A.centers[j, A.clusters[i]]
    end
    return r
end  
