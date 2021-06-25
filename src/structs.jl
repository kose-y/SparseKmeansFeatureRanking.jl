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
struct ImputedMatrix{T} <: AbstractMatrix{T}
    data::AbstractMatrix{T}
    clusters::Vector{Int}
    centers::Matrix{T}
    members::Vector{Int}
    criterion::Vector{T}
end

@inline function Base.size(x::ImputedMatrix)
    return size(x.data)
end

@inline function classes(x::ImputedMatrix)
    return size(x.centers, 2)
end

function ImputedMatrix{T}(data::AbstractMatrix{T}, k::Int) where {T <: Real}
    n, p = size(data)
    clusters = Vector{Int}(undef, n)
    centers = Matrix{T}(undef, p, k)
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
        fill!(centers[:, p], avg)
    end

    # Initialization step
    clusters = initclass!(clusters, data, k)
    members = zeros(Int, k)
    centers = get_centers!(centers, members, data, clusters)
    criterion = zeros(T, p)
    ImputedMatrix{T}(data, clusters, centers, members, criterion)
end

@inline function Base.getindex(A::ImputedMatrix{T}, i::Int, j::Int) where {T}
    r = Base.getindex(A.data, i, j)
    if isnan(r)
        r = A.centers[cluster[i], j]
    end
    return r
end

@inline function Base.setindex!(A::ImputedMatrix{T}, v::T, i::Int, j::Int) where T
    A.data[i, j] = v
end