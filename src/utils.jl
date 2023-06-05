"""
    compute_μ_σ!(X::AbstractImputedMatrix)
Compute feature-by-feature means and standard deviations of the matrix `X`.
"""
function compute_μ_σ!(A::AbstractImputedMatrix{T}) where T
    n, p = size(A)
    A.μ .= zero(T)
    A.σ .= one(T)
    @threads for t in 1:nthreads()
        j = t
        @inbounds while j <= p
            if !A.fixed_normalization
                A.μ[j], A.σ[j] = StatsBase.mean_and_std(@view A[:, j])
            else
                m = zero(T)
                m2 = zero(T)
                cnt = 0
                for i in 1:n
                    v = getindex_raw(A, i, j)
                    if isnan(v)
                        kk = A.clusters[i]
                        v = A.centers_stable[j, kk] * A.σ[j] + A.μ[j]
                    end
                    m += v
                    m2 += v ^ 2
                    cnt += 1
                end
                m /= cnt
                m2 /= cnt
                A.μ[j] = m
                A.σ[j] = sqrt((m2 - m ^ 2) * cnt / (cnt - 1))
            end
            j += nthreads()
        end
    end
end

function compute_μ_σ!(A::AbstractImputedSnpMatrix{T}) where T
    n, p = size(A)
    A.μ .= zero(T)
    A.σ .= one(T)
    @threads for t in 1:nthreads()
        j = t
        @inbounds while j <= p
            if !A.fixed_normalization
                A.μ[j], A.σ[j] = StatsBase.mean_and_std(@view A[:, j])
            else
                m = zero(T)
                m2 = zero(T)
                cnt = 0
                for i in 1:n
                    v = SnpArrays.convert(T, getindex(A.data, i, j), A.model)
                    if !isnan(v)
                        m += v
                        m2 += v ^ 2
                        cnt += 1
                    end
                end
                m /= cnt
                m2 /= cnt
                A.μ[j] = m
                A.σ[j] = sqrt((m2 - m ^ 2) * cnt / (cnt - 1))
            end
            j += nthreads()
        end
    end
end

function get_distances_to_center!(X::AbstractImputedMatrix{T}, selectedvec::AbstractVector{Int}=1:size(X, 2)) where T
    # TODO: skip non-selected variables
    n, p = size(X)
    k = size(X.centers, 2)
    fill!(X.distances, zero(T))
    fill!(X.distances_tmp, zero(T))
    @threads for t in 1:nthreads()
        jj = t
        while jj <= length(selectedvec)
            j = selectedvec[jj]
            for kk in 1:k
                for i in 1:n
                    j = selectedvec[jj]
                    @inbounds X.distances_tmp[i, kk, t] += (X[i, j] - X.centers[j, kk])^2
                end
            end
            jj += nthreads()
        end
    end
    @tullio X.distances[i, kk] = X.distances_tmp[i, kk, t]
    @tullio X.distances[i, kk] = sqrt(X.distances[i, kk])
    X.distances
end

"""
    get_distances_to_center!(X, selectedvec=1:size(X, 2))
Compute distance of sample `i` to cluster `kk`. 
"""
function get_distances_to_center!(X::ImputedMatrix{T}, selectedvec::AbstractVector{Int}=1:size(X, 2)) where T
    # TODO: skip non-selected variables
    n, p = size(X)
    k = size(X.centers, 2)
    fill!(X.distances, zero(T))
    @assert X.renormalize "X.renormalize must be true"
    @tullio X.distances[i, kk] = @inbounds begin
        v = X.data[i, selectedvec[j]]
        nanv = isnan(v)
        v = nanv * X.centers_stable[selectedvec[j], X.clusters_stable[i]] + !nanv * v
        v = (v - X.μ[selectedvec[j]]) / ((X.σ[selectedvec[j]] < eps()) + X.σ[selectedvec[j]])
        (v - X.centers[selectedvec[j], kk]) ^ 2
    end 
    @tullio X.distances[i, kk] = sqrt(X.distances[i, kk])
    X.distances
end

function get_distances_to_center!(X::ImputedSnpMatrix{T}, selectedvec::AbstractVector{Int}=1:size(X, 2)) where T
    # TODO: skip non-selected variables
    n, p = size(X)
    k = size(X.centers, 2)
    fill!(X.distances, zero(T))
    @assert X.renormalize "X.renormalize must be true"
    fill!(X.distances_tmp, zero(T))
    @threads for t in 1:nthreads()
        jj = t
        while jj <= length(selectedvec)
            j = selectedvec[jj]
            @turbo for kk in 1:k, i in 1:n
                ip3 = i + 3
                v = ((X.data.data)[ip3 >> 2, j] >> ((ip3 & 0x03) << 1)) & 0x03
                nanv = (v == 0x01)
                v = (v > 0x01) ? T(v - 0x01) : T(v)
                v = nanv * X.centers_stable[j, X.clusters_stable[i]] + !nanv * v
                v = (v - X.μ[j]) / ((X.σ[j] < eps()) + X.σ[j])
                X.distances_tmp[i, kk, t] += (v - X.centers[j, kk]) ^ 2
            end
            jj += nthreads()
        end
    end
    @tullio X.distances[i, kk] = X.distances_tmp[i, kk, t]
    @tullio X.distances[i, kk] = sqrt(X.distances[i, kk])
    X.distances
end

function get_distances_to_center!(X::ImputedStackedSnpMatrix{T}, selectedvec::AbstractVector{Int}=1:size(X, 2)) where T
    # TODO: skip non-selected variables
    n, p = size(X)
    k = size(X.centers, 2)
    fill!(X.distances, zero(T))
    @assert X.renormalize "X.renormalize should be true"
    fill!(X.distances_tmp, zero(T))
    @threads for t in 1:nthreads()
        jj = t
        while jj <= length(selectedvec)
            j = selectedvec[jj]
            jjj = searchsortedfirst(X.data.offsets, j) - 1
            jlocal = j - X.data.offsets[jjj]
            @turbo for kk in 1:k, i in 1:n
                ip3 = i + 3
                v = ((X.data.arrays[jjj].data)[ip3 >> 2, jlocal] >> ((ip3 & 0x03) << 1)) & 0x03
                nanv = (v == 0x01)
                v = (v > 0x01) ? T(v - 0x01) : T(v)
                v = nanv * X.centers_stable[j, X.clusters_stable[i]] + !nanv * v
                v = (v - X.μ[j]) / ((X.σ[j] < eps()) + X.σ[j])
                X.distances_tmp[i, kk, t] += (v - X.centers[j, kk]) ^ 2
            end
            jj += nthreads()
        end
    end
    @tullio X.distances[i, kk] = X.distances_tmp[i, kk, t]
    @tullio X.distances[i, kk] = sqrt(X.distances[i, kk])
    X.distances
end

""" 
    get_clusters!(X, center)
Compute the closest cluster from each sample to `X.clusters`. 
"""
function get_clusters!(X::AbstractImputedMatrix{T}) where T
    n, p = size(X)
    k = size(X.centers, 2)
    switched = false
    @inbounds for i = 1:n
        kk = argmin(@view(X.distances[i, :])) # class of closest center
        if kk != X.clusters[i]
            switched = true
            k_prev = X.clusters[i]
            X.clusters[i] = kk
            X.members[kk] += 1
            X.members[k_prev] -= 1
            @threads for t in 1:nthreads() 
                j = t
                while j <= p
                    @assert X.members[kk] > 0
                    X.centers_tmp[j, kk] = X.centers_tmp[j, kk] + (X[i, j]- X.centers_tmp[j, kk]) / X.members[kk]
                    if X.members[k_prev] == 0
                        X.centers_tmp[j, k_prev] = zero(T)
                    else
                        X.centers_tmp[j, k_prev] = X.centers_tmp[j, k_prev] - (X[i, j] - X.centers_tmp[j, k_prev]) / X.members[k_prev]
                    end      
                    j += nthreads()
                end
            end
            # for j in 1:p
            #     X.centers_tmp[j, kk] = X.centers_tmp[j, kk] + (X[i, j]- X.centers_tmp[j, kk]) / X.members[kk]
            #     X.centers_tmp[j, k_prev] = X.centers_tmp[j, k_prev] - (X[i, j] - X.centers_tmp[j, k_prev]) / X.members[k_prev]
            # end
        end
    end
    return (X.clusters, switched)
end

"""
    get_centers!(X)
Compute each of the cluster center for each feature to `X.centers_tmp`.
"""
function get_centers!(X::AbstractImputedMatrix{T}) where T <: Real
    n, p = size(X)
    k = size(X.centers, 2)
    @assert length(X.clusters) == n
    @assert size(X.centers, 1) == p
    fill!(X.centers_tmp, zero(T))
    fill!(X.members, zero(eltype(X.members)))
    @threads for t in 1:nthreads()
        j = t
        while j <= p
            @inbounds for i in 1:n
                c = X.clusters[i]
                X.centers_tmp[j, c] = X.centers_tmp[j, c] + X[i, j]
            end
            j += nthreads()
        end
    end
    @inbounds for i in 1:n
        c = X.clusters[i]
        X.members[c] = X.members[c] + 1
    end
    @inbounds for kk = 1:k
        if X.members[kk] > 0
            X.centers[:, kk] .= @view(X.centers_tmp[:, kk]) ./ X.members[kk]
        end
    end
    X.centers_tmp .= X.centers
    X.centers
end

function get_centers!(X::ImputedSnpMatrix{T}) where T <: Real
    n, p = size(X)
    k = size(X.centers, 2)
    @assert length(X.clusters) == n
    @assert size(X.centers, 1) == p
    fill!(X.centers_tmp, zero(T))
    fill!(X.members, zero(eltype(X.members)))
    @tturbo for j in 1:size(X.centers_tmp, 1), i in 1:length(X.clusters)
        ip3 = i + 3
        v = ((X.data.data)[ip3 >> 2, j] >> ((ip3 & 0x03) << 1)) & 0x03
        nanv = (v == 0x01)
        v = (v > 0x01) ? T(v - 0x01) : T(v)
        v = nanv * X.centers_stable[j, X.clusters_stable[i]] + !nanv * v
        v = (v - X.μ[j]) / ((X.σ[j] < eps()) + X.σ[j])
        for c in 1:size(X.centers_tmp, 2)
            X.centers_tmp[j, c] += (X.clusters[i] == c) * v
        end
    end
    @inbounds for i in 1:n
        c = X.clusters[i]
        X.members[c] = X.members[c] + 1
    end
    @inbounds for kk = 1:k
        if X.members[kk] > 0
            X.centers[:, kk] .= @view(X.centers_tmp[:, kk]) ./ X.members[kk]
        end
    end
    X.centers_tmp .= X.centers
    X.centers
end

function get_centers!(X::ImputedStackedSnpMatrix{T}) where T <: Real
    n, p = size(X)
    k = size(X.centers, 2)
    @assert length(X.clusters) == n
    @assert size(X.centers, 1) == p
    fill!(X.centers_tmp, zero(T))
    fill!(X.members, zero(eltype(X.members)))
    for jjj in 1:length(X.data.arrays)
        offsets = X.data.offsets
        @tturbo for j in (offsets[jjj] + 1):(offsets[jjj + 1]), i in 1:length(X.clusters)
            ip3 = i + 3
            v = ((X.data.arrays[jjj].data)[ip3 >> 2, j - offsets[jjj]] >> ((ip3 & 0x03) << 1)) & 0x03
            nanv = (v == 0x01)
            v = (v > 0x01) ? T(v - 0x01) : T(v)
            v = nanv * X.centers_stable[j, X.clusters_stable[i]] + !nanv * v
            v = (v - X.μ[j]) / ((X.σ[j] < eps()) + X.σ[j])
            for c in 1:size(X.centers_tmp, 2)
                X.centers_tmp[j, c] += (X.clusters[i] == c) * v
            end
        end
    end
    @inbounds for i in 1:n
        c = X.clusters[i]
        X.members[c] = X.members[c] + 1
    end
    @inbounds for kk = 1:k
        if X.members[kk] > 0
            X.centers[:, kk] .= @view(X.centers_tmp[:, kk]) ./ X.members[kk]
        end
    end
    X.centers_tmp .= X.centers
    X.centers
end

"""
    filter_aims(src, X, v; des)
Filters the plink file with filename `src`.bed with the AIM list `v`. 
"""
function filter_aims(src::AbstractString, X::ImputedSnpMatrix, v::Vector{<:Integer};
    des=src * ".$(length(v))aims")
    aims = sort(v)
    s = X.data
    n, p = size(s)
    sampleidx = trues(n)
    SnpArrays.filter(src, trues(n), aims; des = des)
end

"""
    get_freq!(freq, denom, s, k, clusters)
Get first-allele frequency (in the order of appearance on the bim file) of each cluster. 
"""
function get_freq!(freq::AbstractMatrix{T}, denom::AbstractMatrix{T}, s::SnpArray, k::Integer, 
    clusters::Vector{Int}) where T
    n, p = size(s)
    fill!(freq, zero(T))
    @tturbo for j in 1:p, i in 1:n
        ip3 = i + 3
        v = ((s.data)[ip3 >> 2, j] >> ((ip3 & 0x03) << 1)) & 0x03
        nanv = (v == 0x01)
        v = (v > 0x01) ? T(v - 0x01) : T(v)
        v = !nanv * v
        for c in 1:k
            freq[c, j] += (clusters[i] == c) * v
            denom[c, j] += (clusters[i] == c) * (!nanv * 2one(T))
        end
    end
    freq ./= denom
end
function get_freq!(freq::AbstractMatrix{T}, denom::AbstractMatrix{T}, X::ImputedSnpMatrix{T}) where T
    get_freq!(freq, denom, X.data, size(X.centers, 2), X.clusters)
end
function get_freq(s::SnpArray, k::Integer, clusters::Vector{Int})
    n, p = size(s)
    freq = Matrix{Float64}(undef, k, p)
    denom = Matrix{Float64}(undef, k, p)
    get_freq!(freq, denom, s, k, clusters)
end
function get_freq(X::ImputedSnpMatrix{T}) where T
    n, p = size(X)
    k = size(X.centers, 2)
    freq = Matrix{T}(undef, k, p)
    denom = Matrix{T}(undef, k, p)
    get_freq!(freq, denom, X)
end

"""
    dists_from_single_row!(dists, X, s)
Distances of each sample from row s, filled in `dists`.
"""
function dists_from_single_row!(dists::Matrix{T}, X::AbstractMatrix{T}, s::Int) where T
    n, p = size(X)
    # @assert length(dists) == n
    fill!(dists, zero(T))
    @threads for t in 1:nthreads()
        j = t
        @inbounds while j <= p
            for i in 1:n
                dists[i, t] += (X[i, j] - X[s, j]) ^ 2 
            end           
            j += nthreads()
        end
    end
    @inbounds for i in 1:n
        for t in 2:nthreads()
            dists[i, 1] += dists[i, t]
        end
    end
    # @tturbo for j in 1:p
    #     for i in 1:n
    #         dists[i] += (X[i, j] - X[s, j]) ^ 2
    #     end
    # end
    dists[:, 1] .= sqrt.(@view(dists[:, 1]))
    dists[s, 1] = zero(T)
    dists
end
function dists_from_single_row!(dists::Matrix{T}, X::ImputedSnpMatrix{T}, s::Int) where T
    n, p = size(X)
    fill!(dists, zero(T))
    @tturbo for j in 1:p
        sp3 = s + 3
        vr = ((X.data.data)[sp3 >> 2, j] >> ((sp3 & 0x03) << 1)) & 0x03
        nanvr = (vr == 0x01)
        vr = (vr > 0x01) ? T(vr - 0x01) : T(vr)
        vr = nanvr * X.centers_stable[j, X.clusters_stable[s]] + !nanvr * vr
        vr = (vr - X.μ[j]) / ((X.σ[j] < eps()) + X.σ[j])
        for i in 1:n
            ip3 = i + 3
            v = ((X.data.data)[ip3 >> 2, j] >> ((ip3 & 0x03) << 1)) & 0x03
            nanv = (v == 0x01)
            v = (v > 0x01) ? T(v - 0x01) : T(v)
            v = nanv * X.centers_stable[j, X.clusters_stable[i]] + !nanv * v
            v = (v - X.μ[j]) / ((X.σ[j] < eps()) + X.σ[j])
            dists[i, 1] += (v - vr) ^ 2
        end
    end
    dists[:, 1] .= sqrt.(@view(dists[:, 1]))
    dists[s, 1] = zero(T)
end

"""
    dist_from_rows!(dists, X, iseeds)
distances from each row, for each `iseed` in `initclass!`.
"""
function dists_from_rows!(dists::Array{T, 3}, X::AbstractMatrix{T}, iseeds::Vector{Int}) where T
    n, p = size(X)
    k = size(dists, 2)
    @assert size(dists, 1) == n
    fill!(dists, zero(T))
    X_subsample = X[iseeds, :]
    @threads for t in 1:nthreads()
        j = t
        while j <= p
            @inbounds for s in 1:length(iseeds)
                for i in 1:n
                    dists[i, s, t] += (X[i, j] - X_subsample[s, j]) ^ 2
                end
            end
            j += nthreads()
        end
    end
    @inbounds for i in 1:n
        for s in 1:length(iseeds)
            for t in 2:nthreads()
                dists[i, s, 1] += dists[i, s, t]
            end
        end
    end
    dists[:, :, 1] .= sqrt.(@view(dists[:, :, 1]))
    # @tturbo dists .= sqrt.(dists)
    dists
end

function dists_from_rows!(dists::Array{T, 3}, X::ImputedSnpMatrix{T}, iseeds::Vector{Int}) where T
    n, p = size(X)
    k = size(dists, 2)
    @assert size(dists, 1) == n
    fill!(dists, zero(T))
    X_subsample = X[iseeds, :]
    @tturbo for j in 1:p
        for i in 1:n
            ip3 = i + 3
            v = ((X.data.data)[ip3 >> 2, j] >> ((ip3 & 0x03) << 1)) & 0x03
            nanv = (v == 0x01)
            v = (v > 0x01) ? T(v - 0x01) : T(v)
            v = nanv * X.centers_stable[j, X.clusters_stable[i]] + !nanv * v
            v = (v - X.μ[j]) / ((X.σ[j] < eps()) + X.σ[j])
            for s in 1:length(iseeds)
                dists[i, s, 1] += (v - X_subsample[s, j]) ^ 2
            end
        end
    end
    dists[:, :, 1] .= sqrt.(@view(dists[:, :, 1]))
    # @tturbo dists .= sqrt.(dists)
    dists
end

"""
    initclass!(class, X, k)

kmeans plusplus initialization for classes, modified from Clustering.jl. 
""" 
function initclass!(class::Vector{Int}, X::AbstractMatrix{T}, k::Int; rng=Random.GLOBAL_RNG) where T 
    n = size(X, 1)
    iseeds = zeros(Int, k)
    s = rand(rng, 1:n)
    iseeds[1] = s

    if k > 1
        mincosts = Matrix{T}(undef, n, nthreads())
        dists_from_single_row!(mincosts, X, s)

        # pick remaining seeds with a chance proportional to mincosts.
        tmpcosts = similar(mincosts)
        for j = 2:k
            s = wsample(rng, 1:n, @view(mincosts[:, 1]))
            iseeds[j] = s
            dists_from_single_row!(tmpcosts, X, s)
            updatemin!(@view(mincosts[:, 1]), @view(tmpcosts[:, 1]))
            mincosts[s, 1] = 0
        end
    end
    dists = Array{T, 3}(undef, n, k, nthreads())
    dists_from_rows!(dists, X, iseeds)
    @threads for t in 1:nthreads()
        i = t
        @inbounds while i <= n
            class[i] = argmin(@view(dists[i, :, 1]))
            i += nthreads()
        end
    end
    return class
end
  
#helper function
function updatemin!(r::AbstractArray, x::AbstractArray)
    n = length(r)
    length(x) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    @inbounds for i = 1:n
        xi = x[i]
        if xi < r[i]
            r[i] = xi
        end
    end
    return r
end
  