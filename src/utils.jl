"""
    powermean(m, s)
power mean for any s.
to prevent numerical underflow, return minimum when mean is rounded to zero
"""
function powmean(m::AbstractArray{T}, s::T) where T <: Union{Float32, Float64}
    res = mean(m .^ s) ^ (1 / s)
    if res > 0
        return( res )
    else
        return( minimum(m))
    end
end
  
"""
    power_kmeans(X, s, k, center)
Implements generalized k-means clustering. The variable `class` should enter
with an initial guess of the classifications.
"""
function power_kmeans(X::AbstractMatrix{T}, s::T,
    k::Integer, center::AbstractArray{T}) where T <: Union{Float32, Float64}
  
    (m_features, N_points) = size(X)
    #(center, weights) = (zeros(T, m_features,k), zeros(T, k, N_points))
    weights =  zeros(T, k, N_points)
    #center = mean(X,2).+5*randn(m_features,k)  #initialization
    obj_old = 1e300; obj = 1e200
    iter = 0
    while (obj_old - obj)/(obj_old*sqrt(m_features)) > 1e-7 || s > -sqrt(m_features)
        iter +=1
        dist = pairwise(Euclidean(), center, X) #rows: centers; columns: distance from center to point i 1 thru N
        #println(dist[:,1])
        #dist[isnan.(dist)] = eps()
  
        coef = sum( dist.^(2*s), 1).^(1/s-1)
        if minimum(coef) < 1e-280 #check for underflow
            println("coef vector small")
            break
        end
        #println(coef[:,1])
        #update weights
        weights = dist.^(2*(s-1)).*coef
        #println(minimum(weights))
        if minimum(weights) < 1e-280 #check for underflow
            println("weight vector small")
            break
        end
        #weights[weights.< eps()] = 3*eps() #make sure non zero
        #weights[isnan.(weights)] = 3*eps()
  
        #update centers
        center = (X*weights')./ sum(weights,2)'
  
        obj_old = obj
        obj_temp = 0
        for j in 1:N_points
            obj_temp += powmean(dist[:,j].^2,s)
        end
        obj = obj_temp
        #println(obj)
        #anneal the s value
        if iter % 2 == 0
            if s > -1.0
                s += -.2
            elseif s > -120.0
                s *= 1.06
                #println( (obj_old - obj)/obj_old )
            end
        end
    end
  
    #print(center)
    #assign labels:
    class = rand(1:k, N_points)
    dist = pairwise(Euclidean(), center, X) # fetch distances
    for point = 1:N_points
        class[point] = argmin(dist[:, point]) # closest center
    end
    println("powermeans final s: $s number of iters: $iter")
    #println(iter)
    return( class, center)
  end
  
"""
    kmeans(X, class, k)
Implements regular k-means
"""
function kmeans(X::Matrix{T}, class::Vector{Int},
    k::Integer) where T <: Real
    (features, points) = size(X)
    (center, members) = (zeros(T, features, k), zeros(Int, k))
    switched = true
    iters = 0
    while switched # iterate until membership stabilizes
        iters += 1
        fill!(center, zero(T)) # update centers
        fill!(members, 0)
        for point = 1:points
            i = class[point]
            center[:, i] = center[:, i] + X[:, point]
            members[i] = members[i] + 1
        end
        for j = 1:k
            center[:, j] = center[:, j] / max(members[j], 1)
        end
        switched = false # update classes
        dist = pairwise(Euclidean(), center, X) # fetch distances
        for point = 1:points
            j = argmin(dist[:, point]) # closest center
            if class[point] != j
                class[point] = j
                switched = true
            end
        end
    end
    println("Lloyd iters: $iters")
    return (class, center)
end
  
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

function get_distances_to_center!(X::ImputedMatrix{T}, selectedvec::AbstractVector{Int}=1:size(X, 2)) where T
    # TODO: skip non-selected variables
    n, p = size(X)
    k = size(X.centers, 2)
    fill!(X.distances, zero(T))
    @assert X.renormalize "X.renormalize should be true"
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
    @assert X.renormalize "X.renormalize should be true"
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

- centers: p x k
- X: n x p
- class: length-n
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
            for i in 1:n
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
    @tturbo for j in 1:p, i in n
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
    freq = Matrix{T}(undef, k, p)
    denom = Matrix{T}(undef, k, p)
    get_freq!(freq, denom, s, k, clusters)
end
function get_freq(X::ImputedSnpMatrix{T})
    freq = Matrix{T}(undef, k, p)
    denom = Matrix{T}(undef, k, p)
    get_freq!(freq, denom, X)
end

"""
Distances from row s. 
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

- dists: n x k. 
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
  
"""
    initcenters(X, k)

taken from Clustering.jl; returns the centers
""" 
function initcenters(X::Matrix{T}, k::Int) where T <: Real
    n = size(X, 2)
    iseeds = zeros(Int, k)
    # randomly pick the first center
    p = rand(1:n)
    iseeds[1] = p
  
    if k > 1
        mincosts = Distances.colwise(Euclidean(), X, view(X,:,p))
        mincosts[p] = 0
  
        # pick remaining (with a chance proportional to mincosts)
        tmpcosts = zeros(n)
        for j = 2:k
            p = wsample(1:n, mincosts)
            iseeds[j] = p
            # update mincosts
            c = view(X,:,p)
            Distances.colwise!(tmpcosts, Euclidean(), X, view(X,:,p))
            updatemin!(mincosts, tmpcosts)
            mincosts[p] = 0
        end
    end
    return X[:, iseeds]
end

"""
    initseeds(X, k)

taken from Clustering.jl; returns the centers
""" 
function initseeds(X::Matrix{T}, k::Int) where T <: Real
    n = size(X, 2)
    iseeds = zeros(Int, k)
    # randomly pick the first center
    p = rand(1:n)
    iseeds[1] = p
  
    if k > 1
        mincosts = Distances.colwise(Euclidean(), X, view(X,:,p))
        mincosts[p] = 0
  
        # pick remaining (with a chance proportional to mincosts)
        tmpcosts = zeros(n)
        for j = 2:k
            p = wsample(1:n, mincosts)
            iseeds[j] = p
            # update mincosts
            c = view(X,:,p)
            Distances.colwise!(tmpcosts, Euclidean(), X, view(X,:,p))
            updatemin!(mincosts, tmpcosts)
            mincosts[p] = 0
        end
    end
    return iseeds
end
  
function seed2center(iseeds::Array{Int},X::Matrix{T}) where T <: Real
    return(X[:,iseeds])
end
  
function seed2class(iseeds::Array{Int},X::Matrix{T}) where T <: Real
    points = size(X,2)
    class = zeros(Int, points)
    dist = pairwise(Euclidean(), X[:, iseeds], X) # fetch distances
    for point = 1:points
        class[point] = argmin(dist[:, point]) # closest center
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
  
  
#taken from clustering.jl
function ARI(c1,c2)
    # rand_index - calculates Rand Indices to compare two partitions
    # (AR, RI, MI, HI) = rand(c1,c2), where c1,c2 are vectors listing the
    # class membership, returns the "Hubert & Arabie adjusted Rand index".
    # (AR, RI, MI, HI) = rand(c1,c2) returns the adjusted Rand index,
    # the unadjusted Rand index, "Mirkin's" index and "Hubert's" index.
    #
    # See L. Hubert and P. Arabie (1985) "Comparing Partitions" Journal of
    # Classification 2:193-218
  
    c = counts(c1,c2,(1:maximum(c1),1:maximum(c2))) # form contingency matrix

    n = round(Int,sum(c))
    nis = sum(sum(c,2).^2)        # sum of squares of sums of rows
    njs = sum(sum(c,1).^2)        # sum of squares of sums of columns

    t1 = binomial(n,2)            # total number of pairs of entities
    t2 = sum(c.^2)                # sum over rows & columnns of nij^2
    t3 = .5*(nis+njs)
  
    # Expected index (for adjustment)
    nc = (n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1))
  
    A = t1+t2-t3;        # no. agreements
    D = -t2+t3;          # no. disagreements
  
    if t1 == nc
        # avoid division by zero; if k=1, define Rand = 0
        ARI = 0
    else
        # adjusted Rand - Hubert & Arabie 1985
        ARI = (A-nc)/(t1-nc)
    end
  
    #RI = A/t1            # Rand 1971      # Probability of agreement
    #MI = D/t1            # Mirkin 1970    # p(disagreement)
    #HI = (A-D)/t1        # Hubert 1977    # p(agree)-p(disagree)
  
    return ARI
end
  
  
function VI(k1::Int, a1::AbstractVector{Int},
            k2::Int, a2::AbstractVector{Int})
    # check input arguments
    n = length(a1)
    length(a2) == n || throw(DimensionMismatch("Inconsistent array length."))

    # count & compute probabilities
    p1 = zeros(k1)
    p2 = zeros(k2)
    P = zeros(k1, k2)
  
    for i = 1:n
        @inbounds l1 = a1[i]
        @inbounds l2 = a2[i]
        p1[l1] += 1.0
        p2[l2] += 1.0
        P[l1, l2] += 1.0
    end
  
    for i = 1:k1
        @inbounds p1[i] /= n
    end
    for i = 1:k2
        @inbounds p2[i] /= n
    end
    for i = 1:(k1*k2)
        @inbounds P[i] /= n
    end
  
    # compute variation of information
    H1 = entropy(p1)
    H2 = entropy(p2)
  
    I = 0.0
    for j = 1:k2, i = 1:k1
        pi = p1[i]
        pj = p2[j]
        pij = P[i,j]
        if pij > 0.0
            I += pij * log(pij / (pi * pj))
        end
    end
  
    return H1 + H2 - I * 2.0
end
  
# X is the d x n matrix of data; center is the d x k centers
function kmeans_obj(center,X)
    sum( minimum(pairwise(Euclidean(), center, X),1) )
end
  
function kgen_obj(center,X,s)
    temp = pairwise(Euclidean(), center,X)
    res = 0.0
    for i = 1:size(X,2)
        res += powmean(temp[:,i],s)
    end
    return res
end
  
"""
      randindex(a, b) -> NTuple{4, Float64}
  Compute the tuple of Rand-related indices between the clusterings `c1` and `c2`.
  `a` and `b` can be either [`ClusteringResult`](@ref) instances or
  assignments vectors (`AbstractVector{<:Integer}`).
  Returns a tuple of indices:
    - Hubert & Arabie Adjusted Rand index
    - Rand index (agreement probability)
    - Mirkin's index (disagreement probability)
    - Hubert's index (``P(\\mathrm{agree}) - P(\\mathrm{disagree})``)
  # References
  > Lawrence Hubert and Phipps Arabie (1985). *Comparing partitions.*
  > Journal of Classification 2 (1): 193–218
  > Meila, Marina (2003). *Comparing Clusterings by the Variation of
  > Information.* Learning Theory and Kernel Machines: 173–187.
  """
function randindex(a, b)
    c = counts(a, b)
  
    n = sum(c)
    nis = sum(abs2, sum(c, dims=2))        # sum of squares of sums of rows
    njs = sum(abs2, sum(c, dims=1))        # sum of squares of sums of columns
  
    t1 = binomial(n, 2)                    # total number of pairs of entities
    t2 = sum(abs2, c)                      # sum over rows & columnns of nij^2
    t3 = .5*(nis+njs)
  
    # Expected index (for adjustment)
    nc = (n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1))
  
    A = t1+t2-t3;        # agreements count
    D = -t2+t3;          # disagreements count
  
    if t1 == nc
        # avoid division by zero; if k=1, define Rand = 0
        ARI = 0
    else
        # adjusted Rand - Hubert & Arabie 1985
        ARI = (A-nc)/(t1-nc)
    end
  
    RI = A/t1            # Rand 1971      # Probability of agreement
    MI = D/t1            # Mirkin 1970    # p(disagreement)
    HI = (A-D)/t1        # Hubert 1977    # p(agree)-p(disagree)

    return (ARI, RI, MI, HI)
end