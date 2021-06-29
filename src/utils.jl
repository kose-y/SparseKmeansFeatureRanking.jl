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
  
function compute_μ_σ!(A::ImputedMatrix{T}) where T
    n, p = size(A)
    @inbounds for j in 1:p
        m = zero(T)
        m2 = zero(T)
        cnt = 0
        for i in 1:n
            v = A.data[i,j]
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
end

function get_distances_to_center!(X::ImputedMatrix{T}) where T
    n, p = size(X)
    k = size(X.centers, 2)
    fill!(X.distances, zero(T))
    for kk in 1:k
        for j in 1:p
            @inbounds @fastmath @simd for i in 1:n
                X.distances[i, kk] = X.distances[i, kk] + (X[i, j] - X.centers[j, kk])^2
            end
        end
    end
    @inbounds for idx in eachindex(X.distances)
        X.distances[idx] = sqrt(X.distances[idx])
    end
    X.distances
end

""" 
    get_clusters!(X, center)
"""
function get_clusters!(X::ImputedMatrix{T}) where T
    n, p = size(X)
    k = size(X.centers, 2)
    switched = false
    for i = 1:n
        kk = argmin(@view(X.distances[i, :])) # class of closest center
        if kk != X.clusters[i]
            switched = true
            X.clusters[i] = kk
        end
    end
    return (X.clusters, switched)
end

"""

- centers: p x k
- X: n x p
- class: length-n
"""
function get_centers!(X::ImputedMatrix{T}) where T <: Real
    n, p = size(X)
    k = size(X.centers, 2)
    @assert length(X.clusters) == n
    @assert size(X.centers, 1) == p
    fill!(X.centers_tmp, zero(T))
    fill!(X.members, zero(eltype(X.members)))
    @inbounds for j in 1:p 
        for i in 1:n
            c = X.clusters[i]
            X.centers_tmp[j, c] = X.centers_tmp[j, c] + X[i, j]
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
    X.centers
end

"""
Distances from row s. 
"""
function dists_from_single_row!(dists::Vector{T}, X::AbstractMatrix{T}, s::Int) where T
    n, p = size(X)
    @assert length(dists) == n
    fill!(dists, zero(T))
    @inbounds for j in 1:p
        for i in 1:n
            dists[i] += (X[i, j] - X[s, j]) ^ 2
        end
    end
    dists .= sqrt.(dists)
    dists[s] = zero(T)
    dists
end

"""

- dists: n x k. 
"""
function dists_from_rows!(dists::Matrix{T}, X::AbstractMatrix{T}, iseeds::Vector{Int}) where T
    n, p = size(X)
    k = size(dists, 2)
    @assert size(dists, 1) == n
    fill!(dists, zero(T))
    X_subsample = X[iseeds, :]
    @inbounds for j in 1:p
        for i in 1:n
            for s in 1:length(iseeds)
                dists[i, s] += (X[i, j] - X_subsample[s, j]) ^ 2
            end
        end
    end
    dists .= sqrt.(dists)
    dists
end

"""
    initclass!(class, X, k)

kmeans plusplus initialization for classes, modified from Clustering.jl. 
""" 
function initclass!(class::Vector{Int}, X::AbstractMatrix{T}, k::Int) where T 
    n = size(X, 1)
    iseeds = zeros(Int, k)
    s = rand(1:n)
    iseeds[1] = s

    if k > 1
        mincosts = Vector{T}(undef, n)
        dists_from_single_row!(mincosts, X, s)

        # pick remaining seeds with a chance proportional to mincosts.
        tmpcosts = zeros(n)
        for j = 2:k
            s = wsample(1:n, mincosts)
            iseeds[j] = s
            dists_from_single_row!(tmpcosts, X, s)
            updatemin!(mincosts, tmpcosts)
            mincosts[s] = 0
        end
    end
    dists = Matrix{T}(undef, n, k)
    dists_from_rows!(dists, X, iseeds)
    for i in 1:n
        class[i] = argmin(dists[i, :])
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