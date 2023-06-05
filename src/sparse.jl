
"""
    sparsekmeans1(X::AbstractImputedMatrix{T}, sparsity::Int; 
    normalize=!X.renormalize, max_iter=1000, fast_impute=true, squares=true)

Implements sparse kmeans clustering. The variable class should
enter with an initial guess of the classifications.

# Input:

* `X`: `n` by `p` `AbstractImputedMatrix``
* `sparsity`: Sparsity level (total nonzeros)
* `normalize`: Normalize the input matrix before the run. Must be `false` if renormalization is done on-the-fly through `AbstractImputedMatrix`.
* `max_iter`: Maximum number of iterations
* `fast_impute`: Update cluster centers each iteration
* `squares`: Also return within-cluster sum of squares and total sum of squares

# Ouptut: 
* `X.clusters`: Cluster labels for each sample
* `X.centers`: Cluster centers (p by k)
* `selectedvec: Informative feature indices (vector).
* `WSSVal`: within-cluster sum of squares (WSS), a vector. `nothing` if `squares==false`
* `TSSval`: total sum of squares (TSS). `nothing` if `squares==false`.
"""
function sparsekmeans1(X::AbstractImputedMatrix{T}, sparsity::Int; 
    normalize::Bool=!X.renormalize, max_iter=1000, fast_impute=true, squares=true, rng=Random.GLOBAL_RNG) where T <: Real

    begin # initialization
        n, p = size(X)
        k = classes(X)
        fill!(X.members, zero(Int))
        fill!(X.criterion, zero(T))
        wholevec=1:p
        if normalize
            for j = 1:p # normalize each feature
                # Do it on the fly with SnpArray. 
                X[:, j] .= zscore(@view(X[:, j]))
            end
        else
            if !X.fixed_normalization
                compute_μ_σ!(X)
            end
        end
        switched = true
        selectedvec = zeros(Int, sparsity)
        cnt = 0
        get_centers!(X)
        idx = Array{Int}(undef, length(X.criterion))
        blockidx = Array{Int}(undef, length(X.criterion_block))
    end
    # println("allocation in init: $r")
    begin
        while switched # iterate until class assignments stabilize
            if cnt >= max_iter
                break
            end
            X.centers .= X.centers_tmp # load full cluster centers 
            @tullio (X.criterion)[j] = (X.members)[kk] * (X.centers)[j, kk] ^ 2
            fill!(X.criterion_block, zero(T))
            @inbounds for j in 1:p
                X.criterion_block[convert(Int, ceil(j / X.blocksize))] += X.criterion[j]
            end
            begin 
                Jblock = partialsortperm!(blockidx, X.criterion_block, 
                    1:(min(sparsity, length(X.criterion_block))), rev=true)
                @inbounds for jblock in eachindex(Jblock)
                    for k in 1:X.blocksize
                        j = (Jblock[jblock] - 1) * (X.blocksize) + k
                        idx[(jblock - 1) * X.blocksize + k] = j
                    end
                end
                J = @view idx[1:(sparsity * X.blocksize)]

                nonselected = setdiff(wholevec, J) # this one allocates.
                fill!(@view(X.centers[nonselected, :]), zero(T))
                #center[:, J] = zeros(length(J),classes)
                selectedvec = intersect(J, wholevec)
                if fast_impute # Update cluster centers for imputation every iteration.
                    @tturbo X.clusters_stable .= X.clusters
                    @tturbo X.centers_stable .= X.μ .+ X.centers .* X.σ
                end
            end
            get_distances_to_center!(X, selectedvec)
            c, switched = get_clusters!(X)
            cnt += 1
        end
    end
    println("cnt of sparse1:", cnt)

    # now calculating the WSS and TSS; used in the permutation test and sparse kpod

    nth = nthreads()
    if squares # Compute WSS
        WSSval = zeros(T, k)
        for j in 1:p
            for i in 1:n
                kk = X.clusters[i]
                @inbounds WSSval[kk] += (X[i, j] - X.centers[j, kk]) ^ 2
            end
        end
    else
        WSSval = nothing
    end
    # WSSval = sum(WSSval; dims=2)[:]

    if squares # Compute TSS
        TSSparts = zeros(T, nth)
        @threads for t in 1:nth
            j = t
            while j <= p
                m = mean(@view(X[:, j]))
                for i in 1:n
                    @inbounds TSSparts[t] += (X[i, j] - m) ^ 2
                end
                j += nth
            end
        end
        TSSval = sum(TSSparts)
    else 
        TSSval = nothing
    end

    return (X.clusters, X.centers, selectedvec, WSSval, TSSval)
end

"""
    sparsekmeans2(X::AbstractImputedMatrix{T}, sparsity::Int; 
    normalize=!X.renormalize, max_iter=1000, fast_impute=true, squares=true)

Implements sparse kmeans clustering and feature selection within
each class. 

# Input:

* `X`: `n` by `p` `AbstractImputedMatrix``
* `sparsity`: Sparsity level (total nonzeros)
* `normalize`: Normalize the input matrix before the run. Must be `false` if renormalization is done on-the-fly through `AbstractImputedMatrix`.
* `max_iter`: Maximum number of iterations
* `fast_impute`: Update cluster centers each iteration
* `squares`: Also return within-cluster sum of squares and total sum of squares

# Ouptut: 
* `X.clusters`: Cluster labels for each sample
* `X.centers`: Cluster centers (p by k)
* `selectedvec: Informative feature indices (vector).
* `WSSVal`: within-cluster sum of squares (WSS), a vector. `nothing` if `squares==false`
* `TSSval`: total sum of squares (TSS). `nothing` if `squares==false`.
"""
function sparsekmeans2(X::AbstractImputedMatrix{T}, sparsity::Int;
    normalize::Bool=!X.renormalize, max_iter=1000, fast_impute=true, squares=true, rng=Random.GLOBAL_RNG) where T <: Real
  
    (n, p) = size(X)
    k = classes(X)
    selectedvec = zeros(Int, k, sparsity * X.blocksize)
    wholevec=1:p
    if normalize
        for j = 1:p # normalize each feature
            X[:, j] .= zscore(@view(X[:, j]))
        end
    else
        if !X.fixed_normalization
            compute_μ_σ!(X)
        end
    end
    switched = true
    get_centers!(X)
    idx = Array{Int}(undef, length(X.criterion))
    blockidx = Array{Int}(undef, length(X.criterion_block))
    cnt = 0
    while switched # iterate until class assignments stabilize
        if cnt >= max_iter
            break
        end
        X.centers .= X.centers_tmp
        # get_centers!(X, selectedvec_)
        for kk = 1:k 
            if X.members[kk] > 0 # set the smallest center components to 0
                fill!(X.criterion, zero(T))
                @inbounds for j in 1:p
                    (X.criterion)[j] += (X.members)[kk] * (X.centers)[j, kk] ^ 2
                end
                fill!(X.criterion_block, zero(T))
                @inbounds for j in 1:p
                    X.criterion_block[convert(Int, ceil(j / X.blocksize))] += X.criterion[j]
                end
                Jblock = partialsortperm!(blockidx, X.criterion_block, 1:sparsity, rev=true)
                @inbounds for jblock in eachindex(Jblock)
                    for k in 1:X.blocksize
                        j = (Jblock[jblock] - 1) * (X.blocksize) + k
                        idx[(jblock - 1) * X.blocksize + k] = j
                    end
                end
                J = @view idx[1:(sparsity * X.blocksize)]
                nonselected = setdiff(wholevec, J)
                fill!(@view(X.centers[nonselected, kk]), zero(T))
                selectedvec[kk,:] .= J
            end
        end
        selectedvec_ = intersect(sort!(unique(selectedvec[:])), wholevec)
        if fast_impute # Update cluster centers for imputation every iteration.
            @tturbo X.clusters_stable .= X.clusters
            @tturbo X.centers_stable .= X.μ .+ X.centers .* X.σ
        end
        get_distances_to_center!(X, selectedvec_)
        _, switched = get_clusters!(X)
        cnt += 1
    end

    nth = nthreads()
    if squares # Compute WSS
        WSSval = zeros(T, k)
        for j in 1:p
            for i in 1:n
                kk = X.clusters[i]
                @inbounds WSSval[kk] += (X[i, j] - X.centers[j, kk]) ^ 2
            end
        end
    else
        WSSval = nothing
    end
    # WSSval = sum(WSSval; dims=2)[:]

    if squares # Compute TSS
        TSSparts = zeros(T, nth)
        @threads for t in 1:nth
            j = t
            while j <= p
                m = mean(@view(X[:, j]))
                for i in 1:n
                    @inbounds TSSparts[t] += (X[i, j] - m) ^ 2
                end
                j += nth
            end
        end
        TSSval = sum(TSSparts)
    else 
        TSSval = nothing
    end

    return (X.clusters, X.centers,selectedvec,WSSval,TSSval)
end

"""
    sparsekmeans_repeat(X::AbstractImputedMatrix{T}, sparsity::Int; 
    normalize=!X.renormalize, ftn = spareskmeans1, iter=20, max_inner_iter=20)

Repeat sparse k-means clustering `iter` times, and choose the best one, where the WSS is the lowest.

# Input:

* `X`: `n` by `p` `AbstractImputedMatrix``
* `sparsity`: Sparsity level (total nonzeros)
* `normalize`: Normalize the input matrix before each run of inner function. Must be `false` if renormalization is done on-the-fly through `AbstractImputedMatrix`.
* `ftn`: Inner SKFR function (`sparsekmeans1` or `sparsekmeans2`)
* `max_iter`: Maximum number of iterations
* `max_inner_iter`: Maximum number of iterations for each call of `ftn`.

# Ouptut: 
* `X.bestclusters`: Best cluster labels for each sample
* `X.bestcenters`: Best cluster centers (p by k)
* `selectedvec: Informative feature indices (vector).
* `WSS`: within-cluster sum of squares (WSS), a vector.
* `TSS`: total sum of squares (TSS).
* `fit`: `1 - sum(WSS)/TSS`.
"""
function sparsekmeans_repeat(X::AbstractImputedMatrix{T}, sparsity::Int;
    normalize::Bool=!X.renormalize, ftn = sparsekmeans1, iter::Int = 20, max_inner_iter=20, rng=Random.GLOBAL_RNG) where T <: Real
    n, p = size(X)
    k = classes(X)
    (clusts, centers, selectedvec, WSS, TSS) = ftn(X, sparsity; normalize=normalize, max_iter=max_inner_iter, fast_impute=true)
    X.bestclusters .= X.clusters
    X.bestcenters .= X.centers
    fit = 1 - (sum(WSS)/TSS)
    println("Iteration 1, fit: ", fit)
    #centers = copy(centerout')
    # Consider dropping this step, or using a lower `max_iter`.
    for i = 2:iter
        reinitialize!(X; rng=rng)
        # By definition, TSS should be the same across initializations. 
        (newclusts, newcenterout, newselectedvec, newWSS, _) = ftn(X, sparsity; normalize=normalize, max_iter=max_inner_iter, fast_impute=true, squares=true)
        newfit = 1 - (sum(newWSS)/TSS)
        println("Iteration $i, fit: ", newfit)
        if fit < newfit
            println("bestcluster updated")
            WSS = newWSS
            X.bestclusters .= X.clusters
            fit = newfit
            X.bestcenters .= X.centers
            selectedvec .= newselectedvec
        end
    end
    X.clusters .= X.bestclusters
    X.centers .= X.bestcenters
    return (X.bestclusters, X.bestcenters, selectedvec, WSS, TSS, fit)
end

"""
    sparsekmeans_path(X::AbstractImputedMatrix{T}, sparsity_list::Vector{Int}; 
    normalize=!X.renormalize, ftn = spareskmeans1, iter=5, max_inner_iter=20)

Repeat sparse k-means clustering `iter` times with sparsity of `sparsity_list[1]`, and choose the best one. 
Then, run SKFR for each value of `sparsity_list[2:end]` once, warm-starting with the previous cluster assignment.

# Input:

* `X`: `n` by `p` `AbstractImputedMatrix``
* `sparsity`: Sparsity level (total nonzeros)
* `normalize`: Normalize the input matrix before each run of inner function. Must be `false` if renormalization is done on-the-fly through `AbstractImputedMatrix`.
* `ftn`: Inner SKFR function (`sparsekmeans1` or `sparsekmeans2`)
* `max_iter`: Maximum number of iterations
* `max_inner_iter`: Maximum number of iterations for each call of `ftn`.

# Ouptut: 
* `bestclusters`: Cluster assignment for each of the value in `sparsity_list`, an `n` x `length(sparsity_list)` matrix.
* `selectedvecs`: Selected features for each value in `sparsity_list`, a `Vector{Vector{Int}}`.
"""
function sparsekmeans_path(X::AbstractImputedMatrix{T}, sparsity_list = Vector{Int};
    normalize::Bool=!X.renormalize, ftn = sparsekmeans1, iter::Int = 5, max_inner_iter=20, rng=Random.GLOBAL_RNG) where T <: Real
    n, p = size(X)
    k = classes(X)
    selectedvecs = Array{Int}[]
    bestclusters = Matrix{Int}(undef, n, length(sparsity_list))
    @assert issorted(sparsity_list; rev=true) "sparsity list must be in decreasing order"
    bestcluster, bestcenters, selectedvec, WSS, TSS, fit = sparsekmeans_repeat(X, sparsity_list[1]; 
        normalize=normalize, ftn = ftn, iter = iter, max_inner_iter=max_inner_iter, rng=rng)
    push!(selectedvecs, selectedvec)
    bestclusters[:, 1] .= bestcluster

    for i in 2:length(sparsity_list)
        (newclusts, _, newselectedvec, _, _) = ftn(X, sparsity_list[i]; 
            normalize=!X.renormalize, max_iter=max_inner_iter, fast_impute=true, squares=true)
        push!(selectedvecs, newselectedvec)
        bestclusters[:, i] .= newclusts
    end

    return (bestclusters, selectedvecs)
end
