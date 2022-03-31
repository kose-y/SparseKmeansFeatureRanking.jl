
"""
    sparsekmeans1(X, class::Vector{Int}, classes::Int, sparsity::Int)

Implements sparse kmeans clustering. The variable class should
enter with an initial guess of the classifications.

# Input:

* `X`: n by p AbstractImputedMatrix
* `sparsity`: sparsity level (total nonzeros)
* `fast_impute`: good only if used via `sparsekpod` function. 

# Ouptut: 

* Class labels
* class centers (p by k)
* informative feature indices (vector).
* within-cluster sum of squares (WSS), a vector.
* total sum of squares (TSS)
"""
function sparsekmeans1(X::AbstractImputedMatrix{T}, sparsity::Int; 
    normalize::Bool=!X.renormalize, max_iter=Inf, fast_impute=false) where T <: Real

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
    while switched # iterate until class assignments stabilize
        if cnt >= max_iter
            break
        end
        # if cnt > 0
        #     get_centers!(X)
        # end
        # compute the sparsity criterion
        X.centers .= X.centers_tmp
        @assert !any(isnan.(X.centers))
        @tullio (X.criterion)[j] = (X.members)[kk] * (X.centers)[j, kk] ^ 2
        # for j = 1:p # compute the sparsity criterion
        #     X.criterion[j] = 0
        #     for kk = 1:k
        #         X.criterion[j] = X.criterion[j] + X.members[kk] * X.centers[j, kk] ^ 2
        #     end
        # end
        # Gather the criterion to the master node
        J = partialsortperm(X.criterion, 1:sparsity, rev=true)
        nonselected = setdiff(wholevec, J)
        fill!(@view(X.centers[nonselected, :]), zero(T))
        #center[:, J] = zeros(length(J),classes)
        selectedvec .= J
        if fast_impute
            @tturbo X.clusters_stable .= X.clusters
            @tturbo X.centers_stable .= X.μ .+ X.centers .* X.σ
        end
        get_distances_to_center!(X, selectedvec)
        c, switched = get_clusters!(X)
        cnt += 1
    end
    println("cnt of sparse1:", cnt)

    # now calculating the WSS and TSS; used in the permutation test and sparse kpod

    nth = nthreads()
    WSSval = zeros(T, k, nth)
    @threads for t in 1:nth
        j = t
        while j <= p
            for i in 1:n
                kk = X.clusters[i]
                @inbounds WSSval[kk, t] += (X[i, j] - X.centers[j, kk]) ^ 2
            end
            j += nth
        end
    end
    WSSval = sum(WSSval; dims=2)[:]

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

    return (X.clusters, X.centers, selectedvec, WSSval, TSSval)
end

"""
    sparsekmeans2(X, class::Vector{Int}, classes::Int, sparsity::Int)

Implements sparse kmeans clustering and feature selection within
each class. The variable class should enter with an initial guess of
the classifications.

# Input:

* `X`: p by n.
* `class`: initial class, a vector.
* `classes`: number of classes
* `sparsity`: sparsity level (nonzeros per class)

# Ouptut: 

* Class labels
* class centers (p by k)
* informative feature indices (vector).
* within-cluster sum of squares (WSS), a vector.
* total sum of squares (TSS)
"""
function sparsekmeans2(X::AbstractImputedMatrix{T}, sparsity::Int;
    normalize::Bool=!X.renormalize, fast_impute=false) where T <: Real
  
    (n, p) = size(X)
    k = classes(X)
    selectedvec = zeros(Int, k, sparsity)
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
    while switched # iterate until class assignments stabilize
        X.centers .= X.centers_tmp
        # get_centers!(X, selectedvec_)
        for kk = 1:k 
            if X.members[kk] > 0 # set the smallest center components to 0
                J = partialsortperm(X.centers[:, kk] .^ 2 .* X.members[kk], 1:sparsity, rev = true, by = abs)
                nonselected = setdiff(wholevec, J)
                fill!(@view(X.centers[nonselected, kk]), zero(T))
                selectedvec[kk,:] .= J
            end
        end
        selectedvec_ = sort!(unique(selectedvec[:]))
        if fast_impute
            @tturbo X.clusters_stable .= X.clusters
            @tturbo X.centers_stable .= X.μ .+ X.centers .* X.σ
        end
        get_distances_to_center!(X, selectedvec_)
        _, switched = get_clusters!(X)
    end

    nth = nthreads()
    WSSval = zeros(T, k, nth)
    @threads for t in 1:nth
        j = t
        while j <= p
            for i in 1:n
                kk = X.clusters[i]
                @inbounds WSSval[kk, t] += (X[i, j] - X.centers[j, kk]) ^ 2
            end
            j += nth
        end
    end
    WSSval = sum(WSSval; dims=2)[:]

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

    return (X.clusters, X.centers,selectedvec,WSSval,TSSval)
end

function sparsekmeans_repeat(X::AbstractImputedMatrix{T}, sparsity::Int;
    normalize::Bool=!X.renormalize, ftn = sparsekmeans1, iter::Int = 20, max_inner_iter=20) where T <: Real
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
        reinitialize!(X)
        # By definition, TSS should be the same across initializations. 
        (newclusts, newcenterout, newselectedvec, newWSS, _) = ftn(X, sparsity; normalize=normalize, max_iter=max_inner_iter, fast_impute=true)
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

function sparsekmeans_path(X::AbstractImputedMatrix{T}, sparsity_list = Vector{Int};
    normalize::Bool=!X.renormalize, ftn = sparsekmeans1, iter::Int = 5, max_inner_iter=20) where T <: Real
    n, p = size(X)
    k = classes(X)
    selectedvecs = Vector{Int}[]
    bestclusters = Matrix{Int}(undef, n, length(sparsity_list))
    @assert issorted(sparsity_list; rev=true) "sparsity list must be in decreasing order"
    bestcluster, bestcenters, selectedvec, WSS, TSS, fit = sparsekmeans_repeat(X, sparsity_list[1]; 
        normalize=normalize, ftn = ftn, iter = iter, max_inner_iter=max_inner_iter)
    push!(selectedvecs, selectedvec)
    bestclusters[:, 1] .= bestcluster

    for i in 2:length(sparsity_list)
        (newclusts, _, newselectedvec, _, _) = ftn(X, sparsity_list[i]; 
            normalize=!X.renormalize, max_iter=max_inner_iter, fast_impute=true)
        push!(selectedvecs, newselectedvec)
        bestclusters[:, i] .= newclusts
    end

    return (bestclusters, selectedvecs)
end