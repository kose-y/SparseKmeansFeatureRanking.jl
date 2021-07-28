
"""
    sparsekmeans1(X, class::Vector{Int}, classes::Int, sparsity::Int)

Implements sparse kmeans clustering. The variable class should
enter with an initial guess of the classifications.

# Input:

* `X`: n by p AbstractImputedMatrix
* `sparsity`: sparsity level (total nonzeros)

# Ouptut: 

* Class labels
* class centers (p by k)
* informative feature indices (vector).
* within-cluster sum of squares (WSS), a vector.
* total sum of squares (TSS)
"""
function sparsekmeans1(X::AbstractImputedMatrix{T}, sparsity::Int; 
    normalize::Bool=!X.renormalize) where T <: Real

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
        #compute_μ_σ!(X)
    end
    switched = true
    selectedvec = zeros(Int, sparsity)
    cnt = 0
    while switched # iterate until class assignments stabilize
        get_centers!(X)
        for j = 1:p # compute the sparsity criterion
            X.criterion[j] = 0
            for kk = 1:k
                X.criterion[j] = X.criterion[j] + X.members[kk] * X.centers[j, kk]^2
            end
        end
        # Gather the criterion to the master node
        # find the (p-s) least informative features and setting them to 0
        J = partialsortperm(X.criterion,1:(p-sparsity),rev=false)
        # TODO: Could be faster the other way...(choose top `sparsity` variables)
        fill!(@view(X.centers[J, :]), zero(T))
        #center[:, J] = zeros(length(J),classes)
        selectedvec .= sort!(setdiff(wholevec,J))
        get_distances_to_center!(X, selectedvec)
        c, switched = get_clusters!(X)
        cnt += 1
    end
    println("cnt of sparse1:", cnt)
    # now calculating the WSS and TSS; used in the permutation test and sparse kpod
    WSSval = zeros(T, k)
    for j in 1:p
        @inbounds for i in 1:n
            kk = X.clusters[i]
            WSSval[kk] += (X[i, j] - X.centers[j, kk]) ^ 2
        end
    end

    TSSval = zero(T)
    @inbounds for j in 1:p
        m = mean(@view(X[:, j]))
        for i in 1:n
            TSSval += (X[i, j] - m) ^ 2
        end
    end
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
    normalize::Bool=!X.renormalize) where T <: Real
  
    (n, p) = size(X)
    k = classes(X)
    selectedvec = zeros(Int, k, sparsity)
    wholevec=1:p
    if normalize
        for j = 1:p # normalize each feature
            X[:, j] .= zscore(@view(X[:, j]))
        end
    else
        #compute_μ_σ!(X)
    end
    switched = true
    while switched # iterate until class assignments stabilize

        get_centers!(X)
        for kk = 1:k 
            if X.members[kk] > 0 # set the smallest center components to 0
                J = partialsortperm(X.centers[:, kk] .^ 2 .* X.members[kk], 1:(p - sparsity), by = abs)
                fill!(@view(X.centers[J, kk]), zero(T))
                selectedvec[kk,:] .= setdiff(wholevec,J)
            end
        end
        selectedvec_ = sort!(unique(selectedvec[:]))
        get_distances_to_center!(X, selectedvec_)
        _, switched = get_clusters!(X)
    end

    WSSval = zeros(T, k)
    for j in 1:p
        @inbounds for i in 1:n
            kk = X.clusters[i]
            WSSval[kk] += (X[i, j] - X.centers[j, kk]) ^ 2
        end
    end

    TSSval = zero(T)
    tmp = zero(T)
    @inbounds for j in 1:p
        m = mean(@view(X[:, j]))
        for i in 1:n
            tmp = (X[i, j] - m) ^ 2
            TSSval += tmp
        end
    end
    return (X.clusters, X.centers,selectedvec,WSSval,TSSval)
end

function sparsekmeans_repeat(X::AbstractImputedMatrix{T}, sparsity::Int;
    normalize::Bool=!X.renormalize, ftn = sparsekmeans1, iter::Int = 20) where T <: Real
    n, p = size(X)
    k = classes(X)
    (clusts, centers, selectedvec, WSS, obj) = ftn(X, sparsity; normalize=normalize)
    X.bestclusters .= X.clusters
    X.bestcenters .= X.centers
    fit = 1 - (sum(WSS)/obj)
    println("Iteration 1, fit: ", fit)
    #centers = copy(centerout')
    # Consider dropping this step, or using a lower `max_iter`.
    for iter = 2:iter
        reinitialize!(X)
        (newclusts, newcenterout, newselectedvec, newWSS, newobj) = ftn(X, sparsity; normalize=normalize)
        newfit = 1 - (sum(newWSS)/obj)
        println("Iteration $iter, fit: ", newfit)
        if fit < newfit
            println("bestcluster updated")
            obj = newobj
            WSS = newWSS
            X.bestclusters .= X.clusters
            fit = newfit
            X.bestcenters .= X.centers
            selectedvec .= newselectedvec
        end
    end
    X.clusters .= X.bestclusters
    X.centers .= X.bestcenters
    return(X.bestclusters, X.bestcenters, selectedvec, WSS, obj, fit)
end