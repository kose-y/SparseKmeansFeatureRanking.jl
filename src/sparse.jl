
"""
    sparsekmeans1(X, class::Vector{Int}, classes::Int, sparsity::Int)

Implements sparse kmeans clustering. The variable class should
enter with an initial guess of the classifications.

# Input:

* `X`: n by p ImputedMatrix
* `sparsity`: sparsity level (total nonzeros)

# Ouptut: 

* Class labels
* class centers (p by k)
* informative feature indices (vector).
* within-cluster sum of squares (WSS), a vector.
* total sum of squares (TSS)
"""
function sparsekmeans1(X::ImputedMatrix{T}, sparsity::Int; 
    normalize::Bool=true) where T <: Real

    n, p = size(X)
    k = classes(X)
    fill!(X.members, zero(Int))
    fill!(X.criterion, zero(T))

    distance = zeros(T, n, k)
    wholevec=1:p
    if normalize
        for j = 1:p # normalize each feature
            # Do it on the fly with SnpArray. 
            X[:, j] .= zscore(@view(X[:, j]))
        end
    end
    switched = true
    selectedvec = zeros(sparsity)
    while switched # iterate until class assignments stabilize
        get_centers!(X.centers, X.members, X, X.clusters)
        for j = 1:p # compute the sparsity criterion
            X.criterion[j] = 0
            for kk = 1:k
                X.criterion[j] = X.criterion[j] + X.members[kk] * X.centers[j, kk]^2
            end
        end
        # Gather the criterion to the master node
        #J = partialsortperm(criterion, 1:sparsity, rev = true)
        # find the (p-s) least informative features and setting them to 0
        J = partialsortperm(X.criterion,1:(p-sparsity),rev=false)
        fill!(@view(X.centers[J, :]), zero(T))
        #center[:, J] = zeros(length(J),classes)
        selectedvec = setdiff(wholevec,J)
        # TODO: distribute this.
        fill!(distance, zero(T))
        @inbounds for kk in 1:k
            for j in 1:p
                for i in 1:n
                    distance[i, kk] += (X[i, j] - X.centers[j, kk]) ^ 2
                end
            end
        end
        distance .= sqrt.(distance)
        #distance = pairwise(Euclidean(), center, X, dims = 2)
        switched = false # update classes
        for i in 1:n
            j = argmin(distance[i, :]) # class of closest center
            if j != X.clusters[i]
                switched = true
                X.clusters[i] = j
            end
        end
    end
    # now calculating the WSS and TSS; used in the permutation test and sparse kpod
    WSSval = zeros(T, k)
    for kk in 1:k
        tempIndex = findall(X.clusters .== kk)
        # do it memory-efficiently or drop it.
        tempX = @view(X[tempIndex, :]) .- transpose(X.centers[:, kk])
        WSSval[kk] = sum(tempX .^ 2)
    end

    #Xtran=transpose(X)
    # Once again, do it memory-efficiently or drop it.
    TSSval = zero(T)
    @inbounds for j in 1:p
        m = mean(X[:, j])
        for i in 1:n
            TSSval += (X[i, j] - m) ^ 2
        end
    end
    # for j=1:p
    #     X[:,j] = X[:,j] .- mean(X[:,j])
    # end
    # TSSval = sum(X .^ 2)
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
function sparsekmeans2(X::ImputedMatrix{T}, sparsity::Int;
    normalize::Bool=true) where T <: Real
  
    (n, p) = size(X)
    # centers are p by k 
    k = classes(X)
    #center = zeros(T, features, classes)
    #members = zeros(Int, classes)
    #criterion = zeros(T, features)
    distance = zeros(T, n, k)
    #distance = zeros(T, classes, cases)
    selectedvec = zeros(T, k, sparsity)
    #selectedvec=zeros(T,classes,sparsity)
    wholevec=1:p
    if normalize
        for j = 1:p # normalize each feature
            X[:, j] .= zscore(@view(X[:, j]))
        end
    end
    switched = true
    while switched # iterate until class assignments stabilize

        get_centers!(X.centers, X.members, X, X.clusters)
        for kk = 1:k 
            if X.members[kk] > 0 # set the smallest center components to 0
                J = partialsortperm(X.centers[:, kk] .^ 2 .* X.members[kk], 1:(p - sparsity), by = abs)
                fill!(@view(X.centers[J, kk]), zero(T))
                selectedvec[kk,:] .= setdiff(wholevec,J)
            end
        end
        switched = false # update classes
        #(j, k) = (p - sparsity + 1, p)
        #distance = pairwise(Euclidean(), center, X, dims = 2)
        fill!(distance, zero(T))
        @inbounds for kk in 1:k
            for j in 1:p
                for i in 1:n
                    distance[i, kk] += (X[i, j] - X.centers[j, kk]) ^ 2
                end
            end
        end
        distance .= sqrt.(distance)

        for i = 1:n
            kk = argmin(distance[i, :]) # class of closest center
            if kk != X.clusters[i]
                switched = true
                X.clusters[i] = kk
            end
        end
    end

    WSSval = zeros(T, k)
    for kk in 1:k
        tempIndex = findall(X.clusters .== kk)
        # do it memory-efficiently or drop it.
        tempX = @view(X[tempIndex, :]) .- transpose(X.centers[:, kk])
        WSSval[kk] = sum(tempX .^ 2)
    end

    TSSval = zero(T)
    @inbounds for j in 1:p
        m = mean(X[:, j])
        for i in 1:n
            TSSval += (X[i, j] - m) ^ 2
        end
    end
    return (X.clusters, X.centers,selectedvec,WSSval,TSSval)
  end