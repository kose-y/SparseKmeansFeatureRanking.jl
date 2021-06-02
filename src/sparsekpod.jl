function assign_clustppSparse(X, k, init_centers,sparsity,
    kmpp_flag= true, max_iter= 20)
    init_classes= get_classes(copy(X'),copy(init_centers'))
    (clusts, centerout,selectedvec,WSS,obj) = sparsekmeans1(copy(X'), init_classes, k, sparsity)
    fit = 1 - (sum(WSS)/obj)
    centers = copy(centerout')
    # Consider dropping this step, or using a lower `max_iter`.
    if kmpp_flag == true
        for iter = 1:max_iter
            classes_kmpp = initclass(copy(X'), k)
            (newclusts, newcenterout,selectedvec,newWSS,newobj) = sparsekmeans1(copy(X'), classes_kmpp, k, sparsity)
            if newobj < obj
                obj = newobj
                clusts = newclusts
                fit = 1 - (sum(newWSS)/newobj)
                centers = copy(newcenterout')
                break
            end
        end
    end
    return(clusts, obj, centers,fit)
end
   
function findMissing(X)
    missing_all=findall(ismissing.(X))
    return(missing_all)
end
   
function initialImpute(X)
    avg = mean(skipmissing(vec(X)))
    X[findall(ismissing.(X))] .= avg
    return(X)
end

"""
    sparsekpod(X, k, sparsity, kmpp_flag::Bool=true, maxiter::Int=20)

sparsekpod function, doing SKFR1 on partially observed data
TRY TO AVOID ANY ADDITIONAL n x p MATRICES.

# Input:

- `X`: n by p data matrix.
- `k`: number of classes.
- `sparsity`: sparsity level.
- `kmpp_flag`: flag for using k-means++ seeding. 
- `maxiter`: max iteration. 

# Output: 

- class labels
- class labels of each iteration
- TSS of each iteration
- fit = 1 - sum(WSS) / obj
- fit of each iteration
"""
function sparsekpod(X, k, sparsity, kmpp_flag::Bool = true,
    maxiter::Int = 20)
   
    n = size(X,1)
    p = size(X,2)
    cluster_vals = zeros(maxiter,n)
    obj_vals = []
    fit = []
    # do not store missing/nonmissing indices. It's also n x p. 
    missingindices = findMissing(X)
    nonmissingindices=setdiff(CartesianIndices(X)[1:end],missingindices)
    # do not create these. Retrieve imputed values on the fly. 
    X_copy = initialImpute(X)
    X_copy = convert(Array{Float64,2}, X_copy)
    # do not require copy of a matrix for these subfunctions.
    # This requires between-sample distances when the features are distributed. 
    init_classes = initclass(copy(X_copy'), k)
    # decide on sparsekmeans1 or sparsekmeans2. maybe use an additional argument? 
    # This requires center-to-sample distances. 
    (clusts, centerout,selectedvec,WSS,obj)= sparsekmeans1(copy(X_copy'), init_classes, k, sparsity)
    # maybe we can do these in-place.
    centers = copy(centerout')
    append!(fit,1 - (sum(WSS)/obj))
    # Avoid creation of `clustMat`.
    clustMat = centers[clusts, :]
    # do it on the fly.
    X_copy[missingindices] = clustMat[missingindices]
    # Write a memory-efficient code for this, or just drop it.
    append!(obj_vals,sum((X[nonmissingindices] .- clustMat[nonmissingindices]).^2))
    cluster_vals[1,:]=clusts
   
    for i = 2:maxiter
        (tempclusts, tempobj, tempcenters,tempfit)= assign_clustppSparse(X_copy, k, centers, sparsity, kmpp_flag)
        clusts = tempclusts
        centers =tempcenters
        append!(fit,tempfit)
        clustMat = centers[clusts,:]
        # don't do this.
        X_copy[missingindices] =clustMat[missingindices]
        # Write a memory-efficient code for this, or just drop it.
        append!(obj_vals,sum((X[nonmissingindices] .- clustMat[nonmissingindices]).^2))
        cluster_vals[i,:] = clusts
        if (all(cluster_vals[i,:] == cluster_vals[i - 1,:]))
            println("Clusters have converged.")
            return(clusts, cluster_vals[1:i,:],obj_vals[1:i],fit[i],fit[1:i])
            break
        end
    end
    return(clusts, cluster_vals[1:i,:],obj_vals[1:i],fit[i],fit[1:i])
end