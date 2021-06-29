using SKFR
using Test
using Distances
using StatsBase
using Statistics
using Random
using Missings
include("ref/k_generalized_source.jl")
include("ref/sparse.jl")
include("ref/sparsekpod.jl")

@testset "SKFR.jl" begin
    @testset "nonmissing" begin
        Random.seed!(16962)
        (features, cases) = (100, 300);
        (classes, sparsity)  = (3, 33);
        X = randn(features, cases);
        (m, n) = (div(features, 3), 2 * div(features, 3));
        (r, s) = (div(cases, 3) + 1, 2 * div(cases, 3));
        X[1:m, r:s] = X[1:m, r:s] .+ 1.0;
        X[1:m, s + 1:end] = X[1:m, s + 1:end] .+ 2.0;
        # getting true labels
        # k means++ initial seeding
        Random.seed!(16962)
        class =initclass(X,classes);
        X1=copy(X);
        X2=copy(X);
        
        class1=copy(class);
        class2=copy(class);
        # classout is the output cluster labels, center is the output cluster centers,
        # selectedvec contains the top s most informative features.
        #WSSval= within cluster sum of squares; TSSval=total sum of squares


        @time (classout1, center1, selectedvec1, WSSval1, TSSval1) = ref_sparsekmeans1(X1, class1, classes,m);
        @time (classout2, center2, selectedvec2, WSSval2, TSSval2) = ref_sparsekmeans2(X2, class2, classes,m);

        Random.seed!(16962)
        IM = SKFR.ImputedMatrix{Float64}(collect(transpose(X)), 3)
        X1 = deepcopy(IM)
        X2 = deepcopy(IM)

        # test correctness of on-the-fly normalization
        SKFR.compute_μ_σ!(IM)

        IM2 = SKFR.ImputedMatrix{Float64}(collect(transpose(X)), 3; renormalize=false)
        for j = 1:features # normalize each feature
            # Do it on the fly with SnpArray. 
            IM2[:, j] .= zscore(@view(IM2[:, j]))
        end
        @test all(IM .≈ IM2)


        # test k-means++ class initialization
        @test all(IM.clusters .== class)

        # test center calculation
        for k in 1:classes
            cnts = count(IM.clusters.==k)
        
            for j in 1:features
                sum = 0.0
                for i in 1:cases
                    if IM.clusters[i] == k
                        sum += transpose(X)[i,j]
                    end
                end 
                @test sum / cnts ≈ IM.centers[j, k]
            end
        end

        @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(X1, m);
        @time (classout2_, center2_, selectedvec2_, WSSval2_, TSSval2_) = SKFR.sparsekmeans2(X2, m);

        # test sparsekmeans results
        @test all(classout1 .== classout1_)
        @test all(isapprox.(center1, center1_))
        @test all(selectedvec1_ .== selectedvec1)
        @test WSSval1 ≈ WSSval1_
        @test TSSval1 ≈ TSSval1_
        @test all(classout2 .== classout2_)
        @test all(isapprox.(center2, center2_))
        @test all(selectedvec2_ .== selectedvec2)
        @test WSSval2 ≈ WSSval2_
        @test TSSval2 ≈ TSSval2_
    end
    @testset "kpod" begin
        #Random.seed!(16962)
        (features, cases) = (100, 300);
        (classes, sparsity)  = (3, 33);
        X = randn(features, cases);
        (m, n) = (div(features, 3), 2 * div(features, 3));
        (r, s) = (div(cases, 3) + 1, 2 * div(cases, 3));
        X[1:m, r:s] = X[1:m, r:s] .+ 1.0;
        X[1:m, s + 1:end] = X[1:m, s + 1:end] .+ 2.0;

        truelabels=[];
        class1labels=ones(100);
        append!(truelabels,class1labels);
        class2labels=ones(100)*2;
        append!(truelabels,class2labels);
        class3labels=ones(100)*3;
        append!(truelabels,class3labels);

        # replacing 10% of entries at random
        missingix=sample(1:features*cases,Int(features*cases*0.1),replace=false)
        y = convert(Array{Union{Missing,Float64},2}, X)
        # y is the partially observed version of X above
        y[ CartesianIndices(y)[missingix]]=missings(Float64, length(missingix))

        #Random.seed!(16962)
        missingindices = findMissing(y)
        nonmissingindices=setdiff(CartesianIndices(y)[1:end],missingindices)
        println(size(y))
        X_copy = initialImpute(y)
        X_copy=convert(Array{Float64,2}, X_copy)
        for l in 1:10
            #Random.seed!(77 + l)
            init_classes = initclass(copy(X_copy), classes)
            @time (classout3,aa,bb,cc,dd)=ref_sparsekpod(copy(y'),classes,m)
            arisparse3=randindex(classout3, convert(Array{Int64,1},truelabels))
            println("ARI of sparsekpod (ref): ",arisparse3[1])
        end

        #Random.seed!(16962)
        y = copy(X)
        y[CartesianIndices(y)[missingix]] .= NaN
        y = collect(transpose(y))
        IM = SKFR.ImputedMatrix{Float64}(y, classes)

        # @test all(init_classes .== IM.clusters)


        ## The first output argument is the cluster labels, and the rest are not of importance in this example.


        y = copy(X)
        y[CartesianIndices(y)[missingix]] .= NaN
        y = collect(transpose(y))
        for l in 1:10
            #Random.seed!(77 + l)
            @time (classout3_,aa_,bb_,cc_,dd_)=SKFR.sparsekpod(y,classes,m, false, 1)
            arisparse3=randindex(classout3_, convert(Array{Int64,1},truelabels))
            println("ARI of sparsekpod (new): ",arisparse3[1])
        end



        # println(classout3_)
        # println(aa_)
        # println(bb_)
        # println(cc_)
        # println(dd_)
        #(clusts, cluster_vals[:,1:i],obj_vals[1:i],fit[i],fit[1:i])

    end
end
