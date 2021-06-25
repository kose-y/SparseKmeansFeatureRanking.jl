using SKFR
using Test
using Distances
using StatsBase
using Statistics
using Random
include("ref/k_generalized_source.jl")
include("ref/sparse.jl")
include("ref/sparsekpod.jl")

@testset "SKFR.jl" begin
    @testset "sparse" begin
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
end
