using SKFR
using Test
using Distances
using SnpArrays
using StatsBase
using Statistics
using Random
using Missings
using BenchmarkTools
include("ref/k_generalized_source.jl")
include("ref/sparse.jl")

@testset "nonmissing" begin
    (features, cases) = (100, 300);
    (classes, sparsity)  = (3, 33);
    X = randn(features, cases);
    (m, n) = (div(features, 3), 2 * div(features, 3));
    (r, s) = (div(cases, 3) + 1, 2 * div(cases, 3));
    X[1:m, r:s] = X[1:m, r:s] .+ 1.0;
    X[1:m, s + 1:end] = X[1:m, s + 1:end] .+ 2.0;
    # getting true labels
    # k means++ initial seeding
    # Random.seed!(16962)
    rng = MersenneTwister(16962)
    class =initclass(X,classes; rng=rng);
    X1=copy(X);
    X2=copy(X);
    
    class1=copy(class);
    class2=copy(class);
    # classout is the output cluster labels, center is the output cluster centers,
    # selectedvec contains the top s most informative features.
    #WSSval= within cluster sum of squares; TSSval=total sum of squares


    @time (classout1, center1, selectedvec1, WSSval1, TSSval1) = ref_sparsekmeans1(X1, class1, classes, sparsity);
    @time (classout2, center2, selectedvec2, WSSval2, TSSval2) = ref_sparsekmeans2(X2, class2, classes, sparsity);

    # Random.seed!(16962)
    rng = MersenneTwister(16962)
    IM = SKFR.get_imputed_matrix(collect(transpose(X)), 3; rng=rng)
    X1 = deepcopy(IM)
    X2 = deepcopy(IM)

    # test correctness of on-the-fly normalization
    SKFR.compute_μ_σ!(IM)

    IM2 = SKFR.get_imputed_matrix(collect(transpose(X)), 3; renormalize=false)
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
                    sum += IM[i, j]
                end
            end 
            @test sum / cnts ≈ IM.centers[j, k]
        end
    end
    # @btime (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1($X1, $sparsity);
    @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(X1, sparsity);
    @time (classout2_, center2_, selectedvec2_, WSSval2_, TSSval2_) = SKFR.sparsekmeans2(X2, sparsity);

    # test sparsekmeans results
    @test all(classout1 .== classout1_)
    @test all(isapprox.(center1, center1_))
    @test all(sort!(selectedvec1_) .== selectedvec1)
    @test WSSval1 ≈ WSSval1_
    @test TSSval1 ≈ TSSval1_
    @test all(classout2 .== classout2_)
    @test all(isapprox.(center2, center2_))
    for i in 1:classes
        sort!(@view(selectedvec2_[i, :]))
    end
    @test all(selectedvec2_ .== selectedvec2)
    @test WSSval2 ≈ WSSval2_
    @test TSSval2 ≈ TSSval2_
end

@testset "SnpArray" begin
    EUR = SnpArray(SnpArrays.datadir("EUR_subset.bed")) # No missing
    EURtrue = convert(Matrix{Float64}, EUR, model=ADDITIVE_MODEL, center=false, scale=false)
    nclusters = 3
    rng = MersenneTwister(263)
    ISM = SKFR.get_imputed_matrix(EUR, nclusters; rng=rng)
    rng = MersenneTwister(263)
    IM = SKFR.get_imputed_matrix(EURtrue, nclusters; rng=rng)
    @time (classout1, center1, selectedvec1, WSSval1, TSSval1) = SKFR.sparsekmeans1(IM, 30);
    # @btime (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1($ISM, 30);
    @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30);        
    @test classout1 == classout1_
    @test all(center1 .≈ center1_)
    @test all(selectedvec1 .== selectedvec1_)
    @test WSSval1 ≈ WSSval1_
    @test TSSval1 ≈ TSSval1_

    ISM = SKFR.get_imputed_matrix(EUR, nclusters; rng=rng)
    @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30);  
    @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30);  
    SKFR.reinitialize!(ISM)
    @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30; squares=false);  
    SKFR.reinitialize!(ISM)
    @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30; squares=false);  
    # @btime begin
    #     ISM = SKFR.ImputedSnpMatrix{Float64}($EUR, $nclusters)
    #     (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30);  
    # end
end

# @testset "StackedSnpArray" begin
#     EUR = SnpArray(SnpArrays.datadir("EUR_subset.bed")) # No missing
#     EURfirsthalf = SnpArrays.filter(SnpArrays.datadir("EUR_subset"), 1:size(EUR, 1), 1:(size(EUR, 2) ÷ 2); des="EUR_subset_1")
#     EURsecondhalf = SnpArrays.filter(SnpArrays.datadir("EUR_subset"), 1:size(EUR, 1), (size(EUR, 2) ÷ 2 + 1):(size(EUR, 2)); des="EUR_subset_2")
#     EUR = SnpArray(SnpArrays.datadir("EUR_subset.bed")) # No missing
#     EURtrue = convert(Matrix{Float64}, EUR, model=ADDITIVE_MODEL, center=false, scale=false)
#     EUR_stacked = StackedSnpArray([EURfirsthalf, EURsecondhalf])
#     nclusters = 3
#     rng = MersenneTwister(263)
#     ISM = SKFR.get_imputed_matrix(EUR_stacked, nclusters; rng=rng)
#     rng = MersenneTwister(263)
#     IM = SKFR.get_imputed_matrix(EURtrue, nclusters; rng=rng)
#     @time (classout1, center1, selectedvec1, WSSval1, TSSval1) = SKFR.sparsekmeans1(IM, 30);
#     # @btime (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1($ISM, 30);
#     @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30);        
#     @test classout1 == classout1_
#     @test all(center1 .≈ center1_)
#     @test all(selectedvec1 .== selectedvec1_)
#     @test WSSval1 ≈ WSSval1_
#     @test TSSval1 ≈ TSSval1_

#     ISM = SKFR.ImputedSnpMatrix{Float64}(EUR, nclusters; rng=rng)
#     @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30);  
#     @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30);  
#     SKFR.reinitialize!(ISM)
#     @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30; squares=false);  
#     SKFR.reinitialize!(ISM)
#     @time (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30; squares=false);     
    
#     rm("EUR_subset_1.bed")
#     rm("EUR_subset_1.fam")
#     rm("EUR_subset_1.bim")

#     rm("EUR_subset_2.bed")
#     rm("EUR_subset_2.fam")
#     rm("EUR_subset_2.bim")
# end



# using Profile
# @testset "alloc" begin
#     EUR = SnpArray(SnpArrays.datadir("EUR_subset.bed")) # No missing
#     nclusters = 3
#     ISM = SKFR.ImputedSnpMatrix{Float64}(EUR, nclusters)
#     # @btime (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1($ISM, 30);
#     (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30);        
#     Profile.clear_malloc_data()
#     ISM = SKFR.ImputedSnpMatrix{Float64}(EUR, nclusters)
#     (classout1_, center1_, selectedvec1_, WSSval1_, TSSval1_) = SKFR.sparsekmeans1(ISM, 30);  

# end