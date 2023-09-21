using SparseKmeansFeatureRanking
using Documenter

makedocs(;
    modules=[SparseKmeansFeatureRanking],
    authors="Seyoon Ko <kos@ucla.edu> and contributors",
    repo="https://github.com/kose-y/SparseKmeansFeatureRanking.jl/blob/{commit}{path}#L{line}",
    sitename="SparseKmeansFeatureRanking.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kose-y.github.io/SparseKmeansFeatureRanking.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    warnonly=true
)

deploydocs(;
    repo="github.com/kose-y/SparseKmeansFeatureRanking.jl",
    devbranch = "master"
)
