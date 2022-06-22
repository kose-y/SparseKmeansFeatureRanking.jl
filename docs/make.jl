using SKFR
using Documenter

makedocs(;
    modules=[SKFR],
    authors="Seyoon Ko <kos@ucla.edu> and contributors",
    repo="https://github.com/kose-y/SKFR.jl/blob/{commit}{path}#L{line}",
    sitename="SKFR.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kose-y.github.io/SKFR.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/OpenMendel/SKFR.jl",
    devbranch = "master"
)