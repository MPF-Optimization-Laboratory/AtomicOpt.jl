using Documenter, AtomicSets

makedocs(;
    modules=[AtomicSets],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "PolarCalculus" => "calculus.md"
    ],
    repo="https://github.com/mpf/AtomicSets.jl/blob/{commit}{path}#L{line}",
    sitename="AtomicSets.jl",
    authors="Michael P. Friedlander",
)
