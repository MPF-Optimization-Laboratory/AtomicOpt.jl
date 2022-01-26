using Documenter

makedocs(;
    modules=[AtomicSets],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "PolarCalculus" => "calculus.md"
    ],
    repo="https://github.com/ZhenanFanUBC/AtomicOpt.jl/blob/{commit}{path}#L{line}",
    sitename="AtomicOpt.jl",
    authors="Zhenan Fan",
)
