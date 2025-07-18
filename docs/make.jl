using Revise # for interactive work on docs
push!(LOAD_PATH,"../src/")

using Documenter
using DocumenterCitations
using SymplecticMatrices

DocMeta.setdocmeta!(SymplecticMatrices, :DocTestSetup, :(using SymplecticMatrices); recursive=true)

function main()
    #bib = CitationBibliography(joinpath(@__DIR__,"src/references.bib"), style=:authoryear)

    makedocs(
    #plugins=[bib],
    doctest = false,
    clean = true,
    sitename = "SymplecticMatrices.jl",
    format = Documenter.HTML(
        assets=["assets/init.js"],
        canonical = "https://apkille.github.io/SymplecticMatrices.jl"
    ),
    modules = [SymplecticMatrices],
    checkdocs = :exports,
    warnonly = false,
    authors = "Andrew Kille",
    pages = [
        "SymplecticMatrices.jl" => "index.md"
    ]
    )

    deploydocs(
        repo = "github.com/apkille/SymplecticMatrices.jl.git"
    )
end

main()
