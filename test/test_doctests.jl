@testitem "Doctests" tags=[:doctests] begin
    using Documenter
    using SymplecticMatrices

    DocMeta.setdocmeta!(SymplecticMatrices, :DocTestSetup, :(using SymplecticMatrices, LinearAlgebra); recursive=true)
    doctest(SymplecticMatrices)
end