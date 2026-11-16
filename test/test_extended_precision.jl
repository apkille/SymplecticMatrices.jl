using Test
using LinearAlgebra
using SymplecticMatrices
using GenericLinearAlgebra
using GenericSchur
using MultiFloats

MultiFloats.use_bigfloat_transcendentals()

function check_extended_precision(::Type{T}) where {T}
    tol = 1000 * eps(T)
    S = T[1 1; 0 1]
    V = T[7 2; 2 1]
    V2 = T[3 1 0 0; 1 3 1 0; 0 1 3 1; 0 0 1 3]
    x = T[2 1; 1 -3]
    z = Complex{T}[1+2im 1im; 1im 3]
    w = Complex{T}[-2 1; 1 -3]

    @testset "$T" begin
        F = polar(S)
        @test eltype(F.P) == T
        @test isapprox(F.O * F.P, S, rtol = tol)

        for y in (x, z, w)
            F = takagi(y)
            @test isapprox(F.Q * Diagonal(F.S) * transpose(F.Q), y, rtol = tol)
            @test isapprox(F.Q' * F.Q, I, atol = tol)
        end
        @test isapprox(takagi(x).S, [(sqrt(T(29)) + 1) / 2, (sqrt(T(29)) - 1) / 2], rtol = tol)

        F = williamson(BlockForm(1), V)
        @test isapprox(F.S * V * F.S', Diagonal(repeat(F.spectrum, 2)), rtol = tol)
        @test issymplectic(BlockForm(1), F.S, atol = tol)
        @test isapprox(only(F.spectrum), sqrt(T(3)), rtol = tol)

        F = williamson(BlockForm(2), V2)
        @test isapprox(F.S * V2 * F.S', Diagonal(repeat(F.spectrum, 2)), rtol = tol)
        @test issymplectic(BlockForm(2), F.S, atol = tol)
        @test issorted(F.spectrum)

        F = williamson(PairForm(2), V2)
        @test isapprox(F.S * V2 * F.S', Diagonal(repeat(F.spectrum, inner = 2)), rtol = tol)
        @test issymplectic(PairForm(2), F.S, atol = tol)

        F = blochmessiah(BlockForm(1), S)
        @test isapprox(F.O * Diagonal(vcat(F.values, inv.(F.values))) * F.Q, S, rtol = tol)
        @test isapprox(only(F.values), (1 + sqrt(T(5))) / 2, rtol = tol)

        F = preiwasawa(BlockForm(1), S)
        @test isapprox(F.L * F.P * F.Q, S, rtol = tol)

        F = iwasawa(BlockForm(1), S)
        @test isapprox(F.N * F.A * F.K, S, rtol = tol)
    end
end

setprecision(BigFloat, 128) do
    check_extended_precision(BigFloat)
end
check_extended_precision(Float64x2)