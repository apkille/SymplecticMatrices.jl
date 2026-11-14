@testitem "takagi" begin
    using SymplecticMatrices
    using LinearAlgebra: Diagonal, Symmetric, I, qr

    istakagi(x, F; atol = 1e-10) = isapprox(F.Q * Diagonal(F.S) * transpose(F.Q), x, atol = atol) &&
        isapprox(F.Q' * F.Q, I, atol = atol) && issorted(F.S, rev = true)

    @testset "random objects" begin
        for n in 1:5
            R_sym = Symmetric(randn(Float64, n, n))
            C_sym = Symmetric(randn(ComplexF64, n, n))

            Fr_sym = takagi(R_sym)
            Qr_sym, Sr_sym = Fr_sym
            @test Fr_sym.Q == Qr_sym && Fr_sym.S == Sr_sym
            @test istakagi(R_sym, Fr_sym)

            Fc_sym = takagi(C_sym)
            Qc_sym, Sc_sym = Fc_sym
            @test Fc_sym.Q == Qc_sym && Fc_sym.S == Sc_sym
            @test istakagi(C_sym, Fc_sym)
        end
    end

    @testset "degenerate and rank-deficient" begin
        U = Matrix(qr(randn(ComplexF64, 6, 6)).Q)
        x = U * Diagonal([2.0, 2.0, 2.0, 1.0, 0.0, 0.0]) * transpose(U)
        F = takagi(x)
        @test istakagi(x, F)
        @test isapprox(F.S, [2.0, 2.0, 2.0, 1.0, 0.0, 0.0], atol = 1e-10)
    end

    @testset "element types" begin
        x = Complex{Int}[1+2im 3im; 3im 4]
        F = takagi(x)
        @test eltype(F.Q) == ComplexF64 && eltype(F.S) == Float64
        @test istakagi(x, F)
        y = Symmetric(randn(Float32, 4, 4))
        G = takagi(y)
        @test eltype(G.Q) == ComplexF32 && eltype(G.S) == Float32
    end

    @testset "takagi close to square root branch cut" begin
        A = ComplexF64[
            -1.1227070369929479+1.9049934518259515e-14im   0.54716967947699913-1.6918680165221716e-14im  -0.21702870484346273+2.3221243940176302e-14im  -0.17908628915764285+8.6808805652760314e-16im;
            0.54716967947699913-1.6918680165221716e-14im -1.9945498708490428+5.894389714343089e-14im   0.1461586192820174-2.067359484355855e-14im  -0.8660130476386011+4.9824533574299278e-14im;
            -0.21702870484346273+2.3221243940176302e-14im  0.1461586192820174-2.067359484355855e-14im -1.5483672649972662+1.423881035981978e-15im -0.30029509809050964+2.8652382211597224e-14im;
            -0.17908628915764285+8.6808805652760314e-16im -0.8660130476386011+4.9824533574299278e-14im -0.30029509809050964+2.8652382211597224e-14im -2.8343758271607444+1.0197862728475361e-13im
        ]

        F = takagi(A)
        @test istakagi(A, F)
    end
end