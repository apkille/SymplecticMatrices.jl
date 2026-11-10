@testitem "symplectic features" begin
    using SymplecticMatrices
    using LinearAlgebra: Diagonal, adjoint, Symmetric

    @testset "random objects" begin

        n = rand(1:5)
        R_sym = Symmetric(rand(Float64, n, n))
        C_sym = Symmetric(rand(ComplexF64, n, n))

        Fr_sym = takagi(R_sym)
        Qr_sym, Sr_sym = takagi(R_sym)
        @test Fr_sym.Q == Qr_sym && Fr_sym.S == Sr_sym
        @test isapprox(Qr_sym * Diagonal(Sr_sym) * transpose(Qr_sym), Matrix{ComplexF64}(R_sym), atol=1e-5)
        
        #= error on LinearAlgebra.jl end
        @test Fc_sym = takagi(C_sym)
        @test Qc_sym, Sc_sym = takagi(C_sym)
        @test Fc_sym.Q == Qc_sym && Fc_sym.S == Sc_sym
        @testisapprox(Qc_sym * Diagonal(Sc_sym) * transpose(Qc_sym), Matrix(C_sym), atol=1e-5)
        =#
    end

    @testset "takagi close to square root branch cut" begin

        A = ComplexF64[
            -1.1227070369929479+1.9049934518259515e-14im   0.54716967947699913-1.6918680165221716e-14im  -0.21702870484346273+2.3221243940176302e-14im  -0.17908628915764285+8.6808805652760314e-16im;
            0.54716967947699913-1.6918680165221716e-14im -1.9945498708490428+5.894389714343089e-14im   0.1461586192820174-2.067359484355855e-14im  -0.8660130476386011+4.9824533574299278e-14im;
            -0.21702870484346273+2.3221243940176302e-14im  0.1461586192820174-2.067359484355855e-14im -1.5483672649972662+1.423881035981978e-15im -0.30029509809050964+2.8652382211597224e-14im;
            -0.17908628915764285+8.6808805652760314e-16im -0.8660130476386011+4.9824533574299278e-14im -0.30029509809050964+2.8652382211597224e-14im -2.8343758271607444+1.0197862728475361e-13im
        ]

        F = takagi(A)
        @test isapprox(F.Q * Diagonal(F.S) * transpose(F.Q), A, atol=1e-5)

    end
end