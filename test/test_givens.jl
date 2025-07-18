@testitem "givens" begin
    using SymplecticMatrices
    using LinearAlgebra: I, adjoint, inv, lmul!, rmul!

    @testset "predefined givens transformations" begin
        θ = rand(Float64)
        c, s = cos(θ), sin(θ)
        G_block, G_pair = givens(BlockForm(4), 2, θ), givens(PairForm(4), 2, θ)
        M_block = [1. 0. 0. 0. 0. 0. 0. 0.;
                   0. c  0. 0. 0. s  0. 0.;
                   0. 0. 1. 0. 0. 0. 0. 0.;
                   0. 0. 0. 1. 0. 0. 0. 0.;
                   0. 0. 0. 0. 1. 0. 0. 0.;
                   0. -s 0. 0. 0. c  0. 0.;
                   0. 0. 0. 0. 0. 0. 1. 0.;
                   0. 0. 0. 0. 0. 0. 0. 1.]
        M_pair =  [1. 0. 0. 0. 0. 0. 0. 0.;
                   0. 1. 0. 0. 0. 0. 0. 0.;
                   0. 0. c  s  0. 0. 0. 0.;
                   0. 0. -s c  0. 0. 0. 0.;
                   0. 0. 0. 0. 1. 0. 0. 0.;
                   0. 0. 0. 0. 0. 1. 0. 0.;
                   0. 0. 0. 0. 0. 0. 1. 0.;
                   0. 0. 0. 0. 0. 0. 0. 1.]
        @test M_block == Matrix(G_block) && M_pair == Matrix(G_pair)
        @test M_block == Array(G_block) && M_pair == Array(G_pair)
        @test AbstractMatrix{Float32}(M_block) == AbstractMatrix{Float32}(G_block) && AbstractMatrix{Float32}(M_pair) == AbstractMatrix{Float32}(G_pair)
        @test G_block[1,1] == G_block[3,3] == G_block[4,4] == 1.0
        @test G_block[5,5] == G_block[7,7] == G_block[8,8] == 1.0
        @test G_block[2,2] == G_block[6,6] == c
        @test G_block[6,2] == -s && G_block[2,6] == s
        @test G_block[1,2] == G_block[2,1] == G_block[7,8] == G_block[8,7] == 0.0
        @test  G_pair[1,1] ==  G_pair[2,2] ==  G_pair[5,5] == 1.0
        @test  G_pair[6,6] ==  G_pair[7,7] ==  G_pair[8,8] == 1.0
        @test  G_pair[3,3] ==  G_pair[4,4] == c
        @test  G_pair[4,3] == -s && G_pair[3,4] == s
        @test  G_pair[1,2] ==  G_pair[2,1] ==  G_pair[7,8] == G_pair[8,7] == 0.0
        @test  G_block[1]  ==  G_block[19] ==  G_block[28] == 1.0
        @test  G_block[37] ==  G_block[55] ==  G_block[64] == 1.0
        @test  G_block[10] ==  G_block[46] == c
        @test  G_block[14] == -s &&  G_block[42] == s
        @test  G_block[2]  ==   G_block[9] ==  G_block[56] == G_block[63] == 0.0
        @test   G_pair[1]  ==   G_pair[10] ==  G_pair[37] == 1.0
        @test   G_pair[46] ==   G_pair[55] ==  G_pair[64] == 1.0
        @test   G_pair[19] ==   G_pair[28] == c
        @test   G_pair[20] == -s &&  G_pair[27] == s
        @test   G_pair[2]  ==    G_pair[9] ==  G_pair[56] ==   G_pair[63] == 0.0
        
    end
    @testset "base methods" begin
        n = rand(3:5)
        k, θ = rand(2:5), rand(Float64)
        n = 2 * n
        J, Omega = BlockForm(n), PairForm(n)
        M = rand(Float64, 2n, 2n)

        @inbounds for form in [J, Omega]
            G = givens(form, k, θ)
            Gnew = givens(form, k-1, θ+0.3)
            @test isequal(G, copy(G)) && isapprox(G, copy(G))
            @test SymplecticGivens(G) == G
            @test copy(G) == G
            @test size(G) == (2n, 2n) && size(G, 1) == size(G, 2) == 2n
            @test length(G) == 4n^2
            @test axes(G) == (1:2n, 1:2n)
            @test eltype(G) == Float64

            GMat = Matrix(G)
            GMatnew = Matrix(Gnew)
            i, j = rand(1:2n), rand(1:2n)
            ind = rand(1:4n^2)
            @test G[i,j] == GMat[i,j] && G[j,i] == GMat[j,i]
            @test G[i] == GMat[i] && G[j] == GMat[j] && G[ind] == GMat[ind]

            Z = zeros(Float64, 2n, 2n)
            SZ = Symplectic(form, copy(Z))
            copyto!(Z, G)
            copyto!(SZ, G)
            @test isapprox(Z, GMat, atol = 1e-8) && isapprox(SZ, Symplectic(form, GMat), atol = 1e-8)
            @test isapprox(G * M, GMat * M, atol = 1e-8) && isapprox(M * G, M * GMat, atol = 1e-8)
            @test isapprox(Matrix(G * G), GMat * GMat, atol = 1e-8) && isapprox(Symplectic(form, Matrix(Gnew * G)), Symplectic(form, GMatnew * GMat), atol = 1e-8)
            @test isapprox(G * Symplectic(form, M), Symplectic(form, GMat * M), atol = 1e-8) && isapprox(Symplectic(form, M) * G, Symplectic(form, M * GMat), atol = 1e-8)
            @test isapprox(G / M, GMat / M, atol = 1e-8) && isapprox(M / G, M / GMat, atol = 1e-8)
            @test isapprox(Matrix(G / G), GMat / GMat, atol = 1e-8) && isapprox(Symplectic(form, Matrix(Gnew / G)), Symplectic(form, GMatnew / GMat), atol = 1e-8)
            @test isapprox(G / Symplectic(form, M), Symplectic(form, GMat / M), atol = 1e-8) && isapprox(Symplectic(form, M) / G, Symplectic(form, M / GMat), atol = 1e-8)
            @test isapprox(G \ M, GMat \ M, atol = 1e-8) && isapprox(M \ G, M \ GMat, atol = 1e-8)
            @test isapprox(Matrix(G \ G), GMat \ GMat, atol = 1e-8) && isapprox(Symplectic(form, Matrix(Gnew \ G)), Symplectic(form, GMatnew \ GMat), atol = 1e-8)
            @test isapprox(G \ Symplectic(form, M), Symplectic(form, GMat \ M), atol = 1e-8) && isapprox(Symplectic(form, M) \ G, Symplectic(form, M \ GMat), atol = 1e-8)
        end
    end
    @testset "LinearAlgebra methods" begin
        n = rand(3:5)
        k, θ = rand(2:n), rand(Float64)
        n = 2 * n
        J, Omega = BlockForm(n), PairForm(n)

        @inbounds for form in [J, Omega]
            G = givens(form, k, θ)
            GMat = Matrix(G)
            @test adjoint(G) == inv(G) == SymplecticGivens(form, k, cos(θ), -sin(θ))
        
            IR, IL = Matrix{Float64}(I, 2n, 2n), Matrix{Float64}(I, 2n, 2n)
            SIR, SIL = Symplectic(form, copy(IR)), Symplectic(form, copy(IL))
            lmul!(G, IL)
            rmul!(IR, G)
            lmul!(G, SIL)
            rmul!(SIR, G)
            @test isapprox(IL, GMat, atol = 1e-8) && isapprox(IR, GMat, atol = 1e-8)
            @test isapprox(SIL, Symplectic(form, GMat), atol = 1e-8) && isapprox(SIR, Symplectic(form, GMat))
        end
    end
end