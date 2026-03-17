@testitem "householder" begin
    using SymplecticMatrices
    using LinearAlgebra: I, adjoint, inv, lmul!, rmul!

    @testset "predefined householder transformations" begin
        k, n = 2, 4
        v = [1.0, 2.0, 1.0]
        H_block, H_pair = householder(BlockForm(n), k, v), householder(PairForm(n), k, v)
        M_block = [1.  0.    0.   0. 0.  0.   0.   0.;
                   0.  2/3 -2/3 -1/3 0.  0.   0.   0.;
                   0. -2/3 -1/3 -2/3 0.  0.   0.   0.;
                   0. -1/3 -2/3  2/3 0.  0.   0.   0.;
                   0.  0.   0.   0.  1.  0.   0.   0.;
                   0.  0.   0.   0.  0.  2/3 -2/3 -1/3;
                   0.  0.   0.   0.  0. -2/3 -1/3 -2/3;
                   0.  0.   0.   0.  0. -1/3 -2/3  2/3]
        M_pair =  [1. 0.  0.    0.    0.   0.   0.   0.;
                   0. 1.  0.    0.    0.   0.   0.   0.;
                   0. 0.  2/3   0.   -2/3  0.  -1/3  0.;
                   0. 0.  0.    2/3   0.  -2/3  0.  -1/3;
                   0. 0. -2/3   0.   -1/3   0. -2/3  0.;
                   0. 0.  0.   -2/3   0.  -1/3  0.  -2/3;
                   0. 0. -1/3   0.   -2/3   0.  2/3  0.;
                   0. 0.  0.   -1/3   0.  -2/3   0.  2/3]
        @test M_block ≈ Matrix(H_block) && M_pair ≈ Matrix(H_pair)
        @test M_block ≈ Array(H_block) && M_pair ≈ Array(H_pair)
        @test AbstractMatrix{Float32}(M_block) ≈ AbstractMatrix{Float32}(H_block) && AbstractMatrix{Float32}(M_pair) ≈ AbstractMatrix{Float32}(H_pair)

        @test H_block[1,1] == H_block[5,5] == 1.0
        @test H_block[2,2] ≈ H_block[6,6] ≈ H_block[4,4] ≈ H_block[8,8] ≈ 2/3
        @test H_block[3,2] ≈ H_block[7,6] ≈ H_block[2,3] ≈ H_block[6,7] ≈ -2/3
        @test H_block[4,3] ≈ H_block[8,7] ≈ H_block[3,4] ≈ H_block[7,8] ≈ -2/3
        @test H_block[4,2] ≈ H_block[8,6] ≈ H_block[2,4] ≈ H_block[6,8] ≈ -1/3
        @test H_block[3,3] ≈ H_block[7,7] ≈ -1/3
        @test H_block[1,2] == H_block[2,1] == H_block[1,8] == H_block[8,1] == 0.0

        @test  H_pair[1,1] ==  H_pair[2,2] == 1.0
        @test  H_pair[3,3] ≈  H_pair[4,4] ≈  H_pair[7,7] ≈  H_pair[8,8] ≈ 2/3
        @test  H_pair[5,3] ≈  H_pair[4,6] ≈  H_pair[3,5] ≈  H_pair[6,4] ≈ -2/3
        @test  H_pair[5,7] ≈  H_pair[6,8] ≈  H_pair[7,5] ≈  H_pair[8,6] ≈ -2/3
        @test  H_pair[3,7] ≈  H_pair[4,8] ≈  H_pair[7,3] ≈  H_pair[8,4] ≈ -1/3
        @test  H_pair[5,5] ≈  H_pair[6,6] ≈ -1/3
        @test  H_pair[1,2] ==  H_pair[2,1] ==  H_pair[7,8] ==  H_pair[8,7] == 0.0
        @test  H_pair[2,3] ==  H_pair[3,2] ==  H_pair[3,4] ==  H_pair[4,3] == 0.0

        @test  H_block[1]  ==  H_block[37] == 1.0
        @test  H_block[10] ≈  H_block[28] ≈  H_block[46] ≈ H_block[64] ≈ 2/3
        @test  H_block[11] ≈  H_block[18] ≈  H_block[20] ≈ H_block[27] ≈ -2/3
        @test  H_block[47] ≈  H_block[54] ≈  H_block[56] ≈ H_block[63] ≈ -2/3
        @test  H_block[12] ≈  H_block[19] ≈  H_block[26] ≈ -1/3
        @test  H_block[48] ≈  H_block[55] ≈  H_block[62] ≈ -1/3
        @test  H_block[2]  ==  H_block[9]  ==  H_block[8]  == H_block[61] == 0.0

        @test   H_pair[1]  ==   H_pair[10] == 1.0
        @test   H_pair[19] ≈   H_pair[28] ≈   H_pair[55] ≈ H_pair[64] ≈ 2/3
        @test   H_pair[21] ≈   H_pair[30] ≈   H_pair[35] ≈ H_pair[39] ≈ -2/3
        @test   H_pair[44] ≈   H_pair[48] ≈   H_pair[53] ≈ H_pair[62] ≈ -2/3
        @test   H_pair[23] ≈   H_pair[32] ≈   H_pair[37] ≈ -1/3
        @test   H_pair[46] ≈   H_pair[51] ≈   H_pair[60] ≈ -1/3
        @test   H_pair[2]  ==   H_pair[9]  ==   H_pair[11] == H_pair[17] == 0.0
        @test   H_pair[20] ==   H_pair[22]  ==  H_pair[43] == H_pair[50] == 0.0

    end
    @testset "base methods" begin
        n = rand(3:5)
        k = rand(2:5)
        n = 2 * n
        v = rand(n-k+1)
        J, Omega = BlockForm(n), PairForm(n)
        M = rand(Float64, 2n, 2n)

        @inbounds for form in [J, Omega]
            H = householder(form, k, v)
            Hnew = householder(form, k-1, rand(n-k+2))
            @test isequal(H, copy(H)) && isapprox(H, copy(H))
            @test copy(H) ≈ H
            @test size(H) == (2n, 2n) && size(H, 1) == size(H, 2) == 2n
            @test length(H) == 4n^2
            @test axes(H) == (1:2n, 1:2n)
            @test eltype(H) == Float64

            HMat = Matrix(H)
            HMatnew = Matrix(Hnew)
            i, j = rand(1:2n), rand(1:2n)
            ind = rand(1:4n^2)
            @test H[i,j] ≈ HMat[i,j] && H[j,i] ≈ HMat[j,i]
            @test H[i] ≈ HMat[i] && H[j] ≈ HMat[j] && H[ind] ≈ HMat[ind]

            Z = zeros(Float64, 2n, 2n)
            SZ = Symplectic(form, copy(Z))
            copyto!(Z, H)
            copyto!(SZ, H)
            @test isapprox(Z, HMat, atol = 1e-8) && isapprox(SZ, Symplectic(form, HMat), atol = 1e-8)
            @test isapprox(H * M, HMat * M, atol = 1e-6) && isapprox(M * H, M * HMat, atol = 1e-8)
            @test isapprox(Matrix(H * H), HMat * HMat, atol = 1e-8) && isapprox(Symplectic(form, Matrix(Hnew * H)), Symplectic(form, HMatnew * HMat), atol = 1e-8)
            @test isapprox(H * Symplectic(form, M), Symplectic(form, HMat * M), atol = 1e-8) && isapprox(Symplectic(form, M) * H, Symplectic(form, M * HMat), atol = 1e-8)
            @test isapprox(H / M, HMat / M, atol = 1e-8) && isapprox(M / H, M / HMat, atol = 1e-8)
            @test isapprox(Matrix(H / H), HMat / HMat, atol = 1e-8) && isapprox(Symplectic(form, Matrix(Hnew / H)), Symplectic(form, HMatnew / HMat), atol = 1e-8)
            @test isapprox(H / Symplectic(form, M), Symplectic(form, HMat / M), atol = 1e-8) && isapprox(Symplectic(form, M) / H, Symplectic(form, M / HMat), atol = 1e-8)
            @test isapprox(H \ M, HMat \ M, atol = 1e-8) && isapprox(M \ H, M \ HMat, atol = 1e-8)
            @test isapprox(Matrix(H \ H), HMat \ HMat, atol = 1e-8) && isapprox(Symplectic(form, Matrix(Hnew \ H)), Symplectic(form, HMatnew \ HMat), atol = 1e-8)
            @test isapprox(H \ Symplectic(form, M), Symplectic(form, HMat \ M), atol = 1e-8) && isapprox(Symplectic(form, M) \ H, Symplectic(form, M \ HMat), atol = 1e-8)
        end
    end
    @testset "LinearAlgebra methods" begin
        n = rand(3:5)
        k = rand(2:n)
        n = 2 * n
        v = rand(n-k+1)
        J, Omega = BlockForm(n), PairForm(n)

        @inbounds for form in [J, Omega]
            H = householder(form, k, v)
            HMat = Matrix(H)
            @test Matrix(adjoint(H)) ≈ adjoint(Matrix(H))
            @test inv(H) ≈ H
        
            IR, IL = Matrix{Float64}(I, 2n, 2n), Matrix{Float64}(I, 2n, 2n)
            SIR, SIL = Symplectic(form, copy(IR)), Symplectic(form, copy(IL))
            lmul!(H, IL)
            rmul!(IR, H)
            lmul!(H, SIL)
            rmul!(SIR, H)
            @test isapprox(IL, HMat, atol = 1e-8) && isapprox(IR, HMat, atol = 1e-8)
            @test isapprox(SIL, Symplectic(form, HMat), atol = 1e-8) && isapprox(SIR, Symplectic(form, HMat))
        end
    end
end